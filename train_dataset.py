import argparse
import torch
import model
import cv2 as cv
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EmoDataset import EMODataset
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import models
from model import Encoder,PerceptualLoss,crop_and_warp_face,get_foreground_mask, CosSimLoss
# from rome_losses import Vgg19 # use vgg19 for perceptualloss 

import mediapipe as mp
# from memory_profiler import profile
import torchvision.transforms as transforms
import os
import torchvision.utils as vutils

from frames_dataset import FramesDataset, DataPrefetcher
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import wandb
os.environ['WANDB_MODE'] = 'offline' # for debug


# # Define the transform for data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])




# Create a directory to save the images (if it doesn't already exist)
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)


face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(False)# this slows thing down - only for debug


'''
Perceptual Loss:

The PerceptualLoss class combines losses from VGG19, VGG Face, and a specialized gaze loss.
It computes the perceptual losses by passing the output and target frames through the respective models and calculating the MSE loss between the features.
The total perceptual loss is a weighted sum of the individual losses.


Adversarial Loss:

The adversarial_loss function computes the adversarial loss for the generator.
It passes the generated output frame through the discriminator and calculates the MSE loss between the predicted values and a tensor of ones (indicating real samples).


Cycle Consistency Loss:

The cycle_consistency_loss function computes the cycle consistency loss.
It passes the output frame and the source frame through the generator to reconstruct the source frame.
The L1 loss is calculated between the reconstructed source frame and the original source frame.


Contrastive Loss:

The contrastive_loss function computes the contrastive loss using cosine similarity.
It calculates the cosine similarity between positive pairs (output-source, output-driving) and negative pairs (output-random, source-random).
The loss is computed as the negative log likelihood of the positive pairs over the sum of positive and negative pair similarities.
The neg_pair_loss function calculates the loss for negative pairs using a margin.


Discriminator Loss:

The discriminator_loss function computes the loss for the discriminator.
It calculates the MSE loss between the predicted values for real samples and a tensor of ones, and the MSE loss between the predicted values for fake samples and a tensor of zeros.
The total discriminator loss is the sum of the real and fake losses.
'''

# @profile
def adversarial_loss(output_frame, discriminator):
    fake_pred = discriminator(output_frame)
    loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    return loss.requires_grad_()

# @profile
def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss.requires_grad_()


def contrastive_loss(output_frame, source_frame, driving_frame, encoder, margin=1.0):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    z_rand = torch.randn_like(z_out, requires_grad=True)

    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]

    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for pos_pair in pos_pairs:
        loss = loss + torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) /
                                (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) +
                                 neg_pair_loss(pos_pair, neg_pairs, margin)))

    return loss

def neg_pair_loss(pos_pair, neg_pairs, margin):
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for neg_pair in neg_pairs:
        loss = loss + torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)
    return loss
# @profile
def discriminator_loss(real_pred, fake_pred):
    real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
    fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss).requires_grad_()

def mse_loss(img1, img2):
    loss = F.mse_loss(img1, img2)
    return loss.mean()


def save_img(inp):
    
    save_list = []
    for img in inp:
        img1 = img.permute(0, 2, 3, 1).cpu().detach().numpy() * 0.5 + 0.5
        img1 = (img1 * 255).astype('uint8')
        img1 = img1[0,:,:,:]
        save_list.append(img1)


    save = np.concatenate(save_list, axis=1)
    
    return save



'''
Perceptual Losses (ℒ_per):
        VGG19 perceptual loss (ℒ_IN)
        VGGFace perceptual loss (ℒ_face)
        Gaze loss (ℒ_gaze)

Adversarial Losses (ℒ_GAN):
        Generator adversarial loss (ℒ_adv)
        Feature matching loss (ℒ_FM)

Cycle Consistency Loss (ℒ_cos)

N.B
Perceptual Loss (w_per): The perceptual loss is often given a higher weight compared to other losses to prioritize the generation of perceptually similar images. A weight of 20 is a reasonable starting point to emphasize the importance of perceptual similarity.
Adversarial Loss (w_adv): The adversarial loss is typically assigned a lower weight compared to the perceptual loss. A weight of 1 is a common choice to balance the adversarial training without overpowering other losses.
Feature Matching Loss (w_fm): The feature matching loss is used to stabilize the training process and improve the quality of generated images. A weight of 40 is a relatively high value to give significant importance to feature matching and encourage the generator to produce realistic features.
Cycle Consistency Loss (w_cos): The cycle consistency loss helps in preserving the identity and consistency between the source and generated images. A weight of 2 is a moderate value to ensure cycle consistency without dominating the other losses.
'''
def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)
    
    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0})
    cosine_loss = CosSimLoss()
    encoder = Encoder(input_nc=3, output_nc=256).to(device)
    total_step = len(dataloader) *cfg.training.base_epochs
    cur_step = 150*len(dataloader)
    scaler = GradScaler()

    # prefetcher = DataPrefetcher(dataloader, device)


    for epoch in range(150, cfg.training.base_epochs):
        print("epoch:", epoch)
        # batch = prefetcher.next()
        for batch in dataloader:
        # while batch is not None:
            cur_step += 1
            source_frame = batch['source'].to(device)
            driving_frame = batch['driving'].to(device)
            random_driving_frame = batch['random_driving'].to(device)

            # Apply face cropping and random warping to the driving frame for losses ONLY!
            # warped_driving_frame =  crop_and_warp_face(driving_frame, pad_to_original=True)
                    
            # Train generator
            optimizer_G.zero_grad()
            with autocast():
                output_frame, pose2drive, dyn2source = Gbase(source_frame, driving_frame) 
                output_frame_random = Gbase(source_frame, random_driving_frame, same_subject=False)


                # 256 x 256 - Resize output_frame to match the driving_frame size
                # output_frame = F.interpolate(output_frame, size=(256, 256), mode='bilinear', align_corners=False)

                # Obtain the foreground mask for the target image
                foreground_mask = get_foreground_mask(driving_frame)
                
                # Move the foreground mask to the same device as output_frame
                foreground_mask = foreground_mask.to(output_frame.device)

                # Multiply the predicted and target images with the foreground mask
                masked_predicted_image = output_frame * foreground_mask
                masked_target_image = driving_frame * foreground_mask
                            
                                        
                # Calculate perceptual losses
                perceptual_loss = perceptual_loss_fn(masked_predicted_image, masked_target_image)

                # Calculate adversarial losses
                loss_adv = adversarial_loss(masked_predicted_image, Dbase)
                loss_fm = perceptual_loss_fn(masked_predicted_image, masked_target_image, use_fm_loss=True)
                
                dynamics_transfer_loss = mse_loss(pose2drive, dyn2source)
                cosine_similar_loss = cosine_loss(output_frame, output_frame_random)
                # Calculate cycle consistency loss
                # loss_cos = contrastive_loss(masked_predicted_image, masked_target_image, masked_predicted_image, encoder)
                loss_cos = 0
                # Combine the losses
                total_loss = cfg.training.w_per * perceptual_loss + cfg.training.w_adv * loss_adv + cfg.training.w_fm * loss_fm + cfg.training.w_cos * loss_cos + 10*dynamics_transfer_loss + 10*cosine_similar_loss
                real_pred = Dbase(driving_frame)
                fake_pred = Dbase(output_frame.detach())    
                loss_D = discriminator_loss(real_pred, fake_pred)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()
            # Backpropagate and update generator
            # total_loss.backward()
            # optimizer_G.step()

            # Train discriminator
            optimizer_D.zero_grad()

            
            # Backpropagate and update discriminator
            # loss_D.backward()
            # optimizer_D.step()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
            
                                # Save the images
            # if save_images:
            #     vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
            #     vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
            #     vutils.save_image(warped_driving_frame, f"{output_dir}/warped_driving_frame_{idx}.png")
            #     vutils.save_image(output_frame, f"{output_dir}/output_frame_{idx}.png")
            #     vutils.save_image(foreground_mask, f"{output_dir}/foreground_mask_{idx}.png")
            #     vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
            #     vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")
            # saved_wandb = save_img([source_frame, driving_frame, output_frame, masked_predicted_image, masked_target_image])

            if cur_step % 5 == 0:
                logging.info(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                             f"Step [{cur_step}/{total_step}], "
                             f"Loss_G: {total_loss.item():.4f}, "
                             f"Loss_per: {perceptual_loss.item():.4f}, "
                             f"Loss_fm: {loss_fm.item():.4f}, "
                            #  f"Loss_cos: {loss_cos.item():.4f}, "
                             f"Loss_adv: {loss_adv.item():.4f}, "
                             f"Loss_cosine: {cosine_similar_loss.item():.4f}, "
                             f"Loss_dyn_transfer: {dynamics_transfer_loss.item():.4f}, "
                             f"Loss_D: {loss_D.item():.4f}, "
                             )
                
            if cur_step % 50 == 0:
                # saved_img_all = Image.fromarray(saved_wandb)
                # saved_img_all.save(f"output_images/train_dataset/all_{cur_step}.png")
                vutils.save_image(output_frame, f"output_images/train_dataset_cross/output_imgs/{cur_step}.png")
                # vutils.save_image(masked_predicted_image, f"output_images/train_dataset_cross/masked_predicted_image_{cur_step}.png")
                vutils.save_image(source_frame, f"output_images/train_dataset_cross/source_imgs/{cur_step}.png")
                vutils.save_image(masked_target_image, f"output_images/train_dataset_cross/driving_imgs/{cur_step}.png")


            # Img = wandb.Image(saved_wandb, caption="source driver result masked_driving masked_result") 
            # wandb.log({"frames and results": Img}, step=cur_step)
    
            wandb.log({"Loss_G": total_loss.item(),
                        "Loss_fm": loss_fm.item(),
                        "Loss_per": perceptual_loss.item(),
                        # "Loss_cos": loss_cos.item(),
                        "Loss_adv": loss_adv.item(),
                        "Loss_dyn_transfer": dynamics_transfer_loss.item(),
                        "Loss_cosine": cosine_similar_loss.item(),
                        "Loss_D": loss_D.item()
                    }, 
                    step=cur_step)
            # batch = prefetcher.next()
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {total_loss.item():.4f}, Loss_D: {loss_D.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"checkpoints/Gbase/Gbase_cross_id_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"checkpoints/Dbase/Dbase_cross_id_epoch{epoch+1}.pth")

def main(cfg: OmegaConf) -> None:
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #记录训练的超参数
    config = wandb.config
    config = {
        "learning_rate": cfg.training.lr,
        "epochs": cfg.training.base_epochs,
        "batch_size": cfg.training.batch_size,
        "device" : device 
    }
    wandb.init(project="MegaPortrait", entity="marvin_tec", name="v1-", config=config, settings=wandb.Settings(start_method="fork"))
    
    # transform = transforms.Compose([

    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5]),
    # ])

    transform = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.Normalize([0.5], [0.5]),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter()
    ])
  #     transforms.RandomHorizontalFlip(),
   #     transforms.ColorJitter() # "as augmentation for both source and target images, we use color jitter and random flip"

    voxceleb_dataset = FramesDataset(is_train=True, transform=transform,  **cfg['data'])
    dataloader = DataLoader(voxceleb_dataset, batch_size=cfg.training.batch_size, pin_memory=True,shuffle=True, num_workers=cfg.training.num_workers, drop_last=True)

    # dataset = EMODataset(
    #     use_gpu=use_cuda,
    #     width=cfg.data.train_width,
    #     height=cfg.data.train_height,
    #     n_sample_frames=cfg.training.n_sample_frames,
    #     sample_rate=cfg.training.sample_rate,
    #     img_scale=(1.0, 1.0),
    #     video_dir=cfg.training.video_dir,
    #     json_file=cfg.training.json_file,
    #     transform=transform
    # )
    
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=cfg.training.num_workers)
    
    Gbase = model.Gbase(is_train=True).to(device)
    Gbase.load_state_dict(torch.load("checkpoints/Gbase/Gbase_alldata_epoch150.pth"))
    Dbase = model.Discriminator(input_nc=3).to(device) # 🤷
    Dbase.load_state_dict(torch.load("checkpoints/Dbase/Dbase_alldata_epoch150.pth"))
    train_base(cfg, Gbase, Dbase, dataloader)    
    torch.save(Gbase.state_dict(), 'Gbase_cross_final.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)