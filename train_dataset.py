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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from model import Encoder,PerceptualLoss,crop_and_warp_face,get_foreground_mask, CosSimLoss, GANLoss
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
# os.environ['WANDB_MODE'] = 'offline' # for debug


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
# torch.autograd.set_detect_anomaly(True)# this slows thing down - only for debug


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

def feature_matching_loss(pred_fake, pred_real, device):
    loss_G_GAN_Feat = 0
        
    feat_weights = 4.0 / (3 + 1)
    D_weights = 1.0 / 3
    for i in range(3):
        for j in range(len(pred_fake[i])-1):
            loss_G_GAN_Feat += D_weights * feat_weights * F.l1_loss(pred_fake[i][j], pred_real[i][j].detach())
        # quit()
    # loss_G_GAN_Feat = torch.mean(torch.Tensor(loss_G_GAN_Feat))
    # print(loss_G_GAN_Feat)
    return loss_G_GAN_Feat


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

# align to cyclegan
def discriminator_loss_bk(real_pred, fake_pred, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = torch.mean((real_pred - 1)**2)
        fake_loss = torch.mean(fake_pred**2)
    elif loss_type == 'vanilla':
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')

    return ((real_loss + fake_loss) * 0.5).requires_grad_()

# # @profile
# def discriminator_loss(real_pred, fake_pred):
#     real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
#     fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
#     return (real_loss + fake_loss).requires_grad_()

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

def cosine_loss(pos_pairs, neg_pairs,  s=5.0,m=0.2): # We also set 𝑠 = 5 and 𝑚 = 0.2 in the cosine loss.
    loss = torch.tensor(0.0, requires_grad=True).to(device)

    for pos_pair in pos_pairs:
        pos_sim = F.cosine_similarity(pos_pair[0], pos_pair[1])
        neg_loss = torch.tensor(0.0, requires_grad=True).to(device)
        
        for neg_pair in neg_pairs:
            neg_loss = neg_loss + torch.exp(s * (F.cosine_similarity(pos_pair[0], neg_pair[1]) - m))
        
        loss = loss + torch.log(torch.exp(s * (pos_sim - m)) / (torch.exp(s * (pos_sim - m)) + neg_loss))
        loss = torch.mean(loss)
    return loss

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
def train_base(cfg, Gbase, Dbase, dataloader, local_rank):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)
    
    perceptual_loss_fn = PerceptualLoss(local_rank, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0})
    GAN_loss = GANLoss()
    fm_loss_func = nn.L1Loss()
    loss_fm = 0
    # encoder = Encoder(input_nc=3, output_nc=256).to(local_rank)
    total_step = len(dataloader) *cfg.training.base_epochs
    begin_epoch = 0
    cur_step = begin_epoch*len(dataloader)
    scaler = GradScaler()
    print(len(dataloader))
    # prefetcher = DataPrefetcher(dataloader, device)


    for epoch in range(begin_epoch, cfg.training.base_epochs):
        print("epoch:", epoch)
        dataloader.sampler.set_epoch(epoch)
        # batch = prefetcher.next()
        for batch in dataloader:
        # while batch is not None:
            cur_step += 1
            source_frame = batch['source'].to(local_rank)
            driving_frame = batch['driving'].to(local_rank)
            random_source_frame = batch['random_source'].to(local_rank)
            random_driving_frame = batch['random_driving'].to(local_rank)

            # Apply face cropping and random warping to the driving frame for losses ONLY!
            # warped_driving_frame =  crop_and_warp_face(driving_frame, pad_to_original=True)
                    
            # Train generator
            
            # with torch.autograd.detect_anomaly():
            optimizer_G.zero_grad()
            # with autocast():
            output_frame = Gbase(source_frame, driving_frame, same_subject=False) 
            s_start_d_pred = Gbase(random_source_frame, driving_frame, same_subject=False)
            # output_frame_random = Gbase(source_frame, random_driving_frame, same_subject=False)

            _, _, z_s_start_d =  Gbase.module.motionEncoder(s_start_d_pred)
            _, _, z_s_d = Gbase.module.motionEncoder(output_frame)
            _, _, z_d = Gbase.module.motionEncoder(driving_frame)
            _, _, z_d_star = Gbase.module.motionEncoder(random_driving_frame)

            # 256 x 256 - Resize output_frame to match the driving_frame size
            # output_frame = F.interpolate(output_frame, size=(256, 256), mode='bilinear', align_corners=False)

            # Obtain the foreground mask for the target image
            foreground_mask = get_foreground_mask(driving_frame)
            
            # Move the foreground mask to the same device as output_frame
            foreground_mask = foreground_mask.to(output_frame.device)

            # Multiply the predicted and target images with the foreground mask
            masked_predicted_image = output_frame * foreground_mask
            masked_target_image = driving_frame * foreground_mask
                        
                                    
            # perceptual losses
            perceptual_loss = perceptual_loss_fn(masked_predicted_image, masked_target_image)

            # adversarial losses
            real_pred = Dbase(masked_target_image)
            fake_pred = Dbase(masked_predicted_image.detach())

            loss_adv = GAN_loss(fake_pred, True, local_rank)

            loss_fm = feature_matching_loss(fake_pred,real_pred, local_rank)
 
            # feat_weights = 4.0 / (3 + 1)
            # D_weights = 1.0 / 3
            # for i in range(3):
            #     print(len(real_pred[i])-1)
            #     for j in range(len(real_pred[i])-1):
            #         print(j)
                    
            #         # loss_fm += F.l1_loss(fake_pred[i][j], real_pred[i][j].detach())
            #         tmp = F.l1_loss(fake_pred[i][j], real_pred[i][j].detach())
            #         print(tmp)
            # quit()
            # perceptual_loss = perceptual_loss_fn(masked_predicted_image, masked_target_image, use_fm_loss=True)
            
            # Calculate cycle consistency loss
            # loss_cos = contrastive_loss(masked_predicted_image, masked_target_image, masked_predicted_image, encoder)
            
            # cos loss for different latent
            pos_pair = [(z_s_d, z_d), (z_s_start_d, z_d)]
            neg_pair = [(z_s_d, z_d_star), (z_s_start_d, z_d_star)]
            loss_cos = cosine_loss(pos_pair, neg_pair)

            # additional loss in VASA
            # dynamics_transfer_loss = mse_loss(pose2drive, dyn2source)
            # cosine_similar_loss = cosine_loss(output_frame, output_frame_random)

            # Combine the losses
            # total_loss = cfg.training.w_per * perceptual_loss + cfg.training.w_adv * loss_adv + cfg.training.w_fm * loss_fm + cfg.training.w_cos * loss_cos + 10*dynamics_transfer_loss + 10*cosine_similar_loss
            
            total_loss =  cfg.training.w_per * perceptual_loss + cfg.training.w_cos * loss_cos+ cfg.training.w_adv * loss_adv + cfg.training.w_fm * loss_fm 
            # cfg.training.w_cos * loss_cos+ cfg.training.w_adv * loss_adv + cfg.training.w_fm * loss_fm 
            #  + cfg.training.w_adv * loss_adv + cfg.training.w_fm * loss_fm 

                
            loss_D_real = GAN_loss(real_pred, True, local_rank)
            loss_D_fake = GAN_loss(fake_pred, False, local_rank)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            scaler.scale(total_loss).backward(retain_graph=True)
            scaler.step(optimizer_G)
            scaler.update()
            # Backpropagate and update generator
            # total_loss.backward()
            # optimizer_G.step()

            # Train discriminator
            # optimizer_D.zero_grad()

            
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

            if cur_step % 5 ==0 and dist.get_rank() == 0:
                logging.info(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                             f"Step [{cur_step}/{total_step}], "
                             f"Loss_G: {total_loss.item():.4f}, "
                             f"Loss_per: {perceptual_loss.item():.4f}, "
                             f"Loss_fm: {loss_fm.item():.4f}, "
                             f"Loss_cos: {loss_cos.item():.4f}, "
                             f"Loss_adv: {loss_adv.item():.4f}, "
                            #  f"Loss_cosine: {cosine_similar_loss.item():.4f}, "
                            #  f"Loss_dyn_transfer: {dynamics_transfer_loss.item():.4f}, "
                             f"Loss_D: {loss_D.item():.4f}, "
                             )
                
            if cur_step % 50 == 0 and dist.get_rank() == 0:
                # saved_img_all = Image.fromarray(saved_wandb)
                # saved_img_all.save(f"output_images/train_dataset/all_{cur_step}.png")
                vutils.save_image(output_frame, f"output_images/train_dataset_cross/output_imgs/{cur_step}.png")
                # vutils.save_image(masked_predicted_image, f"output_images/train_dataset_cross/masked_predicted_image_{cur_step}.png")
                vutils.save_image(source_frame, f"output_images/train_dataset_cross/source_imgs/{cur_step}.png")
                vutils.save_image(driving_frame, f"output_images/train_dataset_cross/driving_imgs/{cur_step}.png")
                vutils.save_image(s_start_d_pred, f"output_images/train_dataset_cross/s_start_d_pred/{cur_step}.png")


            # Img = wandb.Image(saved_wandb, caption="source driver result masked_driving masked_result") 
            # wandb.log({"frames and results": Img}, step=cur_step)
    
            wandb.log({"Loss_G": total_loss.item(),
                        "Loss_fm": loss_fm.item(),
                        "Loss_per": perceptual_loss.item(),
                        "Loss_cos": loss_cos.item(),
                        "Loss_adv": loss_adv.item(),
                        # "Loss_dyn_transfer": dynamics_transfer_loss.item(),
                        # "Loss_cosine": cosine_similar_loss.item(),
                        "Loss_D": loss_D.item()
                    }, 
                    step=cur_step)
            # batch = prefetcher.next()
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.save_interval == 0 and dist.get_rank() == 0:
            torch.save(Gbase.state_dict(), f"checkpoints/Gbase/Gbase_cross_id_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"checkpoints/Dbase/Dbase_cross_id_epoch{epoch+1}.pth")

def set_seed():
    seed = np.random.randint(1, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: OmegaConf) -> None:
    import datetime
    # set_seed()

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
    wandb.init(project="VASA-ddp", entity="marvin_tec", name="dist-v0-253", config=config, settings=wandb.Settings(start_method="fork"))
    
    parser = argparse.ArgumentParser()


    # parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='rank of distributed processes')
    parser.add_argument('--Gbase_path', default='checkpoints/Gbase/Gbase_alldata_epoch150.pth', type=str, help='resume Gbase ckpt')
    parser.add_argument('--Dbase_path', default='checkpoints/Dbase/Dbase_alldata_epoch150.pth', type=str, help='resume Dbase ckpt')
    args = parser.parse_args()
    local_rank = args.local_rank

    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

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
    train_sampler = torch.utils.data.distributed.DistributedSampler(voxceleb_dataset, shuffle=True)
    dataloader = DataLoader(voxceleb_dataset, batch_size=cfg.training.batch_size, 
                            pin_memory=True, num_workers=cfg.training.num_workers, 
                            sampler=train_sampler)

    # dataloader = DataLoader(voxceleb_dataset, batch_size=cfg.training.batch_size, pin_memory=True,shuffle=True, num_workers=cfg.training.num_workers, drop_last=True)

    # dataset = EMODataset(do
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
    rank = dist.get_rank()
    device_id= rank % torch.cuda.device_count()
    Gbase = model.Gbase(is_train=True, local_rank=device_id).to(device_id)
    Dbase = model.MultiscaleDiscriminator(input_nc=3, getIntermFeat=True).to(device_id) # 🤷

    # if dist.get_rank() == 0 and args.Gbase_path is not None and args.Dbase_path is not None :
    #     Gbase.load_state_dict(torch.load(args.Gbase_path))
    #     Dbase.load_state_dict(torch.load(args.Dbase_path))

    Gbase = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gbase) 
    # Gbase = DDP(Gbase, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    Gbase = DDP(Gbase, find_unused_parameters=True, broadcast_buffers=False)

    # Gbase.load_state_dict(torch.load("checkpoints/Gbase/Gbase_alldata_epoch120.pth"))
    ## 注意要使用find_unused_parameters=True，因为有时候模型里面定义的一些模块 在forward函数里面没有调用，如果不使用find_unused_parameters=True 会报错
    Dbase = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dbase) 
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) 
    # Dbase = DDP(Dbase, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    Dbase = DDP(Dbase, find_unused_parameters=True, broadcast_buffers=False)
    # Dbase.load_state_dict(torch.load('checkpoints/Dbase/Dbase_alldata_epoch150.pth'))


    train_base(cfg, Gbase, Dbase, dataloader, local_rank=device_id)  
    torch.save(Gbase.state_dict(), 'Gbase_cross_final.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)