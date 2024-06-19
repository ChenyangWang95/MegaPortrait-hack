import argparse
import torch
import model
import cv2 as cv
import numpy as np
import torch.nn as nn
from torch import autograd
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
from gazenet import get_gaze_model

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


# Create a directory to save the images (if it doesn't already exist)
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.autograd.set_detect_anomaly(True)# this slows thing down - only for debug


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

# align to cyclegan
def discriminator_loss(real_pred, fake_pred, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = torch.mean((real_pred - 1)**2)
        fake_loss = torch.mean(fake_pred**2)
    elif loss_type == 'vanilla':
        real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    
    return ((real_loss + fake_loss) * 0.5).requires_grad_()

def multiscale_discriminator_loss(real_preds, fake_preds, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = sum(torch.mean((real_pred - 1)**2) for real_pred in real_preds)
        fake_loss = sum(torch.mean(fake_pred**2) for fake_pred in fake_preds)
    elif loss_type == 'vanilla':
        real_loss = sum(F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) for real_pred in real_preds)
        fake_loss = sum(F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred)) for fake_pred in fake_preds)
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

def g_nonsaturating_loss(fake_preds):

    loss = sum(F.softplus(-fake_pred).mean() for fake_pred in fake_preds)

    return loss

def d_logistic_loss(real_preds, fake_preds):
    real_loss = sum(F.softplus(-real_pred).mean() for real_pred in real_preds)
    fake_loss = sum(F.softplus(fake_pred).mean() for fake_pred in fake_preds)
    return real_loss + fake_loss

def save_img(inp):
    
    save_list = []
    for img in inp:
        img1 = img.permute(0, 2, 3, 1).cpu().detach().numpy() * 0.5 + 0.5
        img1 = (img1 * 255).astype('uint8')
        img1 = img1[0,:,:,:]
        save_list.append(img1)


    save = np.concatenate(save_list, axis=1)
    
    return save

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def cosine_loss(pos_pairs, neg_pairs, s=5.0, m=0.2):
    assert isinstance(pos_pairs, list) and isinstance(neg_pairs, list), "pos_pairs and neg_pairs should be lists"
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    assert len(neg_pairs) > 0, "neg_pairs should not be empty"
    assert s > 0, "s should be greater than 0"
    assert 0 <= m <= 1, "m should be between 0 and 1"
    
    loss = torch.tensor(0.0, requires_grad=True).to(device)

    for pos_pair in pos_pairs:
        assert isinstance(pos_pair, tuple) and len(pos_pair) == 2, "Each pos_pair should be a tuple of length 2"
        pos_sim = F.cosine_similarity(pos_pair[0], pos_pair[1], dim=0)
        pos_dist = s * (pos_sim - m)
        
        neg_term = torch.tensor(0.0, requires_grad=True).to(device)
        for neg_pair in neg_pairs:
            assert isinstance(neg_pair, tuple) and len(neg_pair) == 2, "Each neg_pair should be a tuple of length 2"
            neg_sim = F.cosine_similarity(neg_pair[0], neg_pair[1], dim=0)
            neg_term = neg_term + torch.exp(s * (neg_sim - m))
        
        assert pos_dist.shape == neg_term.shape, f"Shape mismatch: pos_dist {pos_dist.shape}, neg_term {neg_term.shape}"
        loss = loss + torch.log(torch.exp(pos_dist) / (torch.exp(pos_dist) + neg_term))
        
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    return torch.mean(-loss / len(pos_pairs)).requires_grad_()




def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def train_base(cfg, Gbase, Dbase, dataloader, local_rank, optimizer_G, optimizer_D, scheduler_G, scheduler_D, begin_epoch):
    # Load the pre-trained DeepLabV3 model
    # seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(local_rank)
    # seg_model.eval()
    patch = (1, 256// 2 ** 4, 256 // 2 ** 4)
    Gbase.train()
    Dbase.train()

    total_epoch = int(cfg.training.base_iterations / (len(dataloader)))+1

    
    perceptual_loss_fn = PerceptualLoss(local_rank, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0, 'lpips_loss': 20.0})
    # GAN_loss = GANLoss()
    # hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    hinge_loss = nn.L1Loss()
    # d_loss = nn.MSELoss()
    # fm_loss_func = nn.L1Loss()
    feature_matching_loss = nn.MSELoss()
    
    cur_step = begin_epoch*len(dataloader)

    scaler = GradScaler()

    for epoch in range(begin_epoch, total_epoch):
        dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            cur_step += 1
            source_frame = batch['source'].to(local_rank)
            driving_frame = batch['driving'].to(local_rank)
            random_source_frame = batch['random_source'].to(local_rank)
            random_driving_frame = batch['random_driving'].to(local_rank)

            # foreground_mask_list = get_foreground_mask(seg_model, [source_frame, driving_frame, random_source_frame,random_driving_frame], local_rank)
            
            # source_frame_mask = source_frame * foreground_mask_list[0]
            # driving_frame_mask = driving_frame * foreground_mask_list[1]
            # random_source_frame_mask = random_source_frame * foreground_mask_list[2]
            # random_driving_frame_mask = random_driving_frame * foreground_mask_list[3]

            

            with autocast():
                requires_grad(Gbase, True)
                requires_grad(Dbase, False)
                output_frame = Gbase(source_frame, driving_frame, same_subject=False)

                # mask for perception loss
                # output_frame_mask = output_frame * foreground_mask_list[1]
                # loss_G_per = perceptual_loss_fn(output_frame_mask, driving_frame_mask)
                vgg19_loss, vggface_loss, lpips_loss, gaze_loss = perceptual_loss_fn(output_frame, driving_frame)

                # valid = Variable(torch.Tensor(np.ones((driving_frame.size(0), *patch))), requires_grad=False).to(device)
                # fake = Variable(torch.Tensor(np.ones((driving_frame.size(0), *patch))), requires_grad=False).to(device)

                # real loss
                # real_pred = Dbase(driving_frame, source_frame)
                fake_pred = Dbase(output_frame.detach(), source_frame)
                # loss_G_adv = hinge_loss(fake_pred, valid)
                loss_G_adv = g_nonsaturating_loss(fake_pred)
                # loss_fake = sum([hinge_loss(fake_pred, fake) for fake_pred, fake in zip(fake_preds, fake_list)])
                
                # loss_G_adv = 0.5 * (loss_real + loss_fake)
                # loss_G_adv = loss_real
                # Feature matching loss
                # loss_fm = feature_matching_loss(output_frame, driving_frame)

                s_start_d_pred = Gbase(random_source_frame, driving_frame, same_subject=False)

                _, _, z_s_start_d =  Gbase.module.motionEncoder(s_start_d_pred)
                _, _, z_s_d = Gbase.module.motionEncoder(output_frame)
                _, _, z_d = Gbase.module.motionEncoder(driving_frame)
                _, _, z_d_star = Gbase.module.motionEncoder(random_driving_frame)

                pos_pair = [(z_s_d, z_d), (z_s_start_d, z_d)]
                neg_pair = [(z_s_d, z_d_star), (z_s_start_d, z_d_star)]
                loss_G_cos = cosine_loss(pos_pair, neg_pair)
                
                total_loss = cfg.training.w_per * vgg19_loss + \
                            cfg.training.w_face * vggface_loss + \
                            cfg.training.w_lpips * lpips_loss + \
                            cfg.training.w_cos * loss_G_cos + \
                            cfg.training.w_adv * loss_G_adv +\
                            cfg.training.w_gaze * gaze_loss
                            # cfg.training.w_fm * loss_fm
                            
                

                optimizer_G.zero_grad()
                scaler.scale(total_loss).backward(retain_graph=True)
                nn.utils.clip_grad_norm_(Gbase.parameters(), max_norm=1, norm_type=2)
                scaler.step(optimizer_G)
                scaler.update()

                requires_grad(Gbase, False)
                requires_grad(Dbase, True)
                real_pred = Dbase(driving_frame, source_frame)
                fake_pred = Dbase(output_frame.detach(), source_frame)
                # loss_D = discriminator_loss(real_pred, fake_pred, loss_type='vanilla')
                loss_D = d_logistic_loss(real_pred, fake_pred)
                # Train discriminator
                optimizer_D.zero_grad()
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()

                # if cur_step % 16 == 0:
                #     driving_frame.requires_grad = True
                #     source_frame.requires_grad = True
                #     real_pred = Dbase(driving_frame, source_frame)
                #     r1_loss = d_r1_loss(real_pred, driving_frame)
                #     Dbase.zero_grad()
                #     d_loss_other = 5 / 2 * r1_loss * 16+ 0 * real_pred[0]
                #     print(d_loss_other)
                #     quit()
                #     scaler.scale(d_loss_other).backward()
                #     scaler.step(optimizer_D)
                

                if cur_step % 5 ==0 and dist.get_rank() == 0:
                    logging.info(f"Epoch [{epoch+1}/{total_epoch}], "
                                f"Step [{cur_step}/{cfg.training.base_iterations}], "
                                f"Loss_G: {total_loss.item():.4f}, "
                                f"Loss_per: {vgg19_loss.item():.4f}, "
                                f"Loss_face: {vggface_loss.item():.4f}, "
                                f"Loss_lpips: {lpips_loss.item():.4f}, "
                                f"Loss_cos: {loss_G_cos.item():.4f}, "
                                f"Loss_adv: {loss_G_adv.item():.4f}, "
                                f"Loss_gaze: {gaze_loss.item():.4f}, "
                                # f"Loss_fm: {loss_fm.item():.4f}, "
                                #  f"Loss_cosine: {cosine_similar_loss.item():.4f}, "
                                #  f"Loss_dyn_transfer: {dynamics_transfer_loss.item():.4f}, "
                                f"Loss_D: {loss_D.item():.4f}, "
                                )

                if cur_step % 50 == 0 and dist.get_rank() == 0:
                    img_save = torch.cat([source_frame, random_source_frame, driving_frame, output_frame, s_start_d_pred], dim=-2)

                    vutils.save_image(img_save, f"output_images/withbg_hrdata_refine_gan/{cur_step}.png")
                    del img_save
                if dist.get_rank() == 0:
                    wandb.log({"Loss_G": total_loss.item(),
                                "Loss_per": vgg19_loss.item(),
                                "Loss_lpips": lpips_loss.item(), 
                                "Loss_face": vggface_loss.item(),
                                "Loss_cos": loss_G_cos.item(),
                                "Loss_adv": loss_G_adv.item(),
                                "Loss_gaze": gaze_loss.item(),
                                # "Loss_fm": loss_fm.item(),
                                # "Loss_dyn_transfer": dynamics_transfer_loss.item(),
                                # "Loss_cosine": cosine_similar_loss.item(),
                                "Loss_D": loss_D.item()
                            }, 
                            step=cur_step)
                del total_loss, s_start_d_pred, output_frame, z_s_start_d, z_s_d, z_d, z_d_star
                torch.cuda.empty_cache()

        scheduler_G.step()
        scheduler_D.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.save_interval == 0 and dist.get_rank() == 0:
            state_G = {'epoch': epoch,
                    'model': Gbase.module.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    'scheduler': scheduler_G.state_dict()}
            state_D = {'epoch': epoch,
                    'model': Dbase.module.state_dict(),
                    'optimizer': optimizer_D.state_dict(),
                    'scheduler': scheduler_D.state_dict()}

            torch.save(state_G, f"checkpoints/Gbase/G_base_EP{epoch+1}.pth")
            torch.save(state_D, f"checkpoints/Dbase/D_base_EP{epoch+1}.pth")

def set_seed():
    seed = np.random.randint(1, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: OmegaConf) -> None:
    set_seed()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int, help='rank of distributed processes')
    parser.add_argument('--Gbase_path', default='checkpoints/Gbase/G_base_EP2.pth', type=str, help='resume Gbase ckpt')
    parser.add_argument('--Dbase_path', default='checkpoints/Dbase/D_base_EP2.pth', type=str, help='resume Dbase ckpt')
    parser.add_argument('--resume', default=False, help='Resume FLag')

    args = parser.parse_args()
    local_rank = args.local_rank

    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    rank = dist.get_rank()
    device_id= rank % torch.cuda.device_count()
    
    #记录训练的超参数
    config = wandb.config
    config = {
        "learning_rate": cfg.training.lr,
        "iterations": cfg.training.base_iterations,
        "batch_size": cfg.training.batch_size,
        "device" : device 
    }
    if dist.get_rank() ==0:
        wandb.init(project="Disentanglement-ddp", entity="marvin_tec", name="gaze_refined_MultiScaleD", config=config, settings=wandb.Settings(start_method="fork"))
    
    
    transform = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.Normalize([0.5], [0.5]),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter()
    ])
    voxceleb_dataset = FramesDataset(is_train=True, transform=transform,  **cfg['data'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(voxceleb_dataset, shuffle=True)
    dataloader = DataLoader(voxceleb_dataset, batch_size=cfg.training.batch_size, 
                            pin_memory=True, num_workers=cfg.training.num_workers, 
                            sampler=train_sampler)

    total_epoch = int(cfg.training.base_iterations / (len(dataloader)))+1

    Gbase = model.Gbase(is_train=True, local_rank=device_id).to(device_id)
    Dbase = model.MultiscaleDiscriminator().to(device_id) # 🤷
    # Dbase = model.Discriminator().to(device_id)
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=total_epoch, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=total_epoch, eta_min=1e-6)

    epoch = 0
    if args.resume and args.Gbase_path is not None and args.Dbase_path is not None:
        G_ckpt = torch.load(args.Gbase_path, map_location='cpu')
        D_ckpt = torch.load(args.Dbase_path, map_location='cpu')
        epoch = G_ckpt['epoch']
        Gbase.load_state_dict(G_ckpt['model'])
        Dbase.load_state_dict(D_ckpt['model'])
        optimizer_G.load_state_dict(G_ckpt['optimizer'])
        optimizer_D.load_state_dict(D_ckpt['optimizer'])

        # optimizer_G.param_groups[0]['capturable'] = True
        # optimizer_D.param_groups[0]['capturable'] = True
        
        scheduler_G.load_state_dict(G_ckpt['scheduler'])
        scheduler_D.load_state_dict(D_ckpt['scheduler'])
        print('success load pretrained model')

    Gbase = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gbase) 
    Gbase = DDP(Gbase, find_unused_parameters=True, broadcast_buffers=False)
 
    Dbase = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dbase) 
    Dbase = DDP(Dbase, find_unused_parameters=True, broadcast_buffers=False)


    train_base(cfg, Gbase, Dbase, dataloader, device_id, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epoch)  
    # torch.save(Gbase.state_dict(), 'Gbase_cross_final.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)