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
import decord
from omegaconf import OmegaConf
from torchvision import models
from model import MPGazeLoss,Encoder
from rome_losses import Vgg19 # use vgg19 for perceptualloss 
import cv2
import mediapipe as mp
# from memory_profiler import profile
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition

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
As for driving image, before sending it to Emtn we do a center crop around the face of
a person. Next, we augment it using a random warping based on
thin-plate-splines, which severely degrades the shape of the facial
features, yet keeps the expression intact (ex., it cannot close or open
eyes or change the eyes’ direction). Finally, we apply a severe color
jitter.
'''


def crop_and_warp_face(image_tensor):
    # Check if the input tensor has a batch dimension and handle it
    if image_tensor.ndim == 4:
        # Assuming batch size is the first dimension, process one image at a time
        image_tensor = image_tensor.squeeze(0)
    
    # Convert the single image tensor to a PIL Image
    image = to_pil_image(image_tensor)

    # Detect the face in the image using the numpy array
    face_locations = face_recognition.face_locations(np.array(image))

    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]

        # Crop the face region from the image
        face_image = image.crop((left, top, right, bottom))

        # Convert the face image to a numpy array
        face_array = np.array(face_image)

        # Generate random control points for thin-plate-spline warping
        rows, cols = face_array.shape[:2]
        src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = src_points + np.random.randn(4, 2) * (rows * 0.1)

        # Create a PiecewiseAffineTransform object
        tps = PiecewiseAffineTransform()
        tps.estimate(src_points, dst_points)

        # Apply the thin-plate-spline warping to the face image
        warped_face_array = warp(face_array, tps, output_shape=(rows, cols))

        # Convert the warped face array back to a PIL image
        warped_face_image = Image.fromarray(warped_face_array.astype(np.uint8))

        # Convert the warped PIL image back to a tensor
        return to_tensor(warped_face_image)
    else:
        return None


'''
We load the pre-trained DeepLabV3 model using models.segmentation.deeplabv3_resnet101(pretrained=True). This model is based on the ResNet-101 backbone and is pre-trained on the COCO dataset.
We define the necessary image transformations using transforms.Compose. The transformations include converting the image to a tensor and normalizing it using the mean and standard deviation values specific to the model.
We apply the transformations to the input image using transform(image) and add an extra dimension to represent the batch size using unsqueeze(0).
We move the input tensor to the same device as the model to ensure compatibility.
We perform the segmentation by passing the input tensor through the model using model(input_tensor). The output is a dictionary containing the segmentation map.
We obtain the predicted segmentation mask by taking the argmax of the output along the channel dimension using torch.max(output['out'], dim=1).
We convert the segmentation mask to a binary foreground mask by comparing the predicted class labels with the class index representing the person class (assuming it is 15 in this example). The resulting mask will have values of 1 for foreground pixels and 0 for background pixels.
Finally, we return the foreground mask.
'''

def get_foreground_mask(image):
    # Load the pre-trained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the input image
    input_tensor = transform(image).unsqueeze(0)

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Perform the segmentation
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted segmentation mask
    _, mask = torch.max(output['out'], dim=1)

    # Convert the segmentation mask to a binary foreground mask
    foreground_mask = (mask == 15).float()  # Assuming class 15 represents the person class

    return foreground_mask


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


# @profile
def gaze_loss_fn(predicted_gaze, target_gaze, face_image):
    # Ensure face_image has shape (C, H, W)
    if face_image.dim() == 4 and face_image.shape[0] == 1:
        face_image = face_image.squeeze(0)
    if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
        raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
    
    # Convert face image from tensor to numpy array
    face_image = face_image.detach().cpu().numpy()
    if face_image.shape[0] == 3:  # if channels are first
        face_image = face_image.transpose(1, 2, 0)
    face_image = (face_image * 255).astype(np.uint8)

    # Extract eye landmarks using MediaPipe
    results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return torch.tensor(0.0, requires_grad=True).to(device)

    eye_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
        right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
        eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

    # Compute loss for each eye
    loss = 0.0
    h, w = face_image.shape[:2]
    for left_eye, right_eye in eye_landmarks:
        # Convert landmarks to pixel coordinates
        left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
        right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

        # Create eye mask
        left_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        right_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
        cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

        # Compute gaze loss for each eye
        left_gaze_loss = F.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
        right_gaze_loss = F.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
        loss += left_gaze_loss + right_gaze_loss

    return loss / len(eye_landmarks)


def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device)
    encoder = Encoder(input_nc=3, output_nc=256).to(device)

    for epoch in range(cfg.training.base_epochs):
        print("epoch:", epoch)
        for batch in dataloader:
            source_frames = batch['source_frames'] #.to(device)
            driving_frames = batch['driving_frames'] #.to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx].to(device)
                driving_frame = driving_frames[idx].to(device)

                # B.1 Training details Perform face cropping and random warping on the driving frame - supp. material
                warped_driving_frame = crop_and_warp_face(driving_frame)
                print(f"warped_driving_frame:{warped_driving_frame.shape}")
                if warped_driving_frame is not None:
                    
                    # Train generator
                    optimizer_G.zero_grad()
                    output_frame = Gbase(source_frame, warped_driving_frame)

                    # Resize output_frame to 256x256 to match the driving_frame size
                    output_frame = F.interpolate(output_frame, size=(256, 256), mode='bilinear', align_corners=False)


                    # 💀 Compute losses -  "losses are calculated using ONLY foreground regions" 
                    # Obtain the foreground mask for the target image
                    foreground_mask = get_foreground_mask(source_frame)

                    # Multiply the predicted and target images with the foreground mask
                    masked_predicted_image = output_frame * foreground_mask
                    masked_target_image = source_frame * foreground_mask


                    output_vgg_features = vgg19(masked_predicted_image)
                    driving_vgg_features = vgg19(masked_target_image)  
                    total_loss = 0

                    for output_feat, driving_feat in zip(output_vgg_features, driving_vgg_features):
                        total_loss = total_loss + perceptual_loss_fn(output_feat, driving_feat.detach())

                    loss_adversarial = adversarial_loss(masked_predicted_image, Dbase)

                    loss_gaze = gaze_loss_fn(output_frame, driving_frame, source_frame) # 🤷 fix this
                    # Combine the losses and perform backpropagation and optimization
                    total_loss = total_loss + loss_adversarial + loss_gaze

                    
                    # Accumulate gradients
                    loss_gaze.backward()
                    total_loss.backward(retain_graph=True)
                    loss_adversarial.backward()

                    # Update generator
                    optimizer_G.step()

                    # Train discriminator
                    optimizer_D.zero_grad()
                    real_pred = Dbase(driving_frame)
                    fake_pred = Dbase(output_frame.detach())
                    loss_D = discriminator_loss(real_pred, fake_pred)

                    # Backpropagate and update discriminator
                    loss_D.backward()
                    optimizer_D.step()


        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_gaze.item():.4f}, Loss_D: {loss_D.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")



def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter() # "as augmentation for both source and target images, we use color jitter and random flip"
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    Gbase = model.Gbase().to(device)
    Dbase = model.Discriminator(input_nc=3).to(device) # 🤷
    
    train_base(cfg, Gbase, Dbase, dataloader)    
    torch.save(Gbase.state_dict(), 'Gbase.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)