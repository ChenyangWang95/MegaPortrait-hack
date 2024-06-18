import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.mobilenetv2 import Conv2dNormActivation
from torchvision.transforms import Compose, Lambda

import numpy as np 
import cv2
from PIL import Image
from torchvision import transforms

class GazeNet(nn.Module):

    def __init__(self, device):    
        super(GazeNet, self).__init__()
        self.device = device
        self.preprocess = transforms.Compose([
            # Lambda(lambda x: (x + 1)/2 * 255.0),  # Normalize to [-1, 1]
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            transforms.Resize((112,112)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model = torchvision.models.mobilenet_v2()
        model.features[-1] = Conv2dNormActivation(320, 256, kernel_size=1)
        self.backbone = model.features

        self.Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc_final = nn.Linear(512, 2)

        self._initialize_weight()
        self._initialize_bias()
        self.to(device)


    def _initialize_weight(self):
        nn.init.normal_(self.Conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv3.weight, mean=0.0, std=0.001)

    def _initialize_bias(self):
        nn.init.constant_(self.Conv1.bias, val=0.1)
        nn.init.constant_(self.Conv2.bias, val=0.1)
        nn.init.constant_(self.Conv3.bias, val=1)

    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)
        y = F.relu(self.Conv1(x))
        y = F.relu(self.Conv2(y))
        y = F.relu(self.Conv3(y))
        
        x = F.dropout(F.relu(torch.mul(x, y)), 0.5)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        gaze = self.fc_final(x)

        return gaze

    def get_gaze(self, img):
        img = self.preprocess(img)
        with torch.no_grad():
            x = self.forward(img.to(self.device))
        return x

def get_gaze_model():
    print('Loading MobileFaceGaze model...')
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model = GazeNet(device)

    if(not torch.cuda.is_available()):
        print('Tried to load GPU but found none. Please check your environment')
    state_dict = torch.load("./checkpoints/gazenet.pth", map_location=device)
    model.load_state_dict(state_dict)
    print('Model loaded using {} as device'.format(device))

    model.eval()
    return model


def reverse_transform():
    return Compose([
        Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),  # Denormalize
        #Lambda(lambda x: (x + 1) * 255.0/2),  # Denormalize
        #Lambda(lambda x: x[[2, 1, 0], :, :]),
        Lambda(lambda x: x.permute(1, 2, 0)),
    ])
    
def draw_gaze(image_in, eye_pos, pitchyaw, length=200, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.5)
    return image_out

if __name__ == "__main__":
    # from src.dataloader import VideoDataset, transform
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    
    from frames_dataset import FramesDataset
    cfg = OmegaConf.load("./configs/training/stage1-base.yaml")

    model = get_gaze_model()
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
    for batch in dataloader:
        source_frame = batch['source']
        driving_frame = batch['driving']
        # random_source_frame = batch['random_source']
        # random_driving_frame = batch['random_driving']

        # Pass the input data through the model
        output = model.get_gaze(source_frame)
        output2 = model.get_gaze(driving_frame)
        break

    # Print the shape of the output
    print(output)
    print(output2)
    from torchvision.transforms import ToPILImage
    rev_transform = reverse_transform()

    # Convert tensor to PIL Image and save
    to_pil = ToPILImage()
    import matplotlib.pyplot as plt
    for i, img_tensor in enumerate(source_frame):
        plt.imshow(rev_transform(img_tensor).cpu())
        plt.axis('off')
        plt.savefig(f'test_{i+1}.png', bbox_inches='tight', pad_inches=0)

    for i, img_tensor in enumerate(driving_frame):
        # Display the image using matplotlib
        plt.imshow(rev_transform(img_tensor).cpu())
        plt.axis('off')
        plt.savefig(f'test_{i+4}.png', bbox_inches='tight', pad_inches=0)


