import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# from modules.augmentation import AllAugmentationTransform
# from augmentation import AllAugmentationTransform  #! for debug
import glob
import skvideo.io
import random
import decord
from decord import VideoReader,AVReader
from PIL import Image

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name, memtest=False))
        # video = skvideo.io.vread(name)
        # print(video.shape)
        # quit()
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        
        video_array = img_as_float32(video)
 
        # video_array=video

    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, transform, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = []

        for root, dirs, names in os.walk(root_dir):
            for name in names:
                ext = os.path.splitext(name)[1]  # 获取后缀名
                if ext == '.mp4':
                    fromdir = os.path.join(root, name)  # mp4文件原始地址
                    self.videos.append(fromdir)
        random.shuffle(self.videos)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.cpu()
        self.id_sampling = id_sampling
        # if os.path.exists(os.path.join(root_dir, 'train')):
        #     assert os.path.exists(os.path.join(root_dir, 'test'))
        #     print("Use predefined train-test split.")
        #     if id_sampling:
        #         train_videos = {os.path.basename(video).split('#')[0] for video in
        #                         os.listdir(os.path.join(root_dir, 'train'))}
        #         train_videos = list(train_videos)
        #     else:
        #         train_videos = os.listdir(os.path.join(root_dir, 'train'))
        #     test_videos = os.listdir(os.path.join(root_dir, 'test'))
        #     self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        # else:
        #     print("Use random train-test split.")
        #     train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = self.videos[:1000]
        else:
            self.videos = self.videos[:1000]

        self.is_train = is_train

        if self.is_train:
            # self.transform = AllAugmentationTransform(**augmentation_params)
            self.transform = transform
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # if self.is_train and self.id_sampling:
        #     name = self.videos[idx]
        #     path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        # else:
        #     name = self.videos[idx]
        #     path = os.path.join(self.root_dir, name)


        # video_name = os.path.basename(path)
        path = self.videos[idx]
        random_idx = np.random.choice(len(self.videos), replace=True, size=1)
        video_array_list = []
        video_array_rand_list = []
        
        path_random = self.videos[random_idx[0]]
        try:
            if self.is_train and os.path.isdir(path):
                frames = os.listdir(path)
                num_frames = len(frames)
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
                
                video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            else:
                # video_array = read_video(path, frame_shape=self.frame_shape)
                video_array = VideoReader(path, ctx=self.ctx)
                num_frames = len(video_array)
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                    num_frames)
                # video_array = video_array[frame_idx]
                for id in frame_idx:
                    frame = Image.fromarray(video_array[id].numpy())
                    video_array_list.append(frame)


            if self.is_train:
                # random_video_array = read_video(path_random, frame_shape=self.frame_shape)
                random_video_array = VideoReader(path_random, ctx=self.ctx)
                num_frames_random = len(random_video_array)
                frame_idx_random = np.sort(np.random.choice(num_frames_random, replace=False, size=2)) if self.is_train else range(
                    num_frames)
                # random_video_array = random_video_array[frame_idx_random]
                for id in frame_idx_random:
                    frame = Image.fromarray(random_video_array[id].numpy())
                    video_array_rand_list.append(frame)

        except Exception as r:
            print('未知错误 %s' %(r), path_random, path)
        
        if self.transform is not None:
            video_array_trans = []
            video_array_trans_rand = []

            # for img in video_array:
            #     video_array_trans.append(self.transform(img))
            # for img2 in random_video_array:
            #     video_array_trans_rand.append(self.transform(img2))

            for img in video_array_list:
                video_array_trans.append(self.transform(img))
            for img2 in video_array_rand_list:
                video_array_trans_rand.append(self.transform(img2))

        out = {}
        if self.is_train:
            source = video_array_trans[0]
            driving = video_array_trans[1]
            # random_source = video_array_trans_rand[0]
            random_driving = video_array_trans_rand[1]
            
            out['driving'] = driving
            out['source'] = source
            # out['random_source'] = random_source
            out['random_driving'] = random_driving

            # out['driving'] = driving.transpose((2, 0, 1))
            # out['source'] = source.transpose((2, 0, 1))
            # out['random_source'] = random_source.transpose((2, 0, 1))
            # out['random_driving'] = random_driving.transpose((2, 0, 1))

        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        # out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
    

# if __name__ == "__main__":
#     import yaml
#     from torch.utils.data import DataLoader
#     from utils import save_img
#     from PIL import Image
#     import torchvision.transforms as transforms
#     transform = transforms.Compose([
#                                     transforms.ToTensor(),
#                                     transforms.Resize([256, 256]),
#                                     # Normalize data into range(-1, 1)
#                                     transforms.Normalize([0.5], [0.5]),
#                                     # # Randomly flip train data(left and right).
#                                     # transforms.RandomHorizontalFlip(),
#                                     # # Color jitter on data.
#                                     # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
#                                     ])
    
#     with open('vasa-hack-1/config_latent/vox-256.yaml') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     voxceleb_dataset = FramesDataset(is_train=True, transform=transform, **config['dataset_params'])
#     dataloader = DataLoader(voxceleb_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
#     print(voxceleb_dataset.__len__)
#     for x in dataloader:
#         print(x['source'].shape)
#         print(x['driving'].shape)
#         print(x['random_driving'].shape)
#         # a = x['source'].permute(0, 2, 3, 1).numpy().astype('uint8')

#         # a = a[0,:,:,:]

#         out = save_img(x['source'], x['driving'], x['random_driving'], x['source'],x['source'], x['driving'],)
#         out = Image.fromarray(out)
#         out.save('ttt.png')
#         quit()