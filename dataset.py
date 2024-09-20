import os
import cv2
import pandas as pd
import moviepy.editor as mp
import librosa
import os
import numpy as np
# from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MultiModalDataset(Dataset):
    def __init__(self, audio_dir, image_dir, label_file, phase):
        self.phase = phase
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.label_df = pd.read_csv(label_file)
        self.classes = ['Positive', 'Neutral', 'Negative']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 预先过滤掉无法加载的样本
        initial_len = len(self.label_df)
        self.label_df = self.label_df.apply(self._validate_row, axis=1)
        self.label_df = self.label_df.dropna()
        final_len = len(self.label_df)
        print(f"Filtered out {initial_len - final_len} invalid samples.")
    
    def _validate_row(self, row):
        image_path = os.path.join(self.image_dir, row['image_filename'])
        audio_path = os.path.join(self.audio_dir, row['audio_filename'])
        annotation = row['annotation']
        
        # 检查文件是否存在
        if not os.path.isfile(image_path):
            print(f"Image file does not exist: {image_path}")
            return None
        if not os.path.isfile(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None
        if annotation not in self.class_to_idx:
            print(f"Invalid annotation '{annotation}' in row {row.name}.")
            return None
        return row
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        image_filename = row['image_filename']
        audio_filename = row['audio_filename']
        annotation = row['annotation']
        
        image_path = os.path.join(self.image_dir, image_filename)
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        # 加载图像数据
        image = Image.open(image_path).convert('RGB')
        if self.phase == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        
        # 加载音频数据
        audio_array, sample_rate = librosa.load(audio_path, sr=None)
        target_sample_rate = 1  # 1样本/秒
        if sample_rate != target_sample_rate:
            audio_array_resampled = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sample_rate)
        else:
            audio_array_resampled = audio_array
        audio_tensor = torch.tensor(audio_array_resampled, dtype=torch.float32)
        
        # 将类别标签转换为整数
        label = self.class_to_idx[annotation]
        label = torch.tensor(label, dtype=torch.long)  # 将标签转换为long类型
        
        return image, audio_tensor, label
    
    @staticmethod
    def collate_fn(batch):
        # 过滤掉返回 None 的样本
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None, None, None
        images, audios, annotations = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_audios = cat_list(audios, fill_value=0)
        annotations = torch.stack(annotations, dim=0)  
        return batched_imgs, batched_audios, annotations

def cat_list(items, fill_value=0):
    # 确保 items 是 PyTorch tensor
    if isinstance(items[0], np.ndarray):
        items = [torch.tensor(item) for item in items]
    
    # 处理1D音频数据
    if items[0].ndim == 1:  
        max_length = max([item.shape[0] for item in items])
        batched_items = torch.full((len(items), max_length), fill_value, dtype=items[0].dtype)
        for i, item in enumerate(items):
            batched_items[i, :item.shape[0]].copy_(item)
    else:  # 处理图像数据 (假设为3D: C x H x W)
        batch_shape = [len(items)] + list(items[0].shape)
        batched_items = torch.full(batch_shape, fill_value, dtype=items[0].dtype)
        for i, item in enumerate(items):
            batched_items[i, ...] = item
    
    return batched_items
if __name__ == '__main__':
    # 使用方法
    image_dir = "./output_images"
    audio_dir = "./output_audio"
    label_file = "./val_label.csv"
    tt = MultiModalDataset(audio_dir, image_dir, label_file, phase='train') 
    print(len(tt))
    # 获取第一个数据样本
    image, audio, target = tt[0]

    # train_dataloader=data.DataLoader(tt,batch_size=32,num_workers=8,shuffle=True)
    train_data_loader = data.DataLoader(tt,
                                        batch_size=16,
                                        num_workers=1,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=tt.collate_fn)

    
    
    for i in range(len(train_data_loader.dataset)):
        print(tt[i][1])
        break
