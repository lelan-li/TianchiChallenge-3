import json
import numpy as np
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip
import cv2
IMAGE_SIZE = 331


class FurnitureDataset(Dataset):
    def __init__(self, preffix,  classname, transform=None):
        self.preffix = preffix
        if preffix == 'test':
            path = '/home/weikun/Desktop/compeition code/round2_question.csv'
        else:
            path = '/home/weikun/Desktop/com_next/train_2.csv'
        self.transform = transform
        
        img,label=[],[]
        load=np.loadtxt(path,dtype=str,delimiter=',')
        class_sub=load[load[:,1]==classname,:]
        n=len(class_sub)
        print('[+] `{classname}: %s `{%s}` loaded {%d} images'%(Dataset,preffix,n))
        if preffix == 'train':
          for i in range(int(0.12*n),n):
            img.append('/home/weikun/Desktop/com_next/train/'+class_sub[i,0])
            label.append(int(class_sub[i,2].find('y')))
        elif preffix == 'val':
          for i in range(int(0.12*n)):
            img.append('/home/weikun/Desktop/com_next/train/'+class_sub[i,0])
            label.append(int(class_sub[i,2].find('y')))
        else:
          for i in range(n):
            img.append('/home/weikun/Desktop/compeition code/final-rank/'+class_sub[i,0])
            label.append(int(class_sub[i,2].find('y')))
        self.img=img
        self.label=label
        print('[+] dataset `{preffix}` loaded {len(img)} images')

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        #row = self.data.iloc[idx]
        img = Image.open(self.img[idx])
        if self.transform:
            img = self.transform(img)
        #target = row['label_id'] - 1 if 'label_id' in row else -1
        target = self.label[idx]
        return img, target


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])
preprocess_hflip = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    HorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
preprocess_with_augmentation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 30, IMAGE_SIZE + 30)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.3),
    transforms.ToTensor(),
    normalize
])
