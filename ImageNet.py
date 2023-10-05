import os
import torch
import torch.nn as nn
import torchvision
import argparse
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.models import resnet50

BATCH_SIZE = 128
DEVICE="cuda"
DIR = "~/imagenet/ILSVRC/Data/CLS-LOC/"
TRAIN_DIR = os.path.join(DIR, 'train')
VAL_DIR = os.path.join(DIR, 'val')



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    train_dataset =datasets.ImageFolder(TRAIN_DIR, transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = resnet50().to(DEVICE)
    print(model)

