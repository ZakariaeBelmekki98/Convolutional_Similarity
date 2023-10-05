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
EPOCHS=20
DEVICE="cuda"
DIR = "~/imagenet/ILSVRC/Data/CLS-LOC/"
TRAIN_DIR = os.path.join(DIR, 'train')
VAL_DIR = os.path.join(DIR, 'val')

###
# Useful functions
###

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(loader, model):
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print(acc1, acc5)


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
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for epoch in range(EPOCHS):
        for i, (images, target) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print("[{}/{}] Loss: {:.3f}".format(epoch, EPOCHS, loss.item()))

        validate(val_loader, model)

