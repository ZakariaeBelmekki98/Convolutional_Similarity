import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F
import pandas as pd
import argparse

import time
import matplotlib.pyplot as plt

from ConvSimFunctions import ConvSim2DLoss
from Utils import model_accuracy
from ResNet import ResNet18, ResNet50
from VGG import VGG19
from WideResNet import WideResNet

import random
import numpy as np

DEVICE = torch.device("cuda")
parser = argparse.ArgumentParser()
parser.add_argument('id', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('epochs', type=int)
parser.add_argument('model', type=str)
parser.add_argument('--sch_step_size', type=int, default=20)
parser.add_argument('--activation', type=str, default="relu")
parser.add_argument('--reps', type=int, default=3)
parser.add_argument('--lr2', type=float, default=0.002)
parser.add_argument('--alpha', type=int, default=0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

LR = args.lr
LR2 = args.lr2
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
ID = args.id
ACTIVATION = args.activation
ALPHA = args.alpha
BETA = args.beta
REPS = args.reps
SSS = args.sch_step_size
MODEL = args.model

def InitMsg():
    print("Experiment {} ({}) on {} ({}, {}): ".format(ID, MODEL, ACTIVATION, ALPHA, BETA))
    print("\tBatch Size: {},  Epochs: {} ".format(BATCH_SIZE, EPOCHS))
    print("\tOptimizer1: SGD(lr={}, momentum=0.9)".format(LR))
    if LR == 0.1:
        print("\tScheduler1: StepLR({}, 0.1)".format(SSS))
    if ALPHA != 0:
        print("\tOptimizer2: Adam({})".format(LR2))

if __name__ == '__main__':
    for k in range(REPS):
        InitMsg()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

        net1 = None
        if(MODEL=='RN18'):
            net1 = ResNet18(num_classes=10, activation=ACTIVATION).to(DEVICE)
        elif(MODEL=='RN50'):
            net1 = ResNet50(num_classes=10, activation=ACTIVATION).to(DEVICE)
        elif(MODEL=='VGG19'):
            net1 = VGG19(num_classes=10, activation=ACTIVATION).to(DEVICE)
        elif(MODEL=='WRN28'):
            net1 = WideResNet(28, 10, activation=ACTIVATION).to(DEVICE)
        else:
            print("Invalid Model: {}".format(MODEL))
            exit(1)
        print(net1)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=LR, momentum=0.9)
        optimizer2 = torch.optim.Adam(net1.parameters(), lr=LR2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[100, 150], gamma=0.1, verbose=True)

        # Arrays to store results
        acc_test_list = []
        acc_train_list = []
        css_loss_list = []
        init_css_loss_list = []

        for j in range(ALPHA):
            optimizer2.zero_grad(set_to_none=True)
            loss2_css = ConvSim2DLoss(net1)
            loss2_css.backward()
            optimizer2.step()
            init_css_loss_list.append(loss2_css.item())

        for epoch in range(EPOCHS):
            begin = time.time()
            running_loss = 0.0
            cnt = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                optimizer1.zero_grad(set_to_none=True)
                outputs = net1(inputs)
                loss1 = criterion(outputs, labels)
                
                running_loss += loss1.item()
                cnt += 1
                if BETA != 0:
                    loss1 += BETA * ConvSim2DLoss(net1)
                loss1.backward()
                optimizer1.step()
            end = time.time() - begin
            print("Runtime: {:.5f}".format(end))
            print("Original Loss: {:.5f}".format(running_loss/cnt))
            acc_train = model_accuracy(net1, train_loader, DEVICE)
            acc_test = model_accuracy(net1, test_loader, DEVICE)


            with torch.no_grad():
                css = 0
                css = ConvSim2DLoss(net1)
                css_loss_list.append(css.item())
            print("[{}/{}] Accuracy(Test/Train):\n\t{:.2f}%/{:.2f}\n\t{:.5f}".format(epoch, EPOCHS, acc_test, acc_train, css.item()))

            if(LR==0.1):
                scheduler.step()

            if(acc_test ==10):
                print("Problemo")
                exit(1)
                break
            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)
        output = pd.DataFrame({
            'AccTest' : acc_test_list,
            'AccTrain' : acc_train_list,
            'CssLoss' : css_loss_list,
            })
        output.to_csv('results/ConvSim2D/CIFAR10/CSV/{}_{}_{}_{}_{}.csv'.format(MODEL, ID, ALPHA, BETA, k))

        output2 = pd.DataFrame({
            'CssInitLoss' : init_css_loss_list
        })
        output2.to_csv('results/ConvSim2D/CIFAR10/CSV/{}_{}_{}_{}_{}_cssinit.csv'.format(MODEL, ID, ALPHA, BETA, k))




