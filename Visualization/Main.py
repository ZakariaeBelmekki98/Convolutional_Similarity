import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt 
import numpy as np

from ConvSimFunctions import ConvSim2DLoss
from Utils import model_accuracy

torch.manual_seed(0)

# Model Declaration

class CNN1(torch.nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 4, 5)
        self.conv2 = torch.nn.Conv2d(4, 4, 5)
        self.conv3 = torch.nn.Conv2d(4, 4, 5)
        self.conv4 = torch.nn.Conv2d(4, 4, 5)
        self.act = torch.nn.ReLU()
        self.lin = torch.nn.Linear(576, 10)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        x4 = self.act(self.conv4(x3))
        
        res = self.soft(self.lin(x4.flatten(start_dim=1)))
        return [res, x1, x2, x3, x4]
    
def cos_sim(vectors):
    N = vectors.shape[0]
    res = torch.zeros([N, N], dtype=torch.float)
    vectors = vectors.flatten(start_dim=1)
    for i in range(N):
        for j in range(N):
            res[i, j] += torch.dot(vectors[i], vectors[j])/torch.norm(vectors[i])/torch.norm(vectors[j])
    return res


# MACROS
BATCH_SIZE = 128
EPOCHS = 50
DEVICE = "cuda"

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

    model1 = CNN1().to(DEVICE)
    model2 = CNN1().to(DEVICE)

    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

   
    for e in range(EPOCHS):
        running_loss1 = 0.0
        running_loss2 = 0.0
        cnt = 0
        for i, data in enumerate(train_loader, 0):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
            optimizer1.zero_grad(set_to_none=True)
            outputs1 = model1(images)
            loss1 = loss_fun(outputs1[0], labels) + 0.01 * ConvSim2DLoss(model1)
            loss1.backward()
            optimizer1.step()


            optimizer2.zero_grad(set_to_none=True)
            outputs2 = model2(images)
            loss2 = loss_fun(outputs2[0], labels)
            loss2.backward()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            cnt += 1
            
        acc_test1 = model_accuracy(model1, test_loader, DEVICE)
        acc_train1 = model_accuracy(model1, train_loader, DEVICE)
        acc_test2 = model_accuracy(model2, test_loader, DEVICE)
        acc_train2 = model_accuracy(model2, train_loader, DEVICE)

        with torch.no_grad():
            print("[{}]\n\t[Model 1] Loss : {:.4f} Conv Sim: {:.4f} | Accuracy(Test/Train): {:.3f}%/{:.3f}%".format(e, running_loss1/cnt, ConvSim2DLoss(model1),acc_test1, acc_train1))
            print("\t[Model 2] Loss : {:.4f} Conv Sim: {:.4f} | Accuracy(Test/Train): {:.3f}%/{:.3f}%\n".format(running_loss2/cnt, ConvSim2DLoss(model2),acc_test2, acc_train2))

    with torch.no_grad():
        # Visualize Conv1 Feature maps
        image = next(iter(train_loader))[0].to(DEVICE)

        outputs1 = model1(image[0].view(1, 1, 28, 28))
        print(cos_sim(outputs1[2][0].cpu()))

        outputs2 = model2(image[0].view(1, 1, 28, 28))
        print(cos_sim(outputs2[2][0].cpu()))
    
    # Visualize the weights from Conv1
    """
    plt.figure(figsize=(2, 2))
    plt.axis("off")
    plt.title("Conv1 weights/kernels")
    plt.imshow(np.transpose(make_grid(model.conv1.weight[:], padding=2).cpu(), (1, 2, 0)), cmap='gray')
    plt.show()
    """
    with torch.no_grad():
        # Visualize Conv1 Feature maps
        image = next(iter(train_loader))[0].to(DEVICE)
        outputs1 = model1(image[0].view(1, 1, 28, 28))
        outputs2 = model2(image[0].view(1, 1, 28, 28))
        fig = plt.figure(figsize=(2,2))
        plt.axis("off")
        plt.title("Conv1 Feature Maps")
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
        for ax, im in zip(grid, outputs1[2][0].cpu()):
            ax.imshow(im, cmap="gray")
        plt.show()
        fig = plt.figure(figsize=(2,2))
        plt.axis("off")
        plt.title("Conv1 Feature Maps")
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
        for ax, im in zip(grid, outputs2[2][0].cpu()):
            ax.imshow(im, cmap="gray")
        plt.show()















