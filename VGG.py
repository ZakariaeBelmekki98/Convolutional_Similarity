import torch.nn as nn

config = [64, 64, 'downsample', 128, 128, 'downsample', 256, 256, 256, 256, 'downsample', 512, 512, 512, 512, 'downsample', 512, 512, 512, 512, 'downsample']

def layers(config, activation="relu"):
    layers = []
    in_channels = 3
    for i in config:
        if i == 'downsample':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=i,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False))
            layers.append(nn.BatchNorm2d(num_features=i))
            if activation == "elu":
                layers.append(nn.ELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            in_channels = i
    return nn.Sequential(*layers)


class VGG19(nn.Module):
    def __init__(self, name, num_classes=10, activation="relu"):
        super(VGG19, self).__init__()
        self.layers = layers(config, activation)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, num_classes)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        return y

