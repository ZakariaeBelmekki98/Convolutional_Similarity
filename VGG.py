import torch.nn as nn

_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _make_layers(cfg, activation="relu"):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            if activation == "elu":
                layers.append(nn.ELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)


class _VGG(nn.Module):
    def __init__(self, name, num_classes=10, activation="relu"):
        super(_VGG, self).__init__()
        cfg = _cfg[name]
        self.layers = _make_layers(cfg, activation)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, num_classes)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        return y

def VGG11(num_classes=10, activation="relu"):
    return _VGG('VGG11', num_classes, activation)

def VGG13(num_classes=10, activation="relu"):
    return _VGG('VGG13', num_classes, activation)

def VGG16(num_classes=10, activation="relu"):
    return _VGG('VGG16', num_classes, activation)

def VGG19(num_classes=10, activation="relu"):
    return _VGG('VGG19', num_classes, activation)
