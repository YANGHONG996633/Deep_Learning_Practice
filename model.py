import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 卷积层
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # 全连接层
        self.f5 = nn.Linear(5 * 5 * 16, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.s2(x)

        x = self.c3(x)
        x = self.relu(x)
        x = self.s4(x)

        x = self.flatten(x)

        x = self.f5(x)
        x = self.relu(x)
        x = f.dropout(x, 0.5)
        x = self.f6(x)
        x = self.relu(x)
        x = f.dropout(x, 0.5)
        y = self.f7(x)

        return y


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1
            ),  # 黑白图片通道为1，RGB通道为3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class VGG(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.block1 = self.vgg_block(2, 1, 64)
        self.block2 = self.vgg_block(2, 64, 128)
        self.block3 = self.vgg_block(3, 128, 256)
        self.block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

        # 权重和参数初始化
        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(
                    l.weight, nonlinearity="relu"
                )  # relu激活函数使用恺明初始化
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)
            if isinstance(l, nn.Linear):
                nn.init.normal_(
                    l.weight, 0, 0.01
                )  # 偏置使用均值为0，方差为0.01的正态分布
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y = self.block4(x)

        return y

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels  # 同一块中输出通道相同
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)  # 不加*是将layers作为一个整体


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    summary(model, (1, 28, 28))
