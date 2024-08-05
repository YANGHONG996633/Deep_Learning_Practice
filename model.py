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
                    l.weight, mode="fan_out", nonlinearity="relu"
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


class Inception(nn.Module):
    def __init__(self, in_channels, b1, b2, b3, b4) -> None:
        super().__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, b1, 1),
            nn.ReLU(),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, b2[0], 1),
            nn.ReLU(),
            nn.Conv2d(b2[0], b2[1], 3, padding=1),
            nn.ReLU(),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, b3[0], 1),
            nn.ReLU(),
            nn.Conv2d(b3[0], b3[1], 5, padding=2),
            nn.ReLU(),
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_channels, b4, 1), nn.ReLU()
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)

        return torch.cat([branch_1, branch_2, branch_3, branch_4], 1)


class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(3, 2, 1),
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(3, 2, 1),
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.block6 = nn.Sequential(nn.Flatten(), nn.Linear(1024, 2))
        init_weights(self)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        y = self.block6(x)

        return y


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.branch_2 = (
            None
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, x):
        b1 = self.branch_1(x)
        b2 = x if self.branch_2 == None else self.branch_2(x)
        y = self.relu(b1 + b2)

        return y


class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2, 1),  # 56*56*64
        )
        self.block2 = Residual(64, 64)  # 56*56*64
        self.block3 = Residual(64, 64)  # 56*56*64
        self.block4 = Residual(64, 128, 2)  # 28*28*128
        self.block5 = Residual(128, 128)  # 28*28*128
        self.block6 = Residual(128, 256, 2)  # 14*14*256
        self.block7 = Residual(256, 256)  # 14*14*256
        self.block8 = Residual(256, 512, 2)  # 7*7*512
        self.block9 = Residual(512, 512)  # 7*7*512
        self.block10 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        y = self.block10(x)

        return y


def init_weights(model) -> None:
    # 权重和参数初始化
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_out",
                nonlinearity="relu",  # std=sqrt(2/mode) fan_in保存该层的输入数量，fan_out保存输出
            )  # relu激活函数使用恺明初始化
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        if isinstance(layer, nn.Linear):
            nn.init.normal_(
                layer.weight, 0, 0.01
            )  # 偏置使用均值为0，方差为0.01的正态分布
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    summary(model, (3, 224, 224), 1)
