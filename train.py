import sys
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms  # transforms用于数据处理
import torch.utils.data as d
import numpy as np
import matplotlib.pyplot as plt
from model import *
from copy import deepcopy
import time
import datetime
import pandas as pd
from torchsummary import summary

save_file_name = datetime.datetime.now().strftime("%Y-%m-%d+%H.%M")


def data_process():
    dataset = FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
    )
    train_data, val_data = d.random_split(
        dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))]
    )

    train_data = d.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
    val_data = d.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=8)

    return train_data, val_data


def train_process(model, tran_dataset, val_dataset, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_text = ""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 结合体
    best_acc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    start_time = time.time()

    # 将模型放入设备
    model = model.to(device)
    # 打印模型信息
    file = open("./log/" + save_file_name + ".txt", "a")
    std_out = sys.stdout
    sys.stdout = file
    summary(model, (1, 28, 28))
    sys.stdout = std_out
    file.close()

    best_model_wts = deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print("epoch:", epoch + 1, end=" ")

        # 初始化参数
        train_loss = 0.0
        tran_corrects = 0
        train_samples_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_samples_num = 0

        for step, (train_x, train_y) in enumerate(tran_dataset):
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 将数据输入模型并得到前向传播后的结果
            output = model(train_x)
            lab = torch.argmax(output, dim=1)
            loss = criterion(output, train_y)
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_x.size(0)
            tran_corrects += torch.sum(lab == train_y)
            train_samples_num += train_x.size(0)

        for step, (val_x, val_y) in enumerate(val_dataset):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            model.eval()
            output = model(val_x)
            lab = torch.argmax(output, dim=1)
            loss = criterion(output, val_y)

            val_loss += loss.item() * val_x.size(0)
            val_corrects += torch.sum(lab == val_y)
            val_samples_num += val_x.size(0)

        train_loss_list.append(train_loss / train_samples_num)
        train_acc_list.append(tran_corrects.double().item() / train_samples_num)
        val_loss_list.append(val_loss / val_samples_num)
        val_acc_list.append(val_corrects.double().item() / val_samples_num)

        s = "train loss:{:.4f} train acc:{:.4f} ".format(
            train_loss_list[-1], train_acc_list[-1]
        )
        log_text += s
        print(s, end="")
        s = "val loss:{:.4f} val acc:{:.4f} ".format(
            val_loss_list[-1], val_acc_list[-1]
        )
        log_text += s
        print(s, end="")
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = deepcopy(model.state_dict())
        time_use = time.time() - start_time
        s = "time_cost:{:.0f}m{:.0f}s\n".format(time_use // 60, time_use % 60)
        log_text += s
        print(s, end="")

    with open("./log/" + save_file_name + ".txt", "a") as log:
        log.write(log_text)

    torch.save(best_model_wts, "./weights/best_model.pth")
    visual_data = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_list": train_loss_list,
            "val_loss_list": val_loss_list,
            "train_acc_list": train_acc_list,
            "val_acc_list": val_acc_list,
        }
    )
    return visual_data


def visualize(visual_data):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(
        visual_data["epoch"],
        visual_data.train_loss_list,
        "ro-",
        label="Train_loss",  # ro-为绘图样式 r红色 o实心标记
    )
    plt.plot(visual_data["epoch"], visual_data.val_loss_list, "bs-", label="val_loss")
    plt.legend()  # 用于创建图例
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(visual_data["epoch"], visual_data.train_acc_list, "ro-", label="Train acc")
    plt.plot(visual_data["epoch"], visual_data.val_acc_list, "bs-", label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig("./images/" + save_file_name + ".png")


if __name__ == "__main__":
    train_model = LeNet()
    train_dataset, val_dataset = data_process()
    visual_data = train_process(train_model, train_dataset, val_dataset, 20)
    visualize(visual_data)
