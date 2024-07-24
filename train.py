import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms  # transforms用于数据处理
import torch.utils.data as d
import numpy as np
import matplotlib.pyplot as plt
from model_LeNet import LeNet
from copy import deepcopy


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

    train_data = d.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8)
    val_data = d.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=8)

    return train_data, val_data


def train_model(model, tran_dataset, val_dataset, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizewr = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = deepcopy(model.state_dict())
