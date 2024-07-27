import torch
import torch.utils.data as d
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import *
import time


def data_process():
    dataset = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
    )
    test_data = d.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    return test_data


def test_process(model, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tp = 0
    start_time = time.time()
    model = model.to(device)

    # 测试不需要进行反向传播
    with torch.no_grad():
        for test_x, test_y in test_dataset:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            model.eval()
            output = model(test_x)
            pre_lab = torch.argmax(output, dim=1)
            tp += torch.sum(pre_lab == test_y.data)

        time_cost = time.time() - start_time
        test_acc = tp.double().item() / len(test_dataset)
        print(
            "测试精度为:{:.4f},耗时:{:.0f}m{:.0f}s".format(
                test_acc, time_cost // 60, time_cost % 60
            )
        )


if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(
        torch.load("./weights/best_model.pth", map_location=torch.device("cpu"))
    )
    dataset = data_process()
    test_process(model, dataset)
