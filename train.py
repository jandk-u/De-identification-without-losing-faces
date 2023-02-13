import torchvision

from data import *

from model import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import *


lr = 1e-4
epochs = 20


def train(loader, model, criterion, optimizer, epoch, writer, device='cpu'):

    bar = tqdm(loader)

    for idx, (data1, data2) in enumerate(bar):
        data1 = data1.to(device)
        data2 = data2.to(device).permute(0, 3, 1, 2)

        optimizer.zero_grad()
        # print(data1.shape)
        # print(data2.shape)
        outputs1, outputs2 = model(data1.float(), data2.float())
        # print(outputs1.shape)
        # print(outputs2.shape)
        loss1 = criterion(outputs1, data1)
        loss2 = criterion(outputs2, data2)

        loss2.backward()
        loss1.backward()
        optimizer.step()

        bar.set_postfix(loss1=loss1.item(), loss2=loss2.item())
        if idx % 10 == 1:
            writer.add_scalar('Loss1', loss1.item(), epoch*len(loader) + idx)
            writer.add_scalar('loss2', loss2.item(), epoch*len(loader) + idx)
            img_1 = torchvision.utils.make_grid(outputs1, normalize=True)
            img_2 = torchvision.utils.make_grid(outputs2, normalize=True)
            writer.add_image("real", img_1, global_step=epoch*len(loader) + idx)
            writer.add_image("fake", img_2, global_step=epoch*len(loader) + idx)


def main():
    modelFATM = ModelTrainning()
    optimizer = torch.optim.Adam(modelFATM.parameters(), lr=lr)
    criterion = nn.L1Loss()
    writer = SummaryWriter()

    checkpoint = torch.load('weight/epoch_19.pth')
    modelFATM.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    transform = A.Compose([
        A.RandomCrop(64, 64),
        A.Resize(64, 64),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    celebdata = CelebDataset(root_dir='/home/j/Learn/AI/FATM/save_dir', transform=transform)
    print(len(celebdata))
    dataloader = DataLoader(celebdata, batch_size=12, shuffle=True)

    for epoch in range(epochs):
        train(dataloader, modelFATM, criterion, optimizer,  epoch=epoch, writer=writer, device='cpu')
        save_weight(modelFATM, epoch, optimizer, criterion, f'weight/epoch_{epoch}.pth')


if __name__ == '__main__':
    main()