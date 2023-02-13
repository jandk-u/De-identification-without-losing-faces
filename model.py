import torch
from torch import nn


class FATMModel(nn.Module):

    def __init__(self):
        super(FATMModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(5, 5), stride=(2, 2), bias=False),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), bias=False),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(5, 5), stride=(2, 2), bias=False),
            nn.LeakyReLU(0.2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024*4*4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.output = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fully_connected(x)
        # print(x.shape)
        x = x.view(x.size(0), 1024, 4, 4)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x


class ModelTrainning(nn.Module):
    def __init__(self):
        super(ModelTrainning, self).__init__()
        self.modelTraining = FATMModel()

    def forward(self, x1, x2):
        x1 = self.modelTraining(x1)
        x2 = self.modelTraining(x2)
        return x1, x2

    def forward_one(self, x):
        return self.modelTraining(x)


# if __name__ == '__main__':
#     x1 = torch.randn((12, 3, 64, 64))
#     x2 = torch.randn((12, 3, 64, 64))
#     # fatm = FATMMolde()
#     # rs = fatm(x)
#     model = ModelTrainning()
#     x1, x2 = model(x1, x2)
#     # print(x1.shape)
#     # print(x2.shape)
