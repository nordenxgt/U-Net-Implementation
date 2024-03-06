import torch
from torch import nn
from torchvision.transforms.functional import center_crop

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            Conv(in_channels, out_channels),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_sample(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.up_sample = Conv(in_channels, out_channels)
    
    def forward(self, x_contract: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([center_crop(x_contract, [x.shape[2], x.shape[3]]), x], dim=1)
        return self.up_sample(x)
        
class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottleneck = Conv(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out_conv = nn.Conv2d(64, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.bottleneck(x4)
        
        x6 = self.up1(x4, x5)
        x7 = self.up2(x3, x6)
        x8 = self.up3(x2, x7)
        x9 = self.up4(x1, x8)

        x = self.out_conv(x9)

        return x 