import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
        DoubleConv(in_channels, out_channels),
        nn.MaxPool2d(2))

class Backbone(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.n = n

        self.C1 = DoubleConv(3, self.n)
        self.C2 = DoubleConv(self.n, self.n)
        self.C3 = DoubleConv(self.n, self.n)
        self.down1 = Down(self.n, self.n)
        self.C4 = DoubleConv(self.n, self.n)
        self.C5 = DoubleConv(self.n, self.n)
        self.C6 = DoubleConv(self.n, self.n)
        self.down2 = Down(self.n, self.n)
        self.C7 = DoubleConv(self.n, self.n)
        self.C8 = DoubleConv(self.n, self.n)
        self.C9 = DoubleConv(self.n, self.n)
        self.down3 = Down(self.n, self.n)
        self.C10 = DoubleConv(self.n, self.n)
        self.C11 = DoubleConv(self.n, self.n)
        self.C12 = DoubleConv(self.n, self.n)
        self.down4 = Down(self.n, self.n)
        self.C10 = DoubleConv(self.n, self.n)
        self.C11 = DoubleConv(self.n, self.n)
        self.C12 = DoubleConv(self.n, self.n)
        self.down5 = Down(self.n, self.n)
        self.C13 = DoubleConv(self.n, self.n)
        self.C14 = DoubleConv(self.n, self.n)
        self.C15 = DoubleConv(self.n, self.n)
        self.down6 = Down(self.n, self.n//2)
        # flatten op
        self.L1 = nn.Linear(2048, n*2)
        self.L2 = nn.Linear(n*2, n*2)        
        self.L3 = nn.Linear(n*2, n)
        self.r = nn.ReLU()


    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.down1(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        x = self.down2(x)
        x = self.C7(x)
        x = self.C8(x)
        x = self.C9(x)
        x = self.down3(x)
        x = self.C10(x)
        x = self.C11(x)
        x = self.C12(x)
        x = self.down4(x)
        x = self.C10(x)
        x = self.C11(x)
        x = self.C12(x)
        x = self.down5(x)
        x = self.C13(x)
        x = self.C14(x)
        x = self.C15(x)
        x = self.down5(x)
        x = nn.Flatten()(x)
        x = self.L1(x)
        x = self.r(x)
        x = self.L2(x)
        x = self.r(x)
        x = self.L3(x)
        x = self.r(x)

        return x

if __name__ == "__main__":
    from torchsummary import summary
    summary(Backbone().to('cuda:0'), (3, 288, 512))

