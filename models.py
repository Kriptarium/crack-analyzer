
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCrackClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetLite(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 32)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c3 = DoubleConv(256,128)
        self.c2 = DoubleConv(128,64)
        self.c1 = DoubleConv(64,32)
        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b  = self.b(self.p3(d3))
        x  = self.u3(b)
        x  = self.c3(torch.cat([x, d3], dim=1))
        x  = self.u2(x)
        x  = self.c2(torch.cat([x, d2], dim=1))
        x  = self.u1(x)
        x  = self.c1(torch.cat([x, d1], dim=1))
        return self.out(x)
