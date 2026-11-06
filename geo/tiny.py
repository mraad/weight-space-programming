import torch
import torch.nn.functional as F
from torch import nn


def conv(a, b):
    return nn.Conv2d(a, b, kernel_size=3, padding=1)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0a, self.e0b = conv(3, 4), conv(4, 4)
        self.e1a, self.e1b = conv(4, 8), conv(8, 8)
        self.e2a, self.e2b = conv(8, 12), conv(12, 12)
        self.e3a, self.e3b = conv(12, 12), conv(12, 12)
        self.e4a, self.e4b, self.e4c = conv(12, 12), conv(12, 12), conv(12, 12)
        self.d3 = conv(24, 12)
        self.d2 = conv(24, 12)
        self.d1 = conv(20, 8)
        self.d0 = conv(12, 4)
        self.out = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        def block(a, b, y):
            return F.relu(b(F.relu(a(y))))

        def up(y):
            return F.interpolate(y, scale_factor=2, mode="nearest")

        e0 = block(self.e0a, self.e0b, x)
        e1 = block(self.e1a, self.e1b, F.avg_pool2d(e0, 2))
        e2 = block(self.e2a, self.e2b, F.avg_pool2d(e1, 2))
        e3 = block(self.e3a, self.e3b, F.avg_pool2d(e2, 2))
        e4 = F.relu(self.e4c(block(self.e4a, self.e4b, F.avg_pool2d(e3, 2))))
        d3 = F.relu(self.d3(torch.cat([up(e4), e3], 1)))
        d2 = F.relu(self.d2(torch.cat([up(d3), e2], 1)))
        d1 = F.relu(self.d1(torch.cat([up(d2), e1], 1)))
        d0 = F.relu(self.d0(torch.cat([up(d1), e0], 1)))
        return self.out(d0)
