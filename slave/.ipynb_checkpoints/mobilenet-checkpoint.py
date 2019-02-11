import torch
import torch.nn as nn
import module


class Net(nn.Module):
    def __init__(self, bitwidth = None):
        super(Net, self).__init__()

        if bitwidth == None:
            bitwidth = [(None, None) for i in range(28)]
        print(bitwidth)

        def conv_bn(inp, oup, stride, bitwidth = (None, None)):
            return nn.Sequential(
                module.QuantizeConv2d(inp, oup, 3, stride, 1, bias=False, bit_width = bitwidth),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride, bitwidth):
            return nn.Sequential(
                module.QuantizeConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, bit_width = bitwidth[0]),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                module.QuantizeConv2d(inp, oup, 1, 1, 0, bias=False, bit_width = bitwidth[1]),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2, bitwidth[0]),
            conv_dw( 32,  64, 1, bitwidth[1:3]),
            conv_dw( 64, 128, 2, bitwidth[3:5]),
            conv_dw(128, 128, 1, bitwidth[5:7]),
            conv_dw(128, 256, 2, bitwidth[7:9]),
            conv_dw(256, 256, 1, bitwidth[9:11]),
            conv_dw(256, 512, 2, bitwidth[11:13]),
            conv_dw(512, 512, 1, bitwidth[13:15]),
            conv_dw(512, 512, 1, bitwidth[15:17]),
            conv_dw(512, 512, 1, bitwidth[17:19]),
            conv_dw(512, 512, 1, bitwidth[19:21]),
            conv_dw(512, 512, 1, bitwidth[21:23]),
            conv_dw(512, 1024, 2, bitwidth[23:25]),
            conv_dw(1024, 1024, 1, bitwidth[25:27]),
            nn.AvgPool2d(7),
        )
        self.fc = module.QuantizeLinear(1024, 1000, bit_width=bitwidth[27])

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        print(x)
        return x
