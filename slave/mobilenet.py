import torch
import torch.nn as nn

class MobileNet(nn.Module):
    def __init__(self, bit_width = [None for i in range(28)]):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            module.QuantizeConv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bit_width = bit_width[0]),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[1]),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[2]),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bit_width = bit_width[3]),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[4]),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[5]),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[6]),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bit_width = bit_width[7]),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[8]),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[9]),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[10]),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bit_width = bit_width[11]),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[12]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[13]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[14]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[15]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[16]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[17]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[18]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[19]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[20]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[21]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[22]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bit_width = bit_width[23]),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[24]),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bit_width = bit_width[25]),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            module.QuantizeConv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bit_width = bit_width[26]),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=7, padding=0),
        )
        self.fc = module.QuantizeLinear(in_features=1024, out_features=1000, bias=True, bit_width = bit_width[27])


    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    
def mobilenet(pretrained=True, bit_width = None):
    if bit_width != None:
        model = MobileNet(bit_width = bit_width)
    else:
        model = MobileNet()
    
    if pretrained:
        model.load_state_dict(torch.load('mobilenet_best.weight'))
    
    return model