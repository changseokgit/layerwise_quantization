import torch.nn as nn
import torch
import numpy as np


def probability_generator(error_rate):
    if random.randrange(0,int(error_rate), 1) == 1:
        return True
    else:
        return False

def corrupt(tensor, bit_width, location, error_rate):
    """
    tensor = input data
    bit_width = bit representation of input feature and weight value
    location = the location where you want make error (MSB is most significant 4 bit and LSB is least significant 4 bit)
    error_rate = integer value which means error probability ( 10 mean 1/10 )
    """
    mask_list = []
    if location == 'MSB':
        for i in range(min(4,bit_width)):
            tensor = tensor.mul(2)
            mask_list.append((tensor > 1).type(torch.cuda.FloatTensor))
            tensor = tensor.sub(mask_list[i])
        tensor = tensor.mul(2**-min(4,bit_width))
        for i in range(min(4,bit_width)):
            tensor = tensor.mul(2)
            if probability_generator(error_rate):
                tensor = tensor.sub(1-mask_list[i])
            else:
                tensor = tensor.add(mask_list[i])
        tensor = tensor.mul(2**-min(4,bit_width))
    elif location == 'LSB':
        tensor_ = tensor.mul(2**max(0,bit_width - 4))
        tensor = tensor.mul(2**max(0,bit_width - 4))
        tensor_ = tensor_.clamp(-1,1)
        for i in range(min(4,bit_width)):
            tensor_ = tensor_.mul(2)
            tensor = tensor.mul(2)
            mask_list.append((tensor > 1).type(torch.cuda.FloatTensor))
            tensor = tensor.sub(mask_list[i])
        tensor = tensor.mul(2**-min(4,bit_width))
        tensor_ = tensor_.mul(2**-min(4,bit_width))
        for i in range(min(4,bit_width)):
            tensor_ = tensor_.mul(2)
            tensor = tensor.mul(2)
            if probability_generator(error_rate):
                tensor = tensor.sub(1-mask_list[i])
            else:
                tensor = tensor.add(mask_list[i])
        tensor = tensor.mul(2**-min(4,bit_width))
        tensor_ = tensor_.mul(2**-min(4,bit_width))
        tensor = tensor.mul(2**-max(0,bit_width - 4))
    return torch.nn.Parameter(tensor)

def get_int_value(data):
    for size in range(20):
        if torch.pow(torch.tensor(2.0).cuda(), size).cuda() > data:
            return size

def quantize(data, bitwidth):
    torch1 = torch.tensor(1.0).cuda()
    torch2 = torch.tensor(2.0).cuda()

    max_value = torch.max(torch.abs(data))

    integer_size = get_int_value(max_value)
    if integer_size > bitwidth:
        integer_size = bitwidth
    float_size = bitwidth - integer_size
    F = torch.pow(torch2, float_size)
    F_ = torch.pow(torch2, -float_size)
    I = torch.pow(torch2, integer_size)

    data = torch.mul(data, F)
    # data = torch.trunc(data)
    data = torch.round(data)
    data = torch.mul(data, F_)
    data = torch.clamp(data, -I + torch.div(torch1, F), I - torch.div(torch1, F))
    return torch.nn.parameter.Parameter(data)

def get_info(weight_shape, out_shape):
    parameter_size = 1
    for e in weight_shape:
        parameter_size *= e
    out_size = 1
    for e in out_shape:
        out_size *= e
    return (parameter_size, parameter_size * out_size)

class QuantizeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bit_width = (None, None), print_info = False):
        super(QuantizeConv2d, self).__init__(in_channels, out_channels, stride = stride, kernel_size = kernel_size, padding = padding)
        self.feature_bit_width = bit_width[0]
        self.weight_bit_width = bit_width[1]
        self.print_info = print_info



    def forward(self, input):
        if self.weight_bit_width != None and self.weight_bit_width != 24:
            self.weight = quantize(self.weight, self.weight_bit_width)
            if type(self.bias) != type(None):
                self.bias = quantize(self.bias, self.weight_bit_width)

        out = nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        if self.feature_bit_width != None and self.feature_bit_width != 24:
            out = quantize(out, self.feature_bit_width)

        if self.print_info == True:
            print(get_info(self.weight.shape, out.shape))
            self.print_info = False

        return out


class QuantizeLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bit_width = (None, None), print_info = False):
        super(QuantizeLinear, self).__init__(in_channels, out_channels)
        self.weight_bit_width = bit_width[1]
        self.feature_bit_width = bit_width[0]
        self.print_info = print_info

    def forward(self, input):
        if self.weight_bit_width != None and self.weight_bit_width != 24:
            self.weight = quantize(self.weight, self.weight_bit_width)
            if type(self.bias) != type(None):
                self.bias = quantize(self.bias, self.weight_bit_width)

        out = nn.functional.linear(input, self.weight, self.bias)

        if self.feature_bit_width != None and self.feature_bit_width != 24:
            out = quantize(out, self.feature_bit_width)

        if self.print_info == True:
            print(get_info(self.weight.shape, out.shape))
            self.print_info = False

        return out

class rangeSampler(torch.utils.data.Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
