
# import python library
import os
import shutil
import time
import math
import pickle as pk
import argparse



# import library about pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def validate(val_loader, model, criterion, sn = 1, sc = 0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    input_size = len(val_loader)
    start_point = int(input_size / sc * sn[0])
    end_point = int(input_size / sc * (sn[-1] + 1)) - 1
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if (i < start_point):
                continue
            elif (i > end_point):
                break

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            top5.update(acc5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return (top1.avg, top5.avg)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_int_value(data):
    for size in range(20):
        if 2**size > data:
            return size

def _quantization(data, bitwidth):
    max = torch.max(torch.abs(data))
    
    integer_size = get_int_value(max)
    float_size = bitwidth - integer_size
    
    data = torch.clamp(data, -2**integer_size, 2**integer_size)
    data = torch.mul(data, 2**float_size)
    data = torch.round(data)
    data = torch.div(data, 2**float_size)
    return data

def weight_quantization(kernel, conv, convwidth, linear, linearwidth):
    for i, layer in enumerate(conv):
        for key in kernel.keys():
            key = key.split('.')
            if key[1] == 'features':
                if int(key[2]) == layer:
                    kernel['.'.join(key)] = _quantization(kernel['.'.join(key)], convwidth[i])
                    
    for i, layer in enumerate(linear):
        for key in kernel.keys():
            key = key.split('.')
            if key[1] == 'classifier':
                if int(key[2]) == layer:
                    kernel['.'.join(key)] = _quantization(kernel['.'.join(key)], linearwidth[i])
    return kernel
