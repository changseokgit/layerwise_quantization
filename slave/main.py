import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import vgg
import squeeze
import userDefined
import module

#========================argument parser setting=========================
parser = argparse.ArgumentParser(description='paramater parser')
parser.add_argument("-d", "--data", help="data directory", default='/home/changseok/data/')
parser.add_argument("-b", "--batch", help="batch_size", default=10, type=int)
parser.add_argument("-w", "--workers", help="using cpu count", default=12, type=int)
parser.add_argument("-sc", "--servercount", help="total count of server", default=1, type=int)
# parser.add_argument("-sn", "--servernumber", help="number of this server", default=0, type=int)
parser.add_argument("-sn", "--servernumber", help="number of this server", type=int, nargs='+', default=[0])
parser.add_argument("-wb", "--weight", help="weight bitwidth", type=int, nargs='+', default=[None for i in range(300)])
parser.add_argument("-fb", "--featuremap", help="weight bitwidth", type=int, nargs='+', default=[None for i in range(300)])
parser.add_argument("-m", "--model", help="choose model", type=str, default="vgg")

args = parser.parse_args()
data_dir = args.data
batch_size = args.batch
workers = args.workers
server_count = args.servercount
server_number = args.servernumber
weight_bitwidth = args.weight
feature_bitwidth = args.featuremap
model_name = args.model
#========================argument parser setting=========================


#========================data loading=========================
valdir = os.path.join(data_dir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

input_size = len(val_set)

start_point = int(input_size / server_count * server_number[0])
end_point = int(input_size / server_count * (server_number[-1] + 1))
indices = range(start_point, end_point)

print(start_point, end_point)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True,
sampler=module.rangeSampler(indices))
#========================data loading=========================


#========================create model=========================
if model_name == 'vgg':
    quantization_factor = [(feature_bitwidth[j], weight_bitwidth[j]) for j in range(16)]
    model = vgg.vgg16(pretrained = True, bit_width = quantization_factor)
elif model_name == 'squeeze':
    quantization_factor = [(feature_bitwidth[j], weight_bitwidth[j]) for j in range(26)]
    model = squeeze.squeezenet1_0(pretrained = True, bit_width = quantization_factor)

model = torch.nn.DataParallel(model).cuda()
#========================create model=========================

print(len(val_loader))

#========================validate=========================
criterion = nn.CrossEntropyLoss().cuda()
cudnn.benchmark = True
result = userDefined.validate(val_loader, model, criterion)
print(result)
#========================validate=========================
