import subprocess
import sys
from parallel_run import run
import argparse
import time


start = time.time()

parser = argparse.ArgumentParser(description = 'parameter parser')
parser.add_argument("-m", "--model", help = "choose model", type=str, default = 'vgg')

args = parser.parse_args()
model_name = args.model

if model_name == 'vgg':
    layer_size = 16
elif model_name == 'squeeze':
    layer_size = 26

def init_parameter():
    return ' '.join([str(24) for j in range(layer_size)])



result_list = list()
# wb = init_parameter()
wb = [str(10) for i in range(layer_size)]
wb = ' '.join(wb)
for i in reversed(range(23)):
    fb = [str(i+1) for j in range(layer_size)]
    fb = ' '.join(fb)

    print('python3 /home/changseok/quantization/main.py ' + ' -wb ' + wb + ' -fb ' + fb + ' -b 1' + ' -m ' + model_name)
    result = run('python3 /home/changseok/quantization/main.py ' + ' -wb ' + wb + ' -fb ' + fb + ' -b 1' + ' -m ' + model_name)
    result_list.append(result)
    print(result_list[-1])

file = open('./result/method1_feature.txt', 'w')
file.write(str(result_list))

#--------------------------------------------------------------------------------------------------------------------------------------------------------

result_list = list()
# fb = init_parameter()
fb = [str(11) for i in range(layer_size)]
fb = ' '.join(fb)
for i in reversed(range(23)):
    wb = [str(i+1) for j in range(layer_size)]
    wb = ' '.join(wb)

    print('python3 /home/changseok/quantization/main.py ' + ' -wb ' + wb + ' -fb ' + fb + ' -b 1' + ' -m ' + model_name)
    result = run('python3 /home/changseok/quantization/main.py ' + ' -wb ' + wb + ' -fb ' + fb + ' -b 1' + ' -m ' + model_name)
    result_list.append(result)
    print(result_list[-1])

file = open('./result/method1_weight.txt', 'w')
file.write(str(result_list))



print('finish! , total time consumtion : ', time.time() - start)
