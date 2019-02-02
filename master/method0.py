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


start_time = time.time()


result_list = list()
for i in reversed(range(23)):
    data = ' '.join([str(i+1) for j in range(layer_size)])

    print('python3 /home/changseok/layerwise_quantization/slave/main.py ' + ' -wb ' + data + ' -fb ' + data + ' -b 1' + ' -m ' + model_name)
    result = run('python3 /home/changseok/layerwise_quantization/slave/main.py ' + ' -wb ' + data + ' -fb ' + data + ' -b 1' + ' -m ' + model_name)
    print('result is : ', result)
    result_list.append(result)

print('finished! ', time.time() - start)





