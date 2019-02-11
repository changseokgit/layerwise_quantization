import subprocess
import sys
from parallel_run import run
import argparse
import time

def init_parameter(value, layer_size):
    if type(value) == int:
        return ' '.join([str(value) for j in range(layer_size)])
    elif type(value) == list:
        return ' '.join([str(value[j]) for j in range(layer_size)])

def method1_subroutine(model, layer_size, mode):
    result = list()
    query = 'python3 /home/changseok/layerwise_quantization/slave/main.py -b 1 -m ' + model
    for i in reversed(range(23)):
        if 0 in mode:
            query += ' -wb ' + init_parameter(i+1, layer_size)
        if 1 in mode:
            query += ' -fb ' + init_parameter(i+1, layer_size)
        if 2 in mode:
            query += ' -pt ' + init_parameter(i+1, layer_size)

        print('running command is : ', query)
        result.append(run(query))
        print(result[-1])
    return result



#setup parsers
parser = argparse.ArgumentParser(description = 'parameter parser')
parser.add_argument("-m", "--model", help = "choose model", type=str, default = 'vgg')
args = parser.parse_args()
model_name = args.model

# record start
start = time.time()

# set model
if model_name == 'vgg':
    layer_size = 16
elif model_name == 'squeeze':
    layer_size = 26

# run distributed environment
result_list = list()
result_list.append(method1_subroutine(model_name, layer_size, [2]))

# save result
file = open('/home/changseok/layerwise_quantization/master/result/' + model_name + '_method1.txt', 'w')
file.write(result_list)

#record end
print('finished! ', time.time() - start)
