import subprocess
import sys
from parallel_run import run
import numpy as np
layer_size = 16

official = (71.59, 90.38)
weight_baseline = [9, 8, 9, 8, 12, 8, 8, 7, 11, 8, 9, 8, 10, 9, 10, 12]
feature_baseline = [8, 11, 11, 8, 6, 6, 7, 9, 9, 9, 8, 8, 10, 4, 7, 12]
layer_info = [(1728, 55490641920), (36864, 1183800360960), (73728, 1183800360960), (147456, 2367600721920), (294912, 2367600721920), (589824, 4735201443840), (589824, 4735201443840), (1179648, 4735201443840), (2359296, 9470402887680), (2359296, 9470402887680), (2359296, 2367600721920), (2359296, 2367600721920), (2359296, 2367600721920), (102760448, 4209067950080), (16777216, 687194767360), (4096000, 40960000000)]
# weight_bitwidth = [10, 5, 6, 7, 7, 7, 7, 7, 7, 8, 7, 9, 10, 12, 11, 10] #forward pass
weight_bitwidth = [7, 6, 8, 8, 8, 7, 7, 7, 8, 8, 7, 7, 7, 9, 8, 10]
# featuremap_bitwidth = [10, 14, 15, 5, 6, 10, 10, 9, 11, 13, 12, 12, 13, 8, 11, 12] #forward pass
featuremap_bitwidth = [7, 7, 11, 8, 6, 6, 7, 9, 9, 9, 8, 8, 10, 3, 6, 10]

parameter_size = [layer_info[i][0] * weight_bitwidth[i] for i in range(16)]
computational_cost = np.array([layer_info[i][1] * weight_bitwidth[i] * featuremap_bitwidth[i] for i in range(16)])

def init_parameter():
    return ' '.join([str(24) for j in range(layer_size)])


fb = [7, 7, 11, 8, 6, 6, 7, 9, 9, 9, 8, 8, 10, 3, 6, 10]
fb = [str(e) for e in fb]
fb = ' '.join(fb)
weight_baseline = [e + 3 for e in weight_baseline]
computational_cost = np.array([layer_info[i][1] * weight_baseline[i] * featuremap_bitwidth[i] for i in range(16)])
for i in range(layer_size):
    for j in reversed(range(weight_baseline[i])):
        weight_baseline[i] = j+1
        data = [str(e) for e in weight_baseline]
        data = ' '.join(data)
        result = run('python3 /home/changseok/quantization/main.py ' + ' -wb ' + data + ' -fb ' + fb + ' -b 1')
        print(data)
        if result[1] < official[1]:
            weight_baseline[i] += 1
            break
print('result : ', weight_baseline)


wb = [7, 6, 8, 8, 8, 7, 7, 7, 8, 8, 7, 7, 7, 9, 8, 10]
wb = [str(e) for e in wb]
wb = ' '.join(wb)
feature_baseline = [e + 3 for e in feature_baseline]
computational_cost = np.array([layer_info[i][1] * feature_baseline[i] * weight_bitwidth[i] for i in range(16)])
for i in range(layer_size):
    for j in reversed(range(feature_baseline[i])):
        feature_baseline[i] = j+1
        data = [str(e) for e in feature_baseline]
        data = ' '.join(data)
        result = run('python3 /home/changseok/quantization/main.py ' + ' -wb ' + wb + ' -fb ' + data + ' -b 1')
        print(data)
        if result[1] < official[1]:
            feature_baseline[i] += 1
            break

print('result : ', weight_baseline)
print('result : ', feature_baseline)
