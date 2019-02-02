import subprocess
import re
import collections
from threading import Thread
import time

# serverlist = {'changseok@163.180.172.26 -p 16022' : 378, 'changseok@163.180.172.118 -p 16022' : 594, 'changseok@dgx.komosys.com' : 308}
# serverlist = {'changseok@163.180.172.26 -p 16022' : 1, 'changseok@163.180.172.118 -p 16022' : 1, 'changseok@dgx.komosys.com' : 1}
serverlist = {'dgx.komosys.com' : 1}


def work(server, query, sc, sn, local_result):
    # start_time = time.time()

    query += ' -sc ' + str(sc) + ' -sn ' + str(sn)
    result = subprocess.check_output ('ssh ' + server + ' ' + query , shell=True)
    result = result.decode("utf-8") 
    result = str(result[1:-2]).split(', ')
    result = [int(e) for e in result]
    local_result.append((result[0], result[1]))

    # print('time spend : ', time.time() - start_time)
    return 

def gpu_check(serverlist):
    sub_result = collections.OrderedDict()
    gpu_activation_result = collections.OrderedDict()
    
    # get gpu activation
    for server in serverlist.keys():
        try:
            sub_result[server] = (subprocess.check_output ('ssh ' + server + ' \'nvidia-smi\'' , shell=True))
        except:
            print('[malfunctioning server] : ' + server)
            continue
        print('[working server] : ' + server, end='')

        p = re.compile('(\d+)%[ ]+Default')
        m = p.findall(str(sub_result[server]))
        m = [int(e) for e in m]

        if sum(m) / len(m) == 0:
            print('   [GREEN]')
        elif sum(m) / len(m) < 50:
            print('   [ORANGE]')
        else:
            print('   [RED]')

        for i, e in enumerate(m):
            if e == 0:
                gpu_activation_result[server + ' CUDA_VISIBLE_DEVICES=' + str(i)] = serverlist[server]
        
    return gpu_activation_result



def run(query):
    gpu_activation_result = gpu_check(serverlist)
    result = []
    threads = []
    if len(gpu_activation_result) == 0:
        print('[ABORT] No working server now')
        exit(0)
    print('working gpu count : ', len(gpu_activation_result))

    sc = sum(gpu_activation_result.values())

    savepoint = 0
    for server in gpu_activation_result.keys():
        sn = [str(savepoint + j) for j in range(gpu_activation_result[server])]
        savepoint = int(sn[-1]) + 1
        threads.append(Thread(target=work, args=(server, query, sum(gpu_activation_result.values()), ' '.join(sn), result)))
    start = time.time()
    for i in range(len(gpu_activation_result.keys())):
        threads[i].start()
    for i in range(len(gpu_activation_result.keys())):
        threads[i].join()
    print('time spend : ', time.time() - start)

    now = time.time()
    while True:
        if time.time() - now > 5:
                return run(query)
        if len(result) == len(gpu_activation_result):
            return (sum([pair[0] for pair in result])/50000, sum([pair[1] for pair in result])/50000)
        else:
            continue
        









