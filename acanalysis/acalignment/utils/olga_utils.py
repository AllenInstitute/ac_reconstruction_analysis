# -*- coding: utf-8 -*-
import numpy as np
from zipfile import ZipFile
from pathlib import Path

def load_swc(zippath,filename):
    '''
    Load swc file as a N X 7 numpy array
    '''
    swc = []
    with ZipFile(zippath).open(filename,'r') as f:
        lines = f.read().decode('utf-8').split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) == 7:
                    cells = [float(c) for c in cells]
                    swc.append(cells)
                elif len(cells) == 8:
                    cells = [float(c) for c in cells[0:7]]
                    swc.append(cells)                
    return np.array(swc)

def save_swc(filepath, swc):
    with open(filepath, 'w') as f:
        f.write('# id,type,x,y,z,r,pid\n')
        for i in range(swc.shape[0]):
#             f.write('%d %d %d %d %d %d %d\n' %tuple(swc[i, :].tolist()))
            f.write('%d %d %.4f %.4f %.4f %.4f %d\n' %tuple(swc[i, :].tolist()))

def swc_multi_to_single(multizipfile, dirname, fname, sort=False):
    file_list = ZipFile(multizipfile).namelist() 
    file_list.sort()
    print('number of axons %d'%len(file_list))
    trace_list = []
    for f in file_list:
        trace = load_swc(multizipfile,f)
        trace[:, 1] = 2
        trace_list.append(trace)
    if sort == True:
        # sort traces based on size
        trace_length = np.array([t.shape[0] for t in trace_list])
        idx1 = np.flip(np.argsort(trace_length))
        trace_list = [trace_list[i] for i in idx1]
    offset = 0
    for i, trace in enumerate(trace_list):
        select = np.where(trace[:,-1]!=-1)[0]  
        trace_i = np.copy(trace)
        min_id = np.min(trace_i[:,0])
        trace_i[:,0] = trace_i[:,0] + offset - min_id + 1
        trace_i[select,-1] = trace_i[select, -1] + offset - min_id + 1
        offset = np.max(trace_i[:,0])
        if i == 0:
            trace_new = trace_i
        else:
            trace_new = np.concatenate((trace_new, trace_i))
    print(trace_new.shape) 
    save_swc(Path(dirname) / Path(fname), trace_new)  

