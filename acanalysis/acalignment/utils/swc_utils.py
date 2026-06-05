# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:28:09 2023

@author: kevint
"""

import navis
import numpy
import gzip
import concurrent
from io import BytesIO
import tarfile
from acanalysis.splitup_swc import get_axon_list_from_subtrees


def ori_table(ori):
    oriTuple = {
        "+x": ("x", "POS", 0),
        "-x": ("x", "NEG", 0),
        "+z": ("z", "POS", 2),
        "-z": ("z", "NEG", 2)
    }[ori]
    return oriTuple


def shift_navis_xform(skels,shift_xyz):
    M = numpy.diag([1,1,1,1])
    if shift_xyz:
        M[:3,3] = numpy.array(shift_xyz)
    tr = navis.transforms.AffineTransform(M)
    return navis.xform(skels, tr)

def gzip_array(fn, arr):
    with gzip.open(fn, "wb") as f:
        numpy.save(f, arr)


def read_gzip_array(fn):
    with gzip.open(fn, "rb") as f:
        a = numpy.load(f)
    return a


def write_kimi_skels_tar(tar_fn, skels):
    with tarfile.open(tar_fn, mode="w:gz") as t:
        for skid, skel in skels.items():
            bio = BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{skid}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            
def write_navis_skels_tar(tar_fn, skels, mode='w:gz', swcname=False):
    with tarfile.open(tar_fn, mode=mode) as t:
        for sk in skels:
            id = sk.id
            if swcname:
                id = sk.swcname
            if 'label' not in sk.nodes:
                sk.nodes.insert(1, 'label', list(np.zeros(len(sk.nodes))))
            sk = sk.nodes[['node_id', 'label','x','y','z','radius','parent_id']].values.tolist()
            sk = '\n'.join(str(x)[1:-1] for x in sk).replace(",", "")
            bio = BytesIO(sk.encode())
            info = tarfile.TarInfo(name=f"{id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)


def process_swc_file(swc_data, swcname, file_id):
    neuron = navis.io.read_swc(f=swc_data, swcname=swcname)
    neuron.id = file_id
    return neuron

def read_navis_neurons_tar(tar_fn, concurrency=10, preprocess_func=None, uuid=True):
    preprocess_func = ((lambda x: x) if preprocess_func is None else preprocess_func)
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        futs = []
        with tarfile.open(tar_fn, "r:gz") as t:
            for m in t.getmembers():
                # Extract the SWC file contents
                swc_b = t.extractfile(m).read()
                file_id = m.name.split('.')[0]
                try:
                    file_id = int(file_id)
                except:
                    pass
                if uuid:
                    futs.append(e.submit(navis.io.read_swc,f=swc_b.decode(),swcname=file_id))
                else:
                    futs.append(e.submit(process_swc_file, swc_b.decode(), file_id, file_id))
                
        neurons = [preprocess_func(fut.result()) for fut in concurrent.futures.as_completed(futs)]
        navis_neurons = navis.NeuronList([n for n in neurons if not n is None])
    return navis_neurons


def get_axons_from_tar(tar_fn,concurrency=10,prefix='',swap_xyz=None,patch_func=None):
    func = lambda x: patch_axon(x,prefix,swap_xyz,filter_func=patch_func)
    axons = read_navis_neurons_tar(tar_fn,concurrency=concurrency,preprocess_func=func)
    return axons


def patch_axon(axon,prefix='',swap_xyz=None,filter_func=None):
    aid = current_id_func(axon)
    axon.id = aid
    axon.name = prefix + aid if prefix else axon.swcname.split(".")[-2]
    if not swap_xyz is None and swap_xyz:
        axon.nodes[["x","y","z"]] = axon.nodes[swap_xyz]
    if not filter_func is None and not filter_func(axon):
        return None
    return axon


def axon_filter_func(axon,id_list=None,z_range=None,ori='',mincablelength=0,minradius=0,**kwargs):
    if not id_list is None:
        if not axon.id in id_list:
            return False
    if not z_range is None and len(z_range)==2 and ori:
        axis, sign, idx = ori_table(ori)
        xlocs = axon.nodes.iloc[[0,-1]].x.to_numpy()
        zlocs = axon.nodes.iloc[[0,-1]].z.to_numpy()
        loc_func = {
            "POS": numpy.argmax,
            "NEG": numpy.argmin
        }[sign]
        i_end = loc_func(xlocs)
        z = zlocs[i_end]
        if not (z > z_range[0] and z < z_range[1]):
            return False
    if mincablelength > 0 and axon.cable_length < mincablelength:
        return False
    if minradius > 0 and axon.nodes.radius.mean() < minradius:
        return False
    return True
        
        

def current_id_func(axon):
    a = axon.nodes.iloc[0]
    aid = str(int(numpy.abs(a.z))) + str(int(numpy.abs(a.y))) + str(int(numpy.abs(a.x)))
    return aid


def read_neurons_from_file(filepath,is_tar=False,prefix='',swap_xyz=None,id_list=None,ori='',**filter_kwargs):
    patch_func = lambda x: axon_filter_func(x,id_list=id_list,ori=ori,**filter_kwargs)
    if is_tar:
        navis_neurons = get_axons_from_tar(filepath,prefix=prefix,swap_xyz=swap_xyz,patch_func=patch_func)
    else:
        neurons = get_axon_list_from_subtrees(navis.read_swc(filepath))
        patched = []
        for n in neurons:
            p = patch_axon(n,prefix=prefix,swap_xyz=swap_xyz,filter_func=patch_func)
            if not p is None:
                patched.append(p)
        navis_neurons = navis.NeuronList(patched)
    return navis_neurons