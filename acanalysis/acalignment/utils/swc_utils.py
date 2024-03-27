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


def read_navis_neurons_tar(tar_fn, concurrency=10, preprocess_func=None):
    preprocess_func = ((lambda x: x) if preprocess_func is None else preprocess_func)
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        futs = []
        with tarfile.open(tar_fn, "r:gz") as t:
            for m in t.getmembers():
                swc_b = t.extractfile(m).read()
                futs.append(e.submit(navis.io.read_swc,f=swc_b.decode(),swcname=m.name))
        neurons = [preprocess_func(fut.result()) for
            fut in concurrent.futures.as_completed(futs)]
        navis_neurons = navis.NeuronList([n for n in neurons if not n is None])
    return navis_neurons


def get_axons_from_tar(tar_fn,concurrency=10,prefix='',swap_xyz=None,id_list=None):
    func = lambda x: patch_axon(x,prefix,swap_xyz,id_list)
    axons = read_navis_neurons_tar(tar_fn,concurrency=concurrency,preprocess_func=func)
    return axons


def patch_axon(axon,prefix='',swap_xyz=None,id_list=None):
    aid = current_id_func(axon)
    axon.id = aid
    axon.name = prefix + aid if prefix else axon.swcname.split(".")[-2]
    if not swap_xyz is None and swap_xyz:
        axon.nodes[["x","y","z"]] = axon.nodes[swap_xyz]
    if not id_list is None and not aid in id_list:
        return None
    return axon
        

def current_id_func(axon):
    a = axon.nodes.loc[0]
    aid = str(int(a.z)) + str(int(a.y)) + str(int(a.x))
    return aid


def read_neurons_from_file(filepath,is_tar=False,prefix='',swap_xyz=None,id_list=None):
    if is_tar:
        navis_neurons = get_axons_from_tar(filepath,prefix=prefix,swap_xyz=swap_xyz,id_list=id_list)
    else:
        neurons = get_axon_list_from_subtrees(navis.read_swc(filepath))
        patched = []
        for n in neurons:
            p = patch_axon(n,prefix=prefix,swap_xyz=swap_xyz,id_list=id_list)
            if not p is None:
                patched.append(p)
        navis_neurons = navis.NeuronList(patched)
    return navis_neurons