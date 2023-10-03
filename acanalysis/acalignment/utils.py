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
                futs.append(e.submit(navis.io.read_swc, swc_b.decode()))
        navis_neurons = navis.NeuronList([
            preprocess_func(fut.result()) for
            fut in concurrent.futures.as_completed(futs)])
    return navis_neurons


def get_axons_from_tar(tar_fn,concurrency=10,preprocess_func=None):
    axons = read_navis_neurons_tar(tar_fn,concurrency=concurrency,preprocess_func=preprocess_func)
    for i,axon in enumerate(axons):
        axon.name = str(i)
        axon.id = str(axon.id)
    return axons