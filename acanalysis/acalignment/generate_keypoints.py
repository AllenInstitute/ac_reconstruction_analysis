# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:50:02 2023

@author: kevint
"""

import navis
import numpy
from scipy.interpolate import RegularGridInterpolator as RGI

from keypoints import KeyPoint,write_keypoints_to_file

def ori_lookup(ori):
    oriTuple = {
        "+x": ("x", "POS", 0),
        "-x": ("x", "NEG", 0)
    }[ori]
    return oriTuple


def keypoint_from_neuron(neuron, ori=None):
    name = neuron.name
    axis, sign, idx = ori_lookup(ori)
    endpts = numpy.vstack([neuron.leafs[["x", "y", "z"]], neuron.nodes[neuron.nodes.node_id == neuron.root.flatten()[0]][["x", "y", "z"]]])
    loc_func = {
        "POS": numpy.argmax,
        "NEG": numpy.argmin
    }[sign]
    i_end = loc_func(endpts[:,idx])
    loc0 = endpts[i_end]
    n = neuron.nodes.shape[0]
    if i_end == 1:
        i1 = n-1 if n<=10 else 10
    else:
        i1 = n-10 if n>10 else 0
    loc1 = numpy.array(neuron.nodes.loc[i1,["x","y","z"]].tolist())
    vec = (loc1-loc0)/numpy.linalg.norm(loc1-loc0)
    
    return KeyPoint(name=name,location=loc0,vector=vec)


def filter_skeletons(neurons,names=None,ids=None,mincablelength=None,minradius=None,**kwargs):
    if not names is None:
        neurons = [n for n in neurons if n.name in names]
    if not ids is None:
        neurons = [n for n in neurons if n.id in ids]
    if not mincablelength is None:
        neurons = [n for n in neurons if n.cable_length >= mincablelength]
    if not minradius is None:
        neurons = [n for n in neurons if n.nodes.radius.mean() >= minradius]
    if neurons:
        neurons = navis.NeuronList(neurons)
    return neurons


def filter_surface_keypoints(KeyPtList,distance=0,ori=None,surf_map=None,**kwargs):
    axis, sign, idx = ori_lookup(ori)
    indices = [0,1,2]
    indices.pop(idx)
    if surf_map is None:
        print("Surface map not provided: defaulting to extremal node")
        hlist = numpy.array([keypt.location[idx] for keypt in KeyPtList])
    else:
        print("Interpolating surface map")
        interp = RGI((numpy.arange(surf_map.shape[0]),numpy.arange(surf_map.shape[1])),surf_map,method="nearest")
    if sign == "POS":
        if surf_map is None:
            hmax = hlist.max()
            good = numpy.nonzero(hlist>=hmax-distance)[0]
        else:
            locs = numpy.array([keypt.location for keypt in KeyPtList])
            good = numpy.nonzero(locs[:,idx] >= interp(numpy.array([locs[:,indices[0]],locs[:,indices[1]]]).transpose()) - distance)[0]
    elif sign == "NEG":
        if surf_map is None:
            hmin = hlist.min()
            good = numpy.nonzero(hlist<=hmin+distance)[0]
        else:
            locs = numpy.array([keypt.location for keypt in KeyPtList])
            good = numpy.nonzero(locs[:,idx] <= interp(numpy.array([locs[:,indices[0]],locs[:,indices[1]]]).transpose()) + distance)[0]
    SurfList = [KeyPtList[g] for g in good]
    return SurfList

    
def generate_keypoint_file(swcpath,outputpath,ori=None,surf_file='',**kwargs):
    if surf_file:
        surf = numpy.load(surf_file)
    else:
        surf = None
    skels = navis.read_swc(swcpath)
    neurons = filter_skeletons(skels,ori=ori,**kwargs)
    keypts = [keypoint_from_neuron(neuron,ori) for neuron in neurons]
    surfkeypts = filter_surface_keypoints(keypts,ori=ori,surf_map=surf,**kwargs)
    write_keypoints_to_file(surfkeypts,outputpath)

