# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:50:02 2023

@author: kevint
"""

import navis
import numpy
from scipy.interpolate import RegularGridInterpolator as RGI

from acanalysis.acalignment.keypoints import KeyPoint,write_keypoints_to_file
from acanalysis.splitup_swc import get_axon_list_from_subtrees

def ori_lookup(ori):
    oriTuple = {
        "+x": ("x", "POS", 0),
        "-x": ("x", "NEG", 0),
        "+z": ("z", "POS", 2),
        "-z": ("z", "NEG", 2)
    }[ori]
    return oriTuple


def keypoint_from_neuron(neuron, ori=None, swcmip=0):
    name = str(neuron.id)
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
    norm = numpy.linalg.norm(loc1-loc0)
    if norm > 0:
        vec = (loc1-loc0)/norm
    else:
        vec = None
    if idx == 0:
        location = loc0
    elif idx == 2:
        location = loc0[[2,1,0]]
        if not vec is None:
            vec = vec[[2,1,0]]
    location *= 2**swcmip
    vector = vec
    
    return KeyPoint(name=name,location=location,vector=vector)


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


def filter_surface_keypoints(keypts,distance=0,ori=None,surf_map=None,roi_coords=None,**kwargs):
    axis, sign, idx = ori_lookup(ori)
    # indices = [0,1,2]
    # indices.pop(idx)
    if not roi_coords is None:
        coord_axes = [a for a in range(len(roi_coords)) if roi_coords[a]]
        c0 = numpy.zeros(len(roi_coords))
        for a in coord_axes:
            c0[a] = roi_coords[a][0]
        print(c0)
        KeyPtList = []
        for kp in keypts:
            if not kp.vector is None:
                loc = kp.location
                if all([(loc[a]>=roi_coords[a][0])and(loc[a]<roi_coords[a][1]) for a in coord_axes]):
                    kp.location -= c0
                    if kp.location[2] > 4000:
                        print(kp.location)
                    KeyPtList.append(kp)
    else:
        KeyPtList = [kp for kp in keypts if (not kp.vector is None) and (kp.location[2]<4000)]
    if surf_map is None:
        print("Surface map not provided: defaulting to extremal node")
        hlist = numpy.array([keypt.location[0] for keypt in KeyPtList])
    else:
        print("Interpolating surface map")
        interp = RGI((numpy.arange(surf_map.shape[0]),numpy.arange(surf_map.shape[1])),surf_map,method="nearest")
    if sign == "POS":
        if surf_map is None:
            hmax = hlist.max()
            good = numpy.nonzero(hlist>=hmax-distance)[0]
        else:
            locs = numpy.array([keypt.location for keypt in KeyPtList])
            good = numpy.nonzero(locs[:,0] >= interp(numpy.array([locs[:,1],locs[:,2]]).transpose()) - distance)[0]
    elif sign == "NEG":
        if surf_map is None:
            hmin = hlist.min()
            good = numpy.nonzero(hlist<=hmin+distance)[0]
        else:
            locs = numpy.array([keypt.location for keypt in KeyPtList])
            good = numpy.nonzero(locs[:,0] <= interp(numpy.array([locs[:,1],locs[:,2]]).transpose()) + distance)[0]
    SurfList = [KeyPtList[g] for g in good]
    return SurfList

    
def generate_keypoint_file(swcpath,outputpath,swcmip=0,ori=None,tile_name='',surf_file='',roi_coords=None,**kwargs):
    if surf_file:
        surf = numpy.load(surf_file)
    else:
        surf = None
    #swc_it = iterate_swc_chunks(swcpath)
    #skels = navis.read_swc(swc_it)
    skels = get_axon_list_from_subtrees(navis.read_swc(swcpath))
    neurons = filter_skeletons(skels,ori=ori,**kwargs)
    keypts = [keypoint_from_neuron(neuron,ori,swcmip) for neuron in neurons]
    surfkeypts = filter_surface_keypoints(keypts,ori=ori,surf_map=surf,roi_coords=roi_coords,**kwargs)
    write_keypoints_to_file(surfkeypts,outputpath,tile_name)

