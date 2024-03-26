# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:50:02 2023

@author: kevint
"""

import navis
import numpy
from scipy.interpolate import RegularGridInterpolator as RGI

from acanalysis.acalignment.keypoints import KeyPoint,write_keypoints_to_file
from acanalysis.acalignment.utils import read_neurons_from_file, patch_axon_ids

def ori_lookup(ori):
    oriTuple = {
        "+x": ("x", "POS", 0),
        "-x": ("x", "NEG", 0),
        "+z": ("z", "POS", 2),
        "-z": ("z", "NEG", 2)
    }[ori]
    return oriTuple


def keypoint_from_neuron(neuron,name='',ori=None,swcmip=0):
    """generate keypoint from neuron skeleton (axon)
    
    Parameters
    ----------
    neuron : navis.TreeNeuron
        skeleton representing single axon
    name : str
        keypoint name associated with generating skeleton
    ori : str
        character string defining section surface axis and direction
    swcmip : int
        mip level of skeletons relative to image data

    Returns
    ------
    KeyPoint : keypoints.KeyPoint
        dataclass storing name, 3D surface location, and 3D impact vector
    """
    if not name:
        name = str(neuron.name)
    axis, sign, idx = ori_lookup(ori)
    endpts = numpy.vstack([neuron.leafs[["x", "y", "z"]], neuron.nodes[neuron.nodes.node_id == neuron.root.flatten()[0]][["x", "y", "z"]]])
    loc_func = {
        "POS": numpy.argmax,
        "NEG": numpy.argmin
    }[sign]
    i_end = loc_func(endpts[:,idx])
    loc0 = endpts[i_end]
    n = neuron.nodes.shape[0]
    nend = int(10/(2**swcmip))
    if i_end == 1:
        i1 = n-1 if n<=nend else nend
    else:
        i1 = n-nend if n>nend else 0
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
    filtered = []
    for n in neurons:
        good = True
        if not names is None:
            good = n.name in names
        if not ids is None:
            good = n.id in ids
        if not mincablelength is None:
            good = n.cable_length >= mincablelength
        if not minradius is None:
            good = n.nodes.radius.mean() >= minradius
        if good:
            filtered.append(n)
    if filtered:
        return navis.NeuronList(filtered)
    return None


def filter_surface_keypoints(keypts,distance=0,ori=None,surf_map=None,roi_coords=None,**kwargs):
    axis, sign, idx = ori_lookup(ori)
    # indices = [0,1,2]
    # indices.pop(idx)
    if surf_map is None:
        print("Surface map not provided: defaulting to extremal node")
        hlist = numpy.array([keypt.location[0] for keypt in keypts])
    else:
        print("Interpolating surface map")
        interp = RGI((numpy.arange(surf_map.shape[0]),numpy.arange(surf_map.shape[1])),surf_map,method="nearest")
    KeyPtList = []
    if not roi_coords is None:
        coord_axes = [a for a in range(len(roi_coords)) if roi_coords[a]]
        c0 = numpy.zeros(len(roi_coords))
        for a in coord_axes:
            c0[a] = roi_coords[a][0]
        print(c0)
        for kp in keypts:
            if not kp.vector is None:
                loc = kp.location
                if all([(loc[a]>=roi_coords[a][0])and(loc[a]<roi_coords[a][1]) for a in coord_axes]):
                    kp.location -= c0
                    if (kp.location[2]>=0) and (kp.location[2] < surf_map.shape[1]):
                        KeyPtList.append(kp)
                    else:
                        print(kp.location)
    else:
        if surf_map is None:
            KeyPtList = [kp for kp in keypts if not kp.vector is None]
        else:
            KeyPtList = [kp for kp in keypts if (not kp.vector is None) and (kp.location[2] < surf_map.shape[1])]
    print(len(KeyPtList))
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
    print(str(len(SurfList)) + " surface points")
    return SurfList

    
def generate_keypoint_file(swcpath,
                           outputpath,
                           is_tar=False,
                           swcmip=0,
                           ori=None,
                           swap_xyz=[],
                           tile_name='',
                           surf_file='',
                           **kwargs):
    """write json file containing list of keypoints generated from all skeletons
    
    Parameters
    ----------
    swcpath : Path or Path str
        path to .swc or .gz.tar file containing skeletons
    outputpath : Path or Path str
        path for output json
    is_tar : bool
        flag for loading .gz.tar archive
    swcmip : int
        mip level of skeletons relative to image data
    ori : str
        character string defining section surface axis and direction
    swap_xyz : list of str
        permutation of axes to swap (is this needed?)
    tile_name : str
        optional tile id to add as prefix to keypoint names
    surf_file : Path or Path str
        path to .npy file containing surface map

    Returns
    ------
    KeyPoint : keypoints.KeyPoint
        dataclass storing name, 3D surface location, and 3D impact vector
    """
    if surf_file:
        surf = numpy.load(surf_file)
    else:
        surf = None
    skels = read_neurons_from_file(swcpath,is_tar=is_tar)
    if swap_xyz:
        for sk in skels:
            sk.nodes[["x","y","z"]] = sk.nodes[swap_xyz]
        print("[x,y,z] axes permuted as " + str(swap_xyz))
    print(str(skels.shape[0]) + " initial")
    neurons = filter_skeletons(skels,**kwargs)
    print(str(neurons.shape[0]) + " filtered")
    patch_axon_ids(neurons)
    keypts = [keypoint_from_neuron(neuron,name=tile_name+str(neuron.id),ori=ori,swcmip=swcmip) for neuron in neurons]
    surfkeypts = filter_surface_keypoints(keypts,ori=ori,surf_map=surf,**kwargs)
    write_keypoints_to_file(surfkeypts,outputpath)
    print("saved keypoints to " + str(outputpath))

