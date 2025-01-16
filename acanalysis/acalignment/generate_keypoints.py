# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:50:02 2023

@author: kevint
"""

import navis
import numpy
from scipy.interpolate import RegularGridInterpolator as RGI

from acanalysis.acalignment.keypoints import KeyPoint,write_keypoints_to_file
from acanalysis.acalignment.utils.swc_utils import read_neurons_from_file,ori_table,read_navis_neurons_tar,preprocess_filter_neuron_by_cutout


def ori_lookup(ori):
    return ori_table(ori)


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
    nend = int(8/(2**swcmip))
    if i_end == 1:
        i1 = n-1 if n<=nend else nend
    else:
        i1 = n-nend if n>nend else 0
    loc1 = neuron.nodes.loc[i1,["x","y","z"]].to_numpy()
    norm = numpy.linalg.norm(loc1-loc0)
    if norm > 0:
        vec = (loc1-loc0)/norm
    else:
        # print(str(loc0) + " to " + str(loc1))
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


def filter_surface_keypoints(keypts,distance=0,ori=None,surf_map=None,surf_grid=None,roi_coords=None,**kwargs):
    axis, sign, idx = ori_lookup(ori)
    # indices = [0,1,2]
    # indices.pop(idx)
    if surf_map is None:
        print("Surface map not provided: defaulting to extremal node")
        hlist = numpy.array([keypt.location[0] for keypt in keypts])
    else:
        print("Interpolating surface map")
        if surf_grid is None:
            gridy = numpy.arange(surf_map.shape[0])
            gridx = numpy.arange(surf_map.shape[1])
        elif len(surf_grid) == 2 and type(surf_grid[0]) == int:
            gridy = surf_grid[0] + numpy.arange(surf_map.shape[0])
            gridx = surf_grid[1] + numpy.arange(surf_map.shape[1])
        else:
            gridy = surf_grid[0]
            gridx = surf_grid[1]
        interp = RGI((gridy,gridx),surf_map,method="nearest")
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
            KeyPtList = [kp for kp in keypts if not kp.vector is None and kp.location[1] > gridy[0] and kp.location[1] < gridy[-1] and kp.location[2] > gridx[0] and kp.location[2] < gridx[-1]]
    print(str(len(KeyPtList)) + " within interp grid")
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
                           z_range=None,
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
        if z_range is None:
            offset = (0,0)
        else:
            offset = (0,z_range[0]*(2**swcmip))
    else:
        surf = None

    if z_range is None:
        preprocess_func = lambda x:x
    else:
        preprocess_func = lambda n: preprocess_filter_neuron_by_cutout(n,cutout={"z":z_range,"y":[0,576],"x":[0,576]})
    print("reading from " + str(swcpath))
    if is_tar:
        print("multiprocess read")
        neurons = read_navis_neurons_tar(swcpath,concurrency=20,preprocess_func=preprocess_func)
    else:
        neurons = read_neurons_from_file(swcpath,is_tar=is_tar,prefix=tile_name,swap_xyz=swap_xyz,ori=ori,z_range=z_range,**kwargs)
    #print(str(skels.shape[0]) + " initial")
    #neurons = filter_skeletons(skels,**kwargs)
    print(str(neurons.shape[0]) + " filtered")
    keypts = [keypoint_from_neuron(neuron,name=tile_name+str(neuron.id),ori=ori,swcmip=swcmip) for neuron in neurons]
    print(len(keypts))
    write_keypoints_to_file(keypts,outputpath)
    surfkeypts = filter_surface_keypoints(keypts,ori=ori,surf_map=surf,surf_grid=offset,**kwargs)
    write_keypoints_to_file(surfkeypts,outputpath)
    print("saved keypoints to " + str(outputpath))

