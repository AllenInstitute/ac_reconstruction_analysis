# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:04:15 2024

@author: kevint
"""

from pathlib import Path
import json
import pandas
import json


def get_src_from_json(sourcejson,plane,tile):
    with open(sourcejson,'r') as f:
        js = json.load(f)
    srcList = js[plane]['sources']
    ind = [s.split("_")[-1] for s in srcList].index(tile)
    return srcList[ind]

def surf_inputs_from_alignjs(alignjs,
                             miplvl=0):
    inputs = []
    outputs = []
    outpath = Path(alignjs.get("output_path"))
    for s in alignjs.get("sections"):
        jspath = get_json_path_from_roi(s)
        print(str(jspath),jspath.exists())
        if jspath.exists():
            ori = s.get("ori")
            oristr = str_from_ori(ori)
            surfsup = (oristr == "POS")
            for i,tid in enumerate(s.get("tiles")):
                roi_id = get_roi_label(s,i)
                zpath = get_src_from_json(jspath,s.get("plane_id"),tid)
                cutout = s["cutouts"][i]
                outputpath = outpath / Path(roi_id + "_" + oristr + ".npy")
                kwargs = {
                    "zarr_path":zpath,
                    "cutout":cutout,
                    "miplvl":miplvl,
                    "surfsup":surfsup
                }
                inputs.append(kwargs)
                outputs.append(outputpath)
    return inputs,outputs


def gkp_inputs_from_alignjs_old(alignjs,
                            is_tar=False,
                            swap_xyz=None,
                            mincablelength=None,
                            minradius=None,
                            distance=0,
                            z_range=None):
    swap_xyz = [] if swap_xyz is None else swap_xyz
    inputs = []
    outpath = Path(alignjs.get("output_path"))
    for s in alignjs.get("sections"):
        ori = s.get("ori")
        oristr = str_from_ori(ori)
        #surfsup = (oristr == "POS")
        swcmip = s.get("swc_mip")
        #co_coords = s["cutouts"][0]
        #roi_coords = [[],[],[co_coords["z"][0],co_coords["z"][1]]]
        roi_coords = None # no roi means use all skeletons in swc
        if "swc_kwargs" in s:
            swckwargs = s.get("swc_kwargs")
            is_tar = swckwargs.get("is_tar")
        else:
            swckwargs = None
        for i,tid in enumerate(s.get("tiles")):
            roi_id = get_roi_label(s,i)
            if is_tar:
                swcpath = get_tar_path(s,i)
            else:
                swcpath = get_swc_path(s,i)
            surfmap = outpath / Path(roi_id + "_" + oristr + ".npy")
            outputpath = outpath / Path(roi_id + "_keypoints.json")
            kwargs = {
                "swcpath":swcpath,
                "outputpath":outputpath,
                "is_tar":is_tar,
                "swcmip":swcmip,
                "ori":ori,
                "swap_xyz":swap_xyz,
                "tile_name":roi_id+"_",
                "surf_file":surfmap,
                "roi_coords":roi_coords,
                "mincablelength":mincablelength,
                "minradius":minradius,
                "distance":distance,
                "z_range":z_range
            }
            if not swckwargs is None:
                for key in swckwargs:
                    kwargs[key] = swckwargs.get(key)
            inputs.append(kwargs)
    return inputs


def gkp_inputs_from_alignjs(alignjs,
                            swap_xyz=None,
                            mincablelength=None,
                            minradius=None,
                            distance=0,
                            z_range=None):
    swap_xyz = [] if swap_xyz is None else swap_xyz
    inputs = []
    outpath = Path(alignjs.get("output_path"))
    for s in alignjs.get("sections"):
        ori = s.get("ori")
        oristr = str_from_ori(ori)
        #surfsup = (oristr == "POS")
        swcmip = s.get("swc_mip")
        #co_coords = s["cutouts"][0]
        #roi_coords = [[],[],[co_coords["z"][0],co_coords["z"][1]]]
        roi_coords = None # no roi means use all skeletons in swc
        if "swc_kwargs" in s:
            swckwargs = s.get("swc_kwargs")
        else:
            swckwargs = None
        for i,tid in enumerate(s.get("tiles")):
            roi_id = get_roi_label(s,i)
            if is_tar:
                swcpath = get_tar_path(s,i)
            else:
                swcpath = get_swc_path(s,i)
            surfmap = outpath / Path(roi_id + "_" + oristr + ".npy")
            outputpath = outpath / Path(roi_id + "_keypoints.json")
            kwargs = {
                "swcpath":swcpath,
                "outputpath":outputpath,
                "is_tar":is_tar,
                "swcmip":swcmip,
                "ori":ori,
                "swap_xyz":swap_xyz,
                "tile_name":roi_id+"_",
                "surf_file":surfmap,
                "roi_coords":roi_coords,
                "mincablelength":mincablelength,
                "minradius":minradius,
                "distance":distance,
                "z_range":z_range
            }
            if not swckwargs is None:
                for key in swckwargs:
                    kwargs[key] = swckwargs.get(key)
            inputs.append(kwargs)
    return inputs

def get_tar_path(args,i):
    tid = args.get("tiles")[i]
    return Path(args.get("swc_path")) / Path("highres_" + tid + ".swcs.tar.gz")

def get_swc_path(args,i):
    tid = args.get("tiles")[i]
    #return Path(args.get("swc_path")) / Path(args.get("section_id") + "_" + args.get("acq_id")) / Path(tid + ".swc")
    return Path(args.get("swc_path")) / Path(tid + ".swc")

def get_json_path_from_roi(args):
    return Path(args["ac_datasets_path"]) / Path(args["section_id"]) / Path(args["section_id"] + "_" + args["acq_id"] + "_highres.json")

def get_swc_path_from_roi(args,i):
    return Path(args["swc_path"]) / Path(args["tiles"][i] + ".swc")
    #return Path(args["swc_path"]) / Path(args["section_id"] + "_" + args["acq_id"]) / Path(args["tiles"][i] + ".swc")

def get_section_label(args):
    return args["section_id"] + "_" + args["acq_id"] + "_" + args["plane_id"]

def get_roi_label(args,i):
    return get_section_label(args) + "_" + args["tiles"][i]

def str_from_ori(ori):
    if ori[0] == "+":
        return "POS"
    else:
        return "NEG"

def create_mat_json(matjspath,
                    alignjs,
                    output_path=".",
                    miplvl=1,
                    matrixfiles=["","model.npy"],
                    **kwargs):
    sections = []
    for i,s in enumerate(alignjs["sections"]):
        ntiles = len(s["tiles"])
        oristr = str_from_ori(s["ori"])
        sections.append({
            "align_path":alignjs["output_path"],
            "mip_level":miplvl,
            "sources":[get_src_from_json(get_json_path_from_roi(s),s["plane_id"],tid) for tid in s["tiles"]],
            "surface_maps":[get_roi_label(s,n) + "_" + oristr + ".npy" for n in range(ntiles)],
            "matrix_file":matrixfiles[i]
        })
    matjs = {
        "output_path":output_path,
        "sections":sections
    }
    matjs.update(kwargs)
    with open(matjspath,"w+") as f:
        json.dump(matjs,f,indent=4)


import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import maximum_filter, median_filter,convolve, binary_dilation
from scipy.interpolate import griddata,RBFInterpolator


def my_threshold(data):
    thresh = threshold_otsu(data)
    #thresh = np.percentile(data,95)
    return thresh

def preconvolve(data,size):
    convolved = np.empty(data.shape,dtype=data.dtype)
    k = np.ones((size,size))
    for i in range(data.shape[0]):
        convolved[i,...] = convolve(data[i,...],k,mode="constant",cval=0)
    return convolved

def predilation(data,radius):
    d = int(2*radius + 1)
    k = np.zeros((d,d,d),dtype=int)
    r = radius
    for x in range(d):
        for y in range(d):
            for z in range(d):
                if (x-r)**2 + (y-r)**2 + (z-r)**2 <= r**2:
                    k[x,y,z] = 1
    return binary_dilation(data,structure=k).astype(int)

def premedian(data,size):
    medianed = np.empty(data.shape,dtype=data.dtype)
    for i in range(data.shape[0]):
        medianed[i,...] = median_filter(data[i,...],size=(size,size))
    return medianed

def premax(data,size):
    M = np.empty(data.shape,dtype=data.dtype)
    for i in range(data.shape[0]):
        M[i,...] = maximum_filter(data[i,...],size=(size,size))
    return M

def get_first_z(maskstack,flip=False):
    M = maskstack.astype(int)
    if flip:
        M = np.flip(maskstack,axis=0)
    dims = M.shape # (Nz,Nx,Ny)
    zs = np.zeros((dims[1],dims[2]),dtype='int')
    
    for i1 in range(dims[1]):
        for i2 in range(dims[2]):
            z = np.nonzero(M[:,i1,i2])
            if len(z[0])>0:
                if not flip:
                    zs[i1,i2] = z[0][0]
                else:
                    if z[0][0] > 0:
                        zs[i1,i2] = dims[0] - z[0][0] - 1
                    else:
                        zs[i1,i2] = 0
            else:
                zs[i1,i2] = 0
            
    return zs

def make_surface_map_tps(zIn,gridsize,miplvl,surfsup=False):
    dims = zIn.shape
    d0 = int(np.floor(dims[0]/gridsize[0]))
    d1 = int(np.floor(dims[1]/gridsize[1]))
    surfmap = np.zeros(gridsize,dtype=int)
    mapy = np.zeros(gridsize,dtype=int)
    mapx = np.zeros(gridsize,dtype=int)
    for i0 in range(gridsize[0]):
        for i1 in range(gridsize[1]):
            zIni = zIn[d0*i0:d0*(i0+1),d1*i1:d1*(i1+1)]
            if surfsup:
                z = np.max(zIni[zIni>0]) if np.any(zIni>0) else 0
                y,x = np.unravel_index(np.argmax(zIni, axis=None), zIni.shape)
            else:
                z = np.min(zIni[zIni>0]) if np.any(zIni>0) else 0
                y,x = np.unravel_index(np.argmin(zIni, axis=None), zIni.shape)
            surfmap[i0,i1] = z*(2**miplvl)
            mapy[i0,i1] = int((d0*i0 + y)*(2**miplvl))
            mapx[i0,i1] = int((d1*i1 + x)*(2**miplvl))
    return surfmap,mapy,mapx


#surfsup = False for S32, True for S33
def detect_surface(voldata,z_range,y_range,miplvl,surfsup=False):
    zlength = z_range[1] - z_range[0]
    ylength = y_range[1] - y_range[0]
    A = voldata
    thresh = my_threshold(A)
    print(thresh)
    B = A > thresh
    C = premax(premedian(B.astype(int),size=5),size=10)
    D = predilation(C,radius=4)
    E = get_first_z(D.transpose((2,1,0)),flip=surfsup)
    Z,Y,X = make_surface_map_tps(E,gridsize=(10,20),miplvl=miplvl,surfsup=surfsup)
    tpsy,tpsx = np.meshgrid(np.arange(ylength,step=2**miplvl,dtype=int),np.arange(zlength,step=2**miplvl,dtype=int),indexing='ij')
    tpsyx = np.hstack((tpsy.flatten()[:,np.newaxis],tpsx.flatten()[:,np.newaxis]))
    YX = np.concatenate((Y[Z>0].flatten()[:,np.newaxis],X[Z>0].flatten()[:,np.newaxis]),axis=1)
    F = RBFInterpolator(YX,Z[Z>0].flatten(),smoothing=1,kernel='thin_plate_spline')(tpsyx)
    gridy,gridx = np.meshgrid(np.arange(ylength,dtype=int),np.arange(zlength,dtype=int),indexing='ij')
    G = griddata((tpsy.flatten(),tpsx.flatten()),F,(gridy,gridx),method='nearest')
    return G