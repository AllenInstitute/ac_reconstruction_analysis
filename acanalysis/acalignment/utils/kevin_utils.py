# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:04:15 2024

@author: kevint
"""

from pathlib import Path
import json
from acpreprocessing.stitching_modules.acstitch.zarrutils import get_src_from_json
import pandas

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


def gkp_inputs_from_alignjs(alignjs,
                            is_tar=False,
                            swap_xyz=None,
                            mincablelength=None,
                            minradius=None,
                            distance=0):
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
                "distance":distance
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
                    matrixfiles=["","model.npy"]):
    sections = []
    for i,s in enumerate(alignjs["sections"]):
        ntiles = len(s["tiles"])
        oristr = str_from_ori(s["ori"])
        sections.append({
            "base_path":alignjs["output_path"],
            "sources":[get_src_from_json(get_json_path_from_roi(s),s["plane_id"],tid) for tid in s["tiles"]],
            "surface_maps":[get_roi_label(s,n) + "_" + oristr + ".npy" for n in range(ntiles)],
            "matrix_file":matrixfiles[i]
        })
    matjs = {
        "output_path":alignjs["output_path"],
        "sections":sections
    }
    with open(matjspath,"w+") as f:
        json.dump(matjs,f,indent=4)