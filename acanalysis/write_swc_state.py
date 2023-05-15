# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:45:34 2023

@author: kevint
"""

import pathlib
import argschema
import json
import zarr
import numpy as np


def get_zarr_params(zpath):
    zf = zarr.open(zpath)
    attrs = zf.attrs.asdict()
    grptr = attrs['multiscales'][0]['coordinateTransformations'][0]['translation']
    mipsc = attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    params = {
        "scale": [mipsc[i] for i in range(2, 5)],
        "translation": [int(grptr[i]/mipsc[i]) for i in range(2, 5)]
    }
    return params


def get_scale_dims(zparams):
    sc = zparams["scale"]
    scaleDims = {
        "z": [sc[0], "um"],
        "y": [sc[1], "um"],
        "x": [sc[2], "um"]
    }
    return scaleDims


def get_tr_matrix(zparams,offset):
    trList = zparams["translation"]
    trM = [
        [
            1,
            0,
            0,
            trList[0] + offset[0]
        ],
        [
            0,
            1,
            0,
            trList[1] + offset[1]
        ],
        [
            0,
            0,
            1,
            trList[2] + offset[2]
        ]
    ]
    return trM


def create_seg_layer(precompDir, zparams, cutout_zyx=[0,0,0]):
    scaleDims = get_scale_dims(zparams)
    trMatrix = get_tr_matrix(zparams,offset=cutout_zyx)
    slayer = {
        "type": "segmentation",
        "source": {
            "url": precompDir,
            "transform": {
                "matrix": trMatrix,
                "outputDimensions": scaleDims,
                "inputDimensions": scaleDims
            }
        },
        "tab": "source",
        "segments": [],
        "name": "skeletons"
    }
    return slayer


def write_swc_state(zarrFile, precompDir, outputFile, **kwargs):
    zpath = pathlib.Path(zarrFile)
    if zpath.exists():
        zparams = get_zarr_params(zpath)
        slayer = create_seg_layer(precompDir, zparams, **kwargs)
        with open(outputFile, "w+") as f:
            json.dump(slayer, f, indent=4)
            
            
class SwcCoordinateParameters(argschema.schemas.DefaultSchema):
    cutout_zyx = argschema.fields.List((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=[0, 0, 0])


class WriteSwcNgStateInputParameters(argschema.ArgSchema, SwcCoordinateParameters):
    zarr_file = argschema.fields.Str(required=True)
    precomputed_dir = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)


class WriteSwcNgStateParser(argschema.ArgSchemaParser):
    default_schema = WriteSwcNgStateInputParameters

    def run(self):
        write_swc_state(
            self.args["zarr_file"],
            self.args["precomputed_dir"],
            self.args["output_file"],
            cutout_zyx = self.args["cutout_zyx"])


if __name__ == "__main__":
    mod = WriteSwcNgStateParser()
    mod.run()