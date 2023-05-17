# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:45:56 2023

@author: kevint
"""
import pathlib
import argschema
import json
import zarr
import numpy as np
from skimage.io import imsave


def write_cutout_metadata(jsonpath,
                          zarrFile,
                          tileGroup,
                          zstart,
                          ystart,
                          xstart,
                          **kwargs):
    js = {
        "zarr_file": zarrFile,
        "tile_group": tileGroup,
        "cutout_zyx": [zstart,ystart,xstart]
        }
    with open(jsonpath,"w+") as f:
        json.dump(js,f,indent=4)


def write_cutout_to_tiff(tiffpath,
                         dataset,
                         zstart,
                         zlength,
                         ystart,
                         ylength,
                         xstart,
                         xlength):
    dsdims = dataset.shape
    if zlength < 1:
        zlength = dsdims[2] - zstart
    if ylength < 1:
        ylength = dsdims[3] - ystart
    if xlength < 1:
        xlength = dsdims[4] - xstart
    for i in range(zlength):
        frame = dataset[0,0,zstart+i,ystart:ystart+ylength,xstart:xstart+xlength]
        imsave(tiffpath,frame,append=True,bigtiff=True)


def get_zarr_group(zpath, grpname):
    # key to working with zarr files
    # group contains mip datasets and dataset attributes
    zf = zarr.open(zpath)
    return zf[grpname]


def get_lvl0_dataset(zpath,grp):
    ds = get_zarr_group(zpath,grp)[0]
    return ds


def create_cutout_from_zarr(zarrFile,
                            tileGroup,
                            outputFile,
                            outputFormat="tif",
                            mdpath="",
                            **kwargs):
    zpath = pathlib.Path(zarrFile)
    if zpath.exists():
        ds = get_lvl0_dataset(zarrFile,tileGroup)
        if outputFormat == "tif":
            write_cutout_to_tiff(outputFile,ds,**kwargs)
        if mdpath:
            write_cutout_metadata(mdpath,zarrFile,tileGroup,**kwargs)


class CutoutParameters(argschema.schemas.DefaultSchema):
    z_start = argschema.fields.Int(required=False, default=0)
    z_length = argschema.fields.Int(required=False, default=0)
    y_start = argschema.fields.Int(required=False, default=0)
    y_length = argschema.fields.Int(required=False, default=0)
    x_start = argschema.fields.Int(required=False, default=0)
    x_length = argschema.fields.Int(required=False, default=0)


class CreateCutoutInputParameters(argschema.ArgSchema, CutoutParameters):
    zarr_file = argschema.fields.Str(required=True)
    tile_group = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)
    output_format = argschema.fields.Str(required=False,default="tif")
    output_json = argschema.fields.Str(required=False,default="")


class CreateCutoutParser(argschema.ArgSchemaParser):
    default_schema = CreateCutoutInputParameters

    def run(self):
        create_cutout_from_zarr(
            self.args["zarr_file"],
            self.args["tile_group"],
            self.args["output_file"],
            outputFormat = self.args["output_format"],
            mdpath = self.args["output_json"],
            zstart = self.args["z_start"],
            zlength = self.args["z_length"],
            ystart = self.args["y_start"],
            ylength = self.args["y_length"],
            xstart = self.args["x_start"],
            xlength = self.args["x_length"]
            )


if __name__ == "__main__":
    mod = CreateCutoutParser()
    mod.run()
