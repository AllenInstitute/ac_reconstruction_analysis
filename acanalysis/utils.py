# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:01:37 2023

@author: kevint
"""
import zarr
import dataclasses
from skimage.io import imsave


@dataclasses.dataclass
class zarr_parameters:
    scale: list
    translation: list


def get_zarr_group(zpath, grpname):
    # key to working with zarr files
    # group contains mip datasets and dataset attributes
    zf = zarr.open(zpath)
    return zf[grpname]


def get_zarr_params(zpath,grp):
    zg = get_zarr_group(zpath,grp)
    attrs = zg.attrs.asdict()
    grptr = attrs['multiscales'][0]['coordinateTransformations'][0]['translation']
    mipsc = attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    zparams = zarr_parameters
    zparams.scale = [mipsc[i] for i in range(2, 5)]
    zparams.translation = [int(grptr[i]/mipsc[i]) for i in range(2, 5)]
    return zparams


#TODO: replace with acpreprocessing.io function
def write_tiff_vol_append(tiffpath,dataset,zstart,zlength,ystart,ylength,xstart,xlength):
    for i in range(zlength):
        frame = dataset[0,0,zstart+i,ystart:ystart+ylength,xstart:xstart+xlength]
        imsave(tiffpath,frame,append=True,bigtiff=True)