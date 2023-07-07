""" Save cutout from zarr tile group as tiff stack for segmentation

"""
import pathlib
import argschema
import json
from utils import get_zarr_params, get_miplvl_dataset, write_tiff_vol_append


example_input = {
    "zarr_file": "/ACdata/Users/kevin/ispim_ome_zarr/H17_x55_S32_230412_highres/H17_x55_S32_230412_highres.zarr",
    "tile_group": "highres_Pos79",
    "output_file": "/ACdata/Users/kevin/tiffs/S32_Pos79.tif",
    "output_format": "tif",
    "output_json": "/ACdata/Users/kevin/tiffs/S32_highres_Pos79_2.json",
    "mip_lvl": 0,
    "z_start": 24000,
    "z_length": 4000,
    "y_start": 0,
    "y_length": 400,
    "x_start": 0,
    "x_length": 576,
    "downsample_factor": 1
    }


def write_cutout_metadata(jsonpath,
                          zarrFile,
                          tileGroup,
                          miplvl,
                          zstart,
                          ystart,
                          xstart,
                          dsfactor,
                          **kwargs):
    """write metadata .json file with cutout parameters 

    Parameters
    ----------
    jsonpath : str
        path string to output .json
    zarrFile : str
        path string to image data source zarr
    tileGroup : str
        name of zarr group containing position source data
    miplvl : int
        mip dataset level
    zstart : int
        cutout starting index axis 0
    ystart : int
        cutout starting index axis 1
    xstart : int
        cutout starting index axis 2
    dsfactor : int
        additional downsampling on top of mip
    """
    zparams = get_zarr_params(zarrFile,tileGroup)
    zscale = zparams.scale
    scale_zyx = [zs*dsfactor*(2**miplvl) for zs in zscale]
    js = {
        "zarr_file": zarrFile,
        "tile_group": tileGroup,
        "mip_lvl": miplvl,
        "cutout_zyx": [zstart,ystart,xstart],
        "scale_zyx_um": scale_zyx
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
                         xlength,
                         dsfactor):
    """write tiff stack from cutout of zarr dataset 

    Parameters
    ----------
    tiffpath : str
        path string to output .tif file
    dataset : zarr.dataset
        source dataset for cutout
    tileGroup : str
        name of zarr group containing position source data
    miplvl : int
        mip dataset level
    zstart : int
        cutout starting index axis 0
    zlength : int
        cutout range axis 0
    ystart : int
        cutout starting index axis 1
    ylength : int
        cutout range axis 1
    xstart : int
        cutout starting index axis 2
    xlength : int
        cutout range axis 2
    dsfactor : int
        additional downsampling on top of mip (not yet implemented)
    """
    dsdims = dataset.shape
    if zlength < 1:
        zlength = dsdims[2] - zstart
    if ylength < 1:
        ylength = dsdims[3] - ystart
    if xlength < 1:
        xlength = dsdims[4] - xstart
    write_tiff_vol_append(tiffpath,dataset,zstart,zlength,ystart,ylength,xstart,xlength)


def create_cutout_from_zarr(zarrFile,
                            tileGroup,
                            outputFile,
                            outputFormat="tif",
                            mdpath="",
                            miplvl=0,
                            **kwargs):
    """write tiff stack from cutout of zarr dataset 

    Parameters
    ----------
    zarrFile : str
        path string to image data source zarr
    tileGroup : str
        name of zarr group containing position source data
    outputFile : str
        path string to output .tif file
    outputFormat : str
        format string (only tif)
    mdpath : str
        path string to output metadata .json
    miplvl : int
        dataset mip level to write
    """
    zpath = pathlib.Path(zarrFile)
    if zpath.exists():
        ds = get_miplvl_dataset(zarrFile,tileGroup,miplvl)
        if outputFormat == "tif":
            write_cutout_to_tiff(outputFile,ds,**kwargs)
        if mdpath:
            write_cutout_metadata(mdpath,zarrFile,tileGroup,miplvl,**kwargs)


class CutoutParameters(argschema.schemas.DefaultSchema):
    mip_lvl = argschema.fields.Int(required=False, default=0)
    z_start = argschema.fields.Int(required=False, default=0)
    z_length = argschema.fields.Int(required=False, default=0)
    y_start = argschema.fields.Int(required=False, default=0)
    y_length = argschema.fields.Int(required=False, default=0)
    x_start = argschema.fields.Int(required=False, default=0)
    x_length = argschema.fields.Int(required=False, default=0)
    downsample_factor = argschema.fields.Int(required=False, default=1)


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
            miplvl = self.args["mip_lvl"],
            zstart = self.args["z_start"],
            zlength = self.args["z_length"],
            ystart = self.args["y_start"],
            ylength = self.args["y_length"],
            xstart = self.args["x_start"],
            xlength = self.args["x_length"],
            dsfactor = self.args["downsample_factor"]
            )


if __name__ == "__main__":
    mod = CreateCutoutParser()
    mod.run()
