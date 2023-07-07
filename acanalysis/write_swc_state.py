""" Write layer json for visualizing precomputed skeletons

"""

import pathlib
import argschema
import json
from utils import get_zarr_params


example_input = {
    "zarr_file": "/ACdata/Users/kevin/ispim_ome_zarr/H17_x55_S32_230412_highres/H17_x55_S32_230412_highres.zarr",
    "tile_group": "highres_Pos78",
    "precomputed_dir": "precomputed://http://bigkahuna.corp.alleninstitute.org/ACdata/Users/kevin/skeletons/precomputed/H17_x55/S32_highres_Pos78",
    "output_file": "/ACdata/Users/kevin/skeletons/S32_Pos78.json",
    "cutout_zyx": [24000,0,0],
    "scale_zyx_um": [0.704,0.812,0.812],
    "input_json": "/ACdata/Users/kevin/tiffs/S32_highres_Pos78.json"
    }


def get_scale_dims(zparams):
    """get zarr voxel dimensions for neuroglancer output scale

    Parameters
    ----------
    zparams : dict
        ome-zarr attributes

    Returns
    ----------
    scaleDims : dict
        neuroglancer output dimensions
    """
    sc = zparams.scale
    scaleDims = {
        "z": [sc[0], "um"],
        "y": [sc[1], "um"],
        "x": [sc[2], "um"]
    }
    return scaleDims


def get_tr_matrix(zparams, offset):
    """get coordinate translation matrix for segmentation layer
    for registration with source zarr data

    Parameters
    ----------
    zparams : dict
        ome-zarr attributes
    offset : list of ints
        offset translations according to location of cutout

    Returns
    ----------
    trM : ndarray
        translation matrix
    """
    trList = zparams.translation
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


def create_seg_layer(precompDir, zparams, cutout_zyx=None, scale_zyx=None):
    """create segmentation layer json from input parameters

    Parameters
    ----------
    precompDir : str
        path string to directory of precomputed skeletons
    zparams : dict
        ome-zarr attributes
    cutout_zyx : list of ints
        voxel coordinate origin of segmentation cutout
    scale_zyx : list of floats
        voxel dimensions in um

    Returns
    ----------
    slayer : dict
        neuroglancer segmentation layer json
    """
    outputDims = get_scale_dims(zparams)
    if cutout_zyx is None:
        cutout_zyx = [0,0,0]
    if scale_zyx is None:
        inputDims = outputDims
    else:
        inputDims = {
            "z": [scale_zyx[0], "um"],
            "y": [scale_zyx[1], "um"],
            "x": [scale_zyx[2], "um"]
        }
        
    trMatrix = get_tr_matrix(zparams, offset=cutout_zyx)
    slayer = {
        "type": "segmentation",
        "source": {
            "url": precompDir,
            "transform": {
                "matrix": trMatrix,
                "outputDimensions": outputDims,
                "inputDimensions": inputDims
            }
        },
        "tab": "source",
        "segments": [],
        "name": "skeletons"
    }
    return slayer


def write_swc_state(zarrFile, tileGroup, precompDir, outputFile, **kwargs):
    """write segmentation layer .json file from input parameters
    
    Parameters
    ----------
    zarrFile : str
        path string to image data source zarr
    tileGroup : str
        name of zarr group containing position source data
    precompDir : str
        path string to directory of precomputed skeletons
    outputFile : str
        path string to output .json file
    """
    zpath = pathlib.Path(zarrFile)
    if zpath.exists():
        zparams = get_zarr_params(zpath,tileGroup)
        slayer = create_seg_layer(precompDir, zparams, **kwargs)
        with open(outputFile, "w+") as f:
            json.dump(slayer, f, indent=4)        


class SwcCoordinateParameters(argschema.schemas.DefaultSchema):
    cutout_zyx = argschema.fields.List(
        argschema.fields.Int(), cli_as_single_argument=True, required=False, default=None)
    scale_zyx_um = argschema.fields.List(
        argschema.fields.Float(), cli_as_single_argument=True, required=False, default=None)


class WriteSwcNgStateInputParameters(argschema.ArgSchema, SwcCoordinateParameters):
    zarr_file = argschema.fields.Str(required=True)
    tile_group = argschema.fields.Str(required=True)
    precomputed_dir = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)


class WriteSwcNgStateParser(argschema.ArgSchemaParser):
    default_schema = WriteSwcNgStateInputParameters

    def run(self):
        write_swc_state(
            self.args["zarr_file"],
            self.args["tile_group"],
            self.args["precomputed_dir"],
            self.args["output_file"],
            cutout_zyx=self.args["cutout_zyx"],
            scale_zyx=self.args["scale_zyx_um"]
            )


if __name__ == "__main__":
    mod = WriteSwcNgStateParser()
    mod.run()
