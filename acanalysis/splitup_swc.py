""" Split Olga's SWC file out to individual skeletons
Default output is folder of separate .swc files
with option to create precomputed directory
"""

import pathlib
import argschema
import navis
import json


example_input = {
    "input_swc": "/ACdata/Users/kevin/skeletons/olga_swc/H17_x55_S32_S33/S33_highres_Pos41.swc",
    "output_dir": "/ACdata/Users/kevin/skeletons/precomputed/H17_x55/S33_highres_Pos41",
    "output_format": "precomputed",
    "minimum_length": 1,
    "voxel_nm": 1
    }


def get_tree_from_swc(swcfile):
    """read navis.TreeNeuron from Olga's swc 

    Parameters
    ----------
    swcfile : str
        path string to input swc file

    Returns
    ----------
    tree : navis.TreeNeuron
        single tree graph containing all skeletons
    """
    tree = navis.read_swc(swcfile)
    return tree


def get_axon_list_from_zip(zipfile):
    """load list of skeletons from multi-swc .zip 

    Parameters
    ----------
    zipfile : str
        path string to input zip file

    Returns
    ----------
    axonList : list of navis.TreeNeuron
        list of navis skeletons in swc
    """
    swcs = navis.read_swc(zipfile,include_subdirs=True)
    axonList = [axon for axon in swcs]
    return axonList


def get_axon_list_from_subtrees(treeNeuron,minlength=1):
    """load list of skeletons split from subtrees of single TreeNeuron

    Parameters
    ----------
    treeNeuron : navis.TreeNeuron
        single tree with axon skeleton subtrees
    minlength : int
        minimum number of nodes for subtrees in list

    Returns
    ----------
    axonList : list of navis.TreeNeuron
        list of navis skeletons
    """
    subtrees = treeNeuron.subtrees
    treeNodesById = treeNeuron.nodes.set_index("node_id")
    axonList = []
    for i, tree in enumerate(subtrees):
        if tree.shape[0] >= minlength:
            aid = str(tree[0])
            subtree = treeNodesById.loc[tree]
            axon = navis.TreeNeuron(subtree.reset_index())
            axon.id = aid
            axonList.append(axon)
    return axonList


def write_subtrees(treeNeuron, savedir='.', output="swc", minlength=1,
                   skel_voxel_nm=(1, 1, 1)):
    """write split skeletons to output

    Parameters
    ----------
    treeNeuron : navis.TreeNeuron
        single tree with axon skeleton subtrees
    savedir : str
        path string to output directory
    output : str
        output format (swc or precomputed)
    minlength : int
        minimum number of nodes for subtrees in list
    skel_voxel_nm : tuple of ints
        skeleton coordinate dimensions in nm
    """
    savepath = pathlib.Path(savedir)
    if not savepath.exists():
        savepath.mkdir(parents=True)
    axonList = get_axon_list_from_subtrees(treeNeuron,minlength)
    if output == "swc":
        write_swcs(axonList,savepath)
    elif output == "precomputed" and axonList:
        write_precomputed(axonList,savepath,skel_voxel_nm)
        
def write_swcs(axonList,savepath):
    """write list of skeletons to directory of .swc files

    Parameters
    ----------
    axonList : list of navis.TreeNeuron
        list of navis skeletons
    savepath : str
        path string to output directory
    """
    for axon in axonList:
        aid = axon.id
        swcname = aid + ".swc"
        swcpath = savepath / swcname
        axon.to_swc(swcpath)
        
def write_precomputed(axonList,savepath,skel_voxel_nm=(1, 1, 1)):
    """write list of skeletons to directory of precomputed files

    Parameters
    ----------
    axonList : list of navis.TreeNeuron
        list of navis skeletons
    savepath : str
        path string to output directory
    skel_voxel_nm : tuple of ints
        skeleton coordinate dimensions in nm
    """
    if not savepath.exists():
        savepath.mkdir(parents=True)
    navis.write_precomputed(navis.NeuronList(axonList), savepath)
    write_info(savepath, skel_voxel_nm)
    segpropspath = savepath / "segment_properties"
    if not segpropspath.exists():
        segpropspath.mkdir()
    write_segprops(segpropspath, axonList)


def write_info(savepath, voxel_nm):
    """modify .info file required by precomputed format

    Parameters
    ----------
    savepath : str
        path string to precomputed directory
    voxel_nm : tuple of ints
        skeleton coordinate dimensions in nm
    """
    infofile = savepath / "info"
    with open(infofile) as f:
        info = json.load(f)
    info["transform"] = [voxel_nm[0], 0, 0, 0, 0,
                         voxel_nm[1], 0, 0, 0, 0, voxel_nm[2], 0]
    info["segment_properties"] = "segment_properties"
    with open(infofile, "w+") as f:
        json.dump(info, f)


def write_segprops(segpropspath, axonList):
    """write segment properties .info file for neuroglancer

    Parameters
    ----------
    segpropspath : str
        path string to precomputed segment properties directory
    axonList : list of navis.TreeNeuron
        list of navis skeletons
    """
    ids = [str(a.id) for a in axonList]
    properties = []
    properties.append({"id": "tags", "type": "tags", "tags": [
                      "all"], "values": [[0] for a in axonList]})
    info = {"@type": "neuroglancer_segment_properties",
            "inline": {"ids": ids, "properties": properties}}
    with open(segpropspath / "info", "w+") as f:
        json.dump(info, f)


def split_swc(swcfile, savedir, output="swc", minlength=1, skel_voxel_nm=(1,1,1)):
    """split skeletons from swc file and write to output

    Parameters
    ----------
    swcfile : str
        path string to input swc file
    savedir : str
        path string to output directory
    output : str
        output format (swc or precomputed)
    minlength : int
        minimum number of nodes for subtrees in list
    skel_voxel_nm : tuple of ints
        skeleton coordinate dimensions in nm
    """
    treeNeuron = get_tree_from_swc(swcfile)
    write_subtrees(treeNeuron, savedir=savedir,
                   output=output, minlength=minlength,
                   skel_voxel_nm=skel_voxel_nm)


class SkeletonProperties(argschema.schemas.DefaultSchema):
    voxel_nm = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(1, 1, 1))


class SplitSWCInputParameters(argschema.ArgSchema, SkeletonProperties):
    input_swc = argschema.fields.Str(required=True)
    output_dir = argschema.fields.Str(required=True)
    output_format = argschema.fields.Str(required=False, default="swc")
    minimum_length = argschema.fields.Int(required=False, default=1)


class SplitSWCParser(argschema.ArgSchemaParser):
    default_schema = SplitSWCInputParameters

    def run(self):
        split_swc(
            self.args["input_swc"],
            self.args["output_dir"],
            output=self.args["output_format"],
            minlength=self.args["minimum_length"],
            skel_voxel_nm=self.args["voxel_nm"])


if __name__ == "__main__":
    mod = SplitSWCParser()
    mod.run()
