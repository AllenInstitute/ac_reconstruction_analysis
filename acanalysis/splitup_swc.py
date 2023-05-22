# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:44:58 2023

@author: kevint
"""
import pathlib
import argschema
import navis
import json


# example cmd input
# python splitup_swc.py
# --input_swc /ACdata/Users/kevin/skeletons/olga_swc/H17_x55_S32_S33/S33_highres_Pos41.swc
# --output_dir /ACdata/Users/kevin/skeletons/precomputed/H17_x55/S33_highres_Pos41
# --output_format precomputed



def get_tree_from_swc(swcfile):
    tree = navis.read_swc(swcfile)
    return tree


def write_subtrees(treeNeuron, savedir='.', output="swc", minlength=1,
                   skel_voxel_nm=(1, 1, 1)):
    savepath = pathlib.Path(savedir)
    if not savepath.exists():
        savepath.mkdir(parents=True)
    subtrees = treeNeuron.subtrees
    treeNodesById = treeNeuron.nodes.set_index("node_id")
    axonList = []
    for i, tree in enumerate(subtrees):
        if tree.shape[0] >= minlength:
            aid = str(tree[0])
            subtree = treeNodesById.loc[tree]
            axon = navis.TreeNeuron(subtree.reset_index())
            axon.id = aid
            if output == "swc":
                swcname = aid + ".swc"
                swcpath = savepath / swcname
                axon.to_swc(swcpath)
            elif output == "precomputed":
                axonList.append(axon)
    if output == "precomputed" and axonList:
        navis.write_precomputed(navis.NeuronList(axonList), savepath)
        write_info(savepath, skel_voxel_nm)
        segpropspath = savepath / "segment_properties"
        if not segpropspath.exists():
            segpropspath.mkdir()
        write_segprops(segpropspath, axonList)


def write_info(savepath, voxel_nm):
    infofile = savepath / "info"
    with open(infofile) as f:
        info = json.load(f)
    info["transform"] = [voxel_nm[0], 0, 0, 0, 0,
                         voxel_nm[1], 0, 0, 0, 0, voxel_nm[2], 0]
    info["segment_properties"] = "segment_properties"
    with open(infofile, "w+") as f:
        json.dump(info, f)


def write_segprops(segpropspath, axonList):
    ids = [str(a.id) for a in axonList]
    properties = []
    properties.append({"id": "tags", "type": "tags", "tags": [
                      "all"], "values": [[0] for a in axonList]})
    info = {"@type": "neuroglancer_segment_properties",
            "inline": {"ids": ids, "properties": properties}}
    with open(segpropspath / "info", "w+") as f:
        json.dump(info, f)


def split_swc(swcfile, savedir, output="swc", minlength=1, **kwargs):
    treeNeuron = get_tree_from_swc(swcfile)
    write_subtrees(treeNeuron, savedir=savedir,
                   output=output, minlength=minlength,
                   **kwargs)


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
