# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:44:58 2023

@author: kevint
"""
import pathlib
import numpy as np
import argschema
import navis
import json


def get_tree_from_swc(swcfile):
    tree = navis.read_swc(swcfile)
    return tree


def write_subtrees(treeNeuron, savedir='.', output="swc", minlength=1):
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
        segpropspath = savepath / "segment_properties"
        if not segpropspath.exists():
            segpropspath.mkdir()
        write_segprops(segpropspath, axonList)


def write_segprops(segpropspath, axonList):
    ids = [str(a.id) for a in axonList]
    properties = []
    properties.append({"id": "tags", "type": "tags", "tags": [
                      "all"], "values": [[0] for a in axonList]})
    info = {"@type": "neuroglancer_segment_properties",
            "inline": {"ids": ids, "properties": properties}}
    with open(segpropspath / "info.json", "w+") as f:
        json.dump(info, f, indent=4)


def split_swc(swcfile, savedir, output="swc", minlength=1):
    treeNeuron = get_tree_from_swc(swcfile)
    write_subtrees(treeNeuron, savedir=savedir,
                   output=output, minlength=minlength)


class SplitSWCInputParameters(argschema.ArgSchema, argschema.schemas.DefaultSchema):
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
            minlength=self.args["minimum_length"])


if __name__ == "__main__":
    mod = SplitSWCParser()
    mod.run()
