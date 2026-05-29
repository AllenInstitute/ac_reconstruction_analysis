import argschema
import pathlib
import navis
import numpy as np
import pandas as pd
from joblib import dump, load
import os
from operator import add
from collections import deque, defaultdict, OrderedDict, Counter
from joblib import dump, load, Parallel, delayed
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean
import scipy
import networkx as nx
import itertools
from pathlib import Path
import concurrent
from concurrent.futures import ThreadPoolExecutor
import tarfile
import uuid
import io
from io import BytesIO
import copy
import colorsys
import math
import glob
import networkx as nx

from acanalysis.skeleton_reconstruction.reconnection.h5_skeletons import *
from acanalysis.skeleton_reconstruction.reconnection.h5_reconnect import *

    
    
    
def longest_path_in_tree(G_sub):
    # pick arbitrary starting node
    start = list(G_sub.nodes)[0]
    
    # first BFS
    lengths = nx.single_source_shortest_path_length(G_sub, start)
    farthest_node = max(lengths, key=lengths.get)
    
    # second BFS from farthest_node
    lengths = nx.single_source_shortest_path_length(G_sub, farthest_node)
    farthest_node2 = max(lengths, key=lengths.get)
    
    # return the longest path
    return nx.shortest_path(G_sub, source=farthest_node, target=farthest_node2)
    
    
def create_components(raw_data, method='dist', components_per_file=5000, output_dir=None):
    """
    Build connected components from raw pair data, prune large/branching components,
    and optionally save chunked output files.

    Parameters
    ----------
    raw_data : list
        List of pair records loaded from .npy pair files.
    method : str
        'dist' or 'model' — controls deduplication threshold.
    components_per_file : int or None
        If set, split output into chunks of this size. If None, returns a single consolidated result.
    output_dir : str or Path or None
        If provided, saves .npy files here. If None, skips file I/O and just returns the data.

    Returns
    -------
    out_components : list of np.ndarray
        Each array contains the node IDs of one connected component.
    data : np.ndarray
        Structured array of filtered pairs with fields (id1, count1, id2, count2, score).
    """
    threshold = None if method == 'dist' else 0.3
    data = deduplicate(raw_data, threshold=threshold)

    print('Processing:', method)
    G = nx.Graph()
    edges = data[['id1', 'id2']].to_numpy()
    G.add_edges_from(edges)
    print("# Skeletons:", len(list(G.nodes())))

    components = list(nx.connected_components(G))
    for component in components:
        subgraph = G.subgraph(component)
        if len(component) > 30000:
            print('Removing large component')
            G.remove_nodes_from([n for n in subgraph.nodes])
            continue

        branch_nodes = [n for n, d in subgraph.degree() if d > 2]
        if branch_nodes:
            path_nodes = longest_path_in_tree(subgraph)
            G.remove_nodes_from([n for n in subgraph.nodes if n not in path_nodes])

    out_components = np.array([np.array(x) for x in nx.connected_components(G)], dtype=object)
    out_components = [arr for arr in out_components if arr.size > 0]
    print("# Components:", len(out_components))

    kept_nodes = set(G.nodes())
    data = data[
        data['id1'].apply(lambda x: x in kept_nodes) &
        data['id2'].apply(lambda x: x in kept_nodes)
    ].copy().values.tolist()

    dtype = np.dtype([
        ("id1", np.int64),
        ("count1", np.int32),
        ("id2", np.int64),
        ("count2", np.int32),
        ("score", np.float32),
    ])
    data = np.array([tuple(row) for row in data], dtype=dtype)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if components_per_file is None:
            np.save(output_dir / f"{method}_consolidated.npy", {"pairs": data, "components": np.array(out_components, dtype=object)})
        else:
            n_chunks = math.ceil(len(out_components) / components_per_file)
            print(f"Writing {n_chunks} file(s) ({components_per_file} components per file)")
            for chunk_idx in range(n_chunks):
                chunk_components = out_components[chunk_idx * components_per_file : (chunk_idx + 1) * components_per_file]
                chunk_nodes = set()
                for arr in chunk_components:
                    chunk_nodes.update(arr.tolist())

                chunk_data = data[
                    np.isin(data['id1'], list(chunk_nodes)) &
                    np.isin(data['id2'], list(chunk_nodes))
                ]
                chunk_components_arr = np.array(chunk_components, dtype=object)
                consolidated_path = output_dir / "{0}_consolidated_{1:04d}.npy".format(method, chunk_idx)
                np.save(consolidated_path, {"pairs": chunk_data, "components": chunk_components_arr})
                print(f"  Chunk {chunk_idx:04d}: {len(chunk_components)} components, {len(chunk_data)} pairs")

    return out_components, data




class CloudOptions(argschema.schemas.DefaultSchema):
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-east-1')
    bucket = argschema.fields.String(required=False, default=None, allow_none=True)
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
     

class CreateComponentsParameters(argschema.ArgSchema):
    pair_files = argschema.fields.String(required=False, dump_default=None, metadata={'description': 'Output file for CreateComponentsed skeletons'})
    method = argschema.fields.String(required=False, dump_default='dist', metadata={'description': 'Which pair type to process: "dist" or "model"'})
    components_per_file = argschema.fields.Int(required=False, dump_default=5000, allow_none=True, metadata={'description': 'If set, split output into multiple files with this many components each. If None, writes a single file (original behavior).'})

    
class CreateComponentsModule(argschema.ArgSchemaParser):
    default_schema = CreateComponentsParameters

    def output(self, d):
        out_json = self.args.get("output_json")
        if out_json:
            pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(f, d)
                
    @property
    def cloud_options(self):
        try:
            return self.args["cloud_options"]  
        except:
            return {}

    def run(self):
        method = self.args['method']
        if method not in ('dist', 'model'):
            raise ValueError(f"method must be 'dist' or 'model', got '{method}'")
    
        pair_files = list(Path(self.args["pair_files"]).glob("*{0}_pairs.npy".format(method)))
        print("# Pair Files:", len(pair_files))
    
        raw_data = []
        for fp in pair_files:
            if method in str(fp):
                data = np.load(fp, allow_pickle=True)
                for d in data:
                    raw_data.append(d)
    
        if not raw_data:
            raise ValueError(f"No {method} pair files found")
    
        create_components(
            raw_data=raw_data,
            method=method,
            components_per_file=self.args.get('components_per_file'),
            output_dir=self.args["pair_files"],
        )
            

if __name__ == "__main__":
    mod = CreateComponentsModule()
    mod.run()


__all__ = [
    "CreateComponentsModule",
    "CreateComponentsParameters"
]
       
 
    
            