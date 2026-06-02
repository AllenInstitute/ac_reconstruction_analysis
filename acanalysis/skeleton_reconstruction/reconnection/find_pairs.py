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

from acanalysis.skeleton_reconstruction.reconnection.utils import write_cv_skels_tar, read_cv_neurons_tar
from acanalysis.skeleton_reconstruction.reconnection.h5_skeletons import *
from acanalysis.skeleton_reconstruction.reconnection.h5_reconnect import *


import json
import numpy as np



def make_json_safe(obj):
    """
    Recursively convert anything to a JSON-safe format:
    - NumPy arrays -> lists
    - NumPy numbers -> Python scalars
    - tuples -> lists
    - lists/dicts -> recurse
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, tuple) or isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    return obj


def dump_json(filename, data):
    with open(filename, "w") as f:
        json.dump(make_json_safe(data), f, separators=(",", ":"))



def load_json(filename):
    data = json.load(open(filename))
    return [
        (tuple(map(tuple, item[0])), np.array(item[1], dtype=np.float32))
        for item in data
    ]



class CloudOptions(argschema.schemas.DefaultSchema):
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-east-1')
    bucket = argschema.fields.String(required=False, default=None, allow_none=True)
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
     

class FindPairsParameters(argschema.ArgSchema):
    skels = argschema.fields.String(required=True, metadata = {'description': 'Input skeletons, as navis objects, swc, or swc.gz'})
    out_file = argschema.fields.String(required=False, dump_default=None, metadata = {'description': 'Output file for FindPairsed skeletons'})
    cl = argschema.fields.String(required=False, dump_default=None, allow_none=True,metadata = {'description': 'Model File'})
    sc = argschema.fields.String(required=False, dump_default=None, allow_none=True, metadata = {'description': 'Scalar File'})    
    min_nodes = argschema.fields.Int(required=False, dump_default=10, description='Minimum skeleton node length')
    query_dis = argschema.fields.Int(required=False, dump_default=20, description='Maximum query distance for matching end nodes')
    min_collin = argschema.fields.Float(required=False, dump_default=.8, description='Minimum collinearity for finding skeleton merge pairs')
    cutout = argschema.fields.String(required=False, allow_none=True, dump_default=None)
    method = argschema.fields.String(required=False, dump_default='dist', metadata={'description': 'Pair-finding method: "model" or "dist"'})
    n_workers = argschema.fields.Int(required=False, dump_default=10)
    
class FindPairsModule(argschema.ArgSchemaParser):
    default_schema = FindPairsParameters

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
        
        out_dir = os.path.dirname(self.args["out_file"])
        os.makedirs(out_dir, exist_ok=True)
        
        method = self.args['method']
        if method not in ('model', 'dist'):
            raise ValueError(f"method must be 'model' or 'dist', got '{method}'")
            
            
        if 'swc' in self.args['skels']:
            skels = read_cv_neurons_tar(self.args['skels'], n_workers = self.args['n_workers'])       
        
        else:
            # --- Convert cutout from string to list if present ---
            if self.args['cutout'] is not None:
                self.args['cutout'] = [int(x.strip("'")) for x in self.args["cutout"].split(',')]
                x1,x2,y1,y2,z1,z2 = self.args['cutout']
                skels, shards = query_skeletons_by_bb((x1, y1, z1, x2, y2, z2), self.args['skels'], n_workers=self.args['n_workers']) 
            else:
                skels, shards = load_all_skeletons(self.args['skels'], n_workers=self.args['n_workers'])   

        print("# of Skels: ", len(skels))             
        
        if len(skels)>0:
            if method == 'model':
                if not self.args['sc'] or not self.args['cl']:
                    raise ValueError("'sc' and 'cl' are required when method is 'model'")
                sc = load(self.args['sc'])
                cl = load(self.args['cl'])
                pairs = find_pairs(skels, query_dis=self.args['query_dis'], min_collin=self.args['min_collin'], sc=sc, cl=cl, batch_size=1000, min_nodes=self.args['min_nodes'])     
                print('Total Model Pairs: ', len(pairs))
                np.save(os.path.join(out_dir, "{0}_model_pairs.npy".format(str(self.args['cutout']))), pairs)
    
            elif method == 'dist':
                pairs = find_pairs(skels, query_dis=self.args['query_dis'], min_collin=self.args['min_collin'], min_nodes=self.args['min_nodes'])        
                print('Total Distance Pairs: ', len(pairs))
                np.save(os.path.join(out_dir, "{0}_dist_pairs.npy".format(str(self.args['cutout']))), pairs)
       

if __name__ == "__main__":
    mod = FindPairsModule()
    mod.run()


__all__ = [
    "FindPairsModule",
    "FindPairsParameters"
]


 
    
            


 
    
            