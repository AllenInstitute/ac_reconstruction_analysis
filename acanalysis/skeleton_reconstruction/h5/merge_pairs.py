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


from acanalysis.skeleton_reconstruction.utils import write_cv_skels_tar, read_cv_neurons_tar
from h5_skeletons import *
from h5_reconnect import *




class CloudOptions(argschema.schemas.DefaultSchema):
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-east-1')
    bucket = argschema.fields.String(required=False, default=None, allow_none=True)
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
     

class MergePairsParameters(argschema.ArgSchema):
    skels = argschema.fields.String(required=True)
    pair_files = argschema.fields.String(required=False, dump_default=None)
    component_index = argschema.fields.String(required=True, dump_default=None)
    prob_thresh = argschema.fields.Float(required=True, dump_default=.1)
    method = argschema.fields.String(required=False, dump_default='dist', metadata={'description': 'Which pair type to merge: "dist" or "model"'})
    n_workers = argschema.fields.Int(required=False, dump_default=10)

    
class MergePairsModule(argschema.ArgSchemaParser):
    default_schema = MergePairsParameters

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

        data = np.load(os.path.join(self.args["pair_files"], f"{method}_consolidated.npy"), allow_pickle=True)
        components = np.load(os.path.join(self.args["pair_files"], f"{method}_components.npy"), allow_pickle=True)
        components = [list(x.tolist()) for x in components]

        if self.args["component_index"]:
            i1, i2 = [int(x) for x in self.args["component_index"].split() if x.isdigit()]
            components = components[i1:i2]

        skel_ids = {item for sublist in components for item in sublist}
        data = [item for item in data if int(item[0]) in skel_ids and int(item[2]) in skel_ids]
        
        
        
         #write swc
        if 'swc' in self.args['skels']:
            skels = read_cv_neurons_tar(self.args['skels'], n_workers = self.args['n_workers'])             
        #write h5 
        else:
            skels, shards = query_skeletons_by_id(skel_ids, self.args["skels"], n_workers=self.args['n_workers'])
        

        if method == 'model':
            merged, non_merged, merge_ids = merge_pairs(skels, data, prob_thresh=self.args['prob_thresh'])
        else:
            merged, non_merged, merge_ids = merge_pairs(skels, data, min_collin=.8)

        skels = merged + non_merged
                        
        
        #write swc
        if 'swc' in self.args['skels']:
            current_path = Path(self.args['skels'])
            out_path = os.path.join(str(current_path.parent), "out_skels.swcs.tar.gz")
            write_cv_skels_tar(out_path, skels, mode='w:gz')

        #write h5 
        else:
            out_path = os.path.join(self.args["skels"], f"skeleton_shards_{method}")  
            out_skels_dic = {i.id: i for i in skels}
            global_index = shard_and_write_skeletons(
                out_skels_dic,
                out_path,
                max_skeletons_per_shard=10000,
                n_workers=10
            )
        
        
             
        

if __name__ == "__main__":
    mod = MergePairsModule()
    mod.run()


__all__ = [
    "MergePairsModule",
    "MergePairsParameters"
]


 
    
 
    
            