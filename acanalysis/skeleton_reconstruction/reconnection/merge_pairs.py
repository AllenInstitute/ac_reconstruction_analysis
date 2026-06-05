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


from acanalysis.skeleton_reconstruction.reconnection.utils import write_cv_skels_tar, read_cv_neurons_tar
from acanalysis.skeleton_reconstruction.reconnection.h5_skeletons import *
from acanalysis.skeleton_reconstruction.reconnection.h5_reconnect import *




class CloudOptions(argschema.schemas.DefaultSchema):
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-east-1')
    bucket = argschema.fields.String(required=False, default=None, allow_none=True)
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
     

class MergePairsParameters(argschema.ArgSchema):
    skels = argschema.fields.String(required=True)
    pair_file = argschema.fields.String(required=False, dump_default=None)
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

        pair_file_arg = self.args["pair_file"]
        if os.path.isdir(pair_file_arg):
            pattern = os.path.join(pair_file_arg, f"{method}_consolidated*.npy")
            pair_files = sorted(glob.glob(pattern))
            if not pair_files:
                raise FileNotFoundError(f"No {method}_consolidated*.npy files found in {pair_file_arg}")
            print(f"Found {len(pair_files)} pair file(s)")
        else:
            pair_files = [pair_file_arg]
        
        all_pairs = []
        all_components = []
        for pf in pair_files:
            combined = np.load(pf, allow_pickle=True).item()
            all_pairs.append(combined["pairs"])
            all_components.extend([list(x.tolist()) for x in combined["components"]])
        
        data = np.concatenate(all_pairs) if len(all_pairs) > 1 else all_pairs[0]
        components = all_components

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
            out_path = os.path.join(str(current_path.parent), "merged_skels.swcs.tar.gz")
            write_cv_skels_tar(out_path, skels, mode='w:gz')

        #write h5 
        else:
            delete_skeletons_parallel(self.args["skels"], skel_ids, n_workers=self.args['n_workers'])
            out_skels_dic = {i.id: i for i in skels}
            global_index = shard_and_write_skeletons(
                out_skels_dic,
                self.args["skels"],
                max_skeletons_per_shard=10000,
                n_workers=self.args['n_workers'])
                  
        
             
        

if __name__ == "__main__":
    mod = MergePairsModule()
    mod.run()


__all__ = [
    "MergePairsModule",
    "MergePairsParameters"
]


 
    
            