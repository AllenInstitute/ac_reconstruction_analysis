import os
import numpy as np
import pandas as pd
import argschema as ags
import navis
import acanalysis.skeleton_reconstruction.util as util
from joblib import dump, load


class ReconnectParameters(ags.ArgSchema):
    skels = ags.fields.InputFile(required=True, metadata = {'description': 'Input skeletons, as navis objects, swc, or swc.gz'})
    out_file = ags.fields.OutputFile(required=False, dump_default=None, metadata = {'description': 'Output file for reconnected skeletons'})
    cl = ags.fields.InputFile(required=False, dump_default=None, allow_none=True,metadata = {'description': 'Model File'})
    sc = ags.fields.InputFile(required=False, dump_default=None, allow_none=True, metadata = {'description': 'Scalar File'})    
    min_nodes = ags.fields.Int(required=False, dump_default=10, description='Minimum skeleton node length')
    prob_thresh = ags.fields.Float(required=False, dump_default=0.5, description='Minimum probability allowed for merge model prediction')
    resample = ags.fields.Int(required=False, dump_default=2, description='Factor for upsampling skeletons')
    split = ags.fields.Bool(required=False, dump_default=True, description='Whether to search for and split branches')
    query_dis = ags.fields.Int(required=False, dump_default=10, description='Maximum query distance for matching end nodes')
    min_collin = ags.fields.Float(required=False, dump_default=.1, description='Minimum collinearity for finding skeleton merge pairs')
    bound_box = ags.fields.Tuple((ags.fields.Int(),ags.fields.Int(),ags.fields.Int()),required=False, dump_default=None, allow_none=True)
    dis_end = ags.fields.Int(required=False, dump_default=0, description='How many nodes from the tips of skeletos to use for pair finding')
            

def reconnect(skels, out_file = None, cl=None, sc=None, min_nodes=10, downsample=4, split=True, query_dis=10, min_collin=.8, prob_thresh=0.1, bound_box=None, dis_end=0):     
    # load skeletons
    if isinstance(skels, navis.core.neuronlist.NeuronList):
      pass
    else:
      if skels.endswith('.gz'):
        skels = util.read_navis_neurons_tar(skels)
      elif skels.endswith('.swc'):
        skels = navis.read_swc(skels)
        
    #load  model and scalar files
    if sc and cl:
      sc = load(sc)
      cl = load(cl)
    else:
      prob_thresh = None
      pass
    
    if split==True:
      # Split branches 
      skels = util.swc_split_branches(skels, min_nodes=min_nodes)
    
    # Upsample skeletons
    if downsample:
      skels = navis.downsample_neuron(skels, downsampling_factor=downsample, parallel=True, progress=False)

    # Find pairs
    pair_data_iter = util.find_pairs(neuro_list=skels, sc=sc, cl=cl, query_dis=query_dis, min_collin=min_collin, bound_box=bound_box, dis_end=dis_end)

    try:
      # Merge segment pairs with prob above thresh
      unmerged, merged, id_remap = util.merge_pairs(skels, pair_data_iter, prob_thresh)
    except:
      unmerged, merged, id_remap = skels, navis.NeuronList(None), None              
      
    if out_file:
      util.write_navis_skels_tar(out_file, navis.NeuronList([unmerged,merged]), mode='w:gz')
    else:
      return unmerged, merged, id_remap
            

    
            
class Reconnect(ags.ArgSchemaParser):
    def run(self):
        reconnect(self.args['skels'], self.args['out_file'], self.args['cl'],self.args['sc'],
        self.args['min_nodes'], self.args['prob_thresh'], self.args['downsample'], 
        self.args['split'], self.args['query_dis'], self.args['min_collin'],
        self.args['bound_box'], self.args['dis_end'])
   
        
if __name__ == "__main__":
    mod = Reconnect(schema_type=ReconnectParameters)
    mod.run()       
