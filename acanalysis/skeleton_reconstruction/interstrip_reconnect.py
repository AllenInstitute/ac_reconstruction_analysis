import zarr
import numpy as np
import matplotlib.pyplot as plt
import navis
import os
import requests
import glob
import numpy as np
from pathlib import Path
from io import BytesIO
from scipy.spatial import distance

import concurrent.futures
from joblib import dump,load, Parallel, delayed, parallel_config
from natsort import natsorted
import uuid
import random
import pandas as pd
from acanalysis.skeleton_reconstruction.reconnect_skeletons import *
from acanalysis.skeleton_reconstruction.util import read_navis_neurons_tar, write_navis_skels_tar, remove_cutout_nodes, translate_nodes, create_rectangle_volume, remove_overlap_nodes, filter_skeletons



def find_overlap_volumes(files, translations, im_shape):
    vertices,vols = [],[]
    translate_df = {files[i]: translations[i] for i in range(len(files))}
    for file,trans in translate_df.items():
        #translate dimensions
        sx,sy,sz = np.array(im_shape)
        tx,ty,tz = trans
        post = np.array([0,sx,0,sy,0,sz]) + np.array([tx,tx,ty,ty,tz,tz])
        #create volume
        vol = create_rectangle_volume(post, file)
        vols.append(vol), vertices.append(vol.vertices.tolist())
    
    matches = []
    for file,vert in zip(files,vertices):
        #test keypoints against all volumes
        result = navis.in_volume(vert, vols)
        #save matches
        for ind,(key,value) in enumerate(result.items()):
            if any(value) == True:
                if file == key:
                    continue
                #gauge overlap
                overlap = abs(vert[0]-vols[ind].vertices[0])
                overlap[overlap!=0] = 1
                #sort and combine with overlap
                match = [file,key]
                match.sort()
                match += [tuple(overlap.astype('int'))]
                if match not in matches:
                    matches.append(match)
    volumes = pd.DataFrame(zip(files,vols), columns=['File','Vol'])
    matches = pd.DataFrame(matches, columns=['File1','File2','xyz_overlap'])

    #arrange match order according to priximity to origin
    for ind,row in matches.iterrows():
        p1, p2 = translate_df[row['File1']], translate_df[row['File2']]
        dis = [distance.euclidean((0,0,0), tuple(p1)), distance.euclidean((0,0,0), tuple(p2))]
        dis = dis.index(min(dis))
        if dis == 1:
            rep = {'File1':row['File2'],'File2':row['File1'], 'xyz_overlap':row['xyz_overlap']}
            matches.loc[ind,rep.keys()] = list(rep.values())
    
    return matches,volumes
    

###Order the matches so there are no duplicate files in a job batch    
def order_matches(matches, n_jobs):
    store = matches.copy()
    ordered = []
    attempts = 0
    while ((attempts<n_jobs*1000) and (len(store)>=1)):
        try:
            sample = store.sample(n_jobs)
        except:
            sample = store.sample(len(store))
        fns = list(sample['File1'])+list(sample['File2'])
        if (len(fns) != len(set(fns))) == False:
            for ind,row in sample.iterrows():
                ordered.append(list(row))
            store = store.drop(list(sample.index))
            store.reset_index()
        attempts+=1
    ordered += store.values.tolist()
    matches = pd.DataFrame(ordered, columns=['File1','File2','xyz_overlap'])
    return matches



def find_overlap_bounds(matches,volumes):
    file_overlap = {}
    #set empty values
    for ind,row in matches.iterrows():
        file_overlap[row['File1']] = []
        file_overlap[row['File2']] = []
        
    for ind,row in matches.iterrows():
        over_dim = [(0,0),(0,0),(0,0)]
        v1 = volumes[volumes['File']==row['File1']]['Vol'].item().bbox
        v2 = volumes[volumes['File']==row['File2']]['Vol'].item().bbox
        v1d = np.around(v1.T,1)
        for ind,dim in enumerate(row['xyz_overlap']):
            if dim != 0:
                d1 = np.around(v1.T[ind],1)
                d2 = np.around(v2.T[ind],1)
                r1 = np.around(np.arange(d1[0],d1[1], .1),1)
                r2 = np.around(np.arange(d2[0],d2[1], .1),1)
                inter = list(set(r1).intersection(r2))
                inter.sort()
                inter = inter[0],inter[-1]
                over_dim[ind]=inter
    
        #skip if corner intersection
        res = sum(1 for i in over_dim if i == (0,0))
        if res == 1:
            continue
        #if no overlap, replace with standard boundary
        for ind,dim in enumerate(over_dim):
            if dim == (0,0):
                over_dim[ind] = tuple(v1d[ind])
        file_overlap[row['File1']] = file_overlap[row['File1']] + [[i for sub in over_dim for i in sub]]

    return file_overlap
    
    
def postprocess_strip(out_dir, file, cl, sc, bound_boxs=None, trans=[0,0,0], min_nodes=10, query_dis=10, min_collin=.7, downsample=2, smooth=2):
    #translate strip
    skels = read_navis_neurons_tar(file)
    fname = file.split("/")[-1]

    if trans != [0,0,0]:
        skels = translate_nodes(skels, trans=trans)
    
    #remove overlapping nodes
    if bound_boxs != None:
        for bb in bound_boxs:
            skels = remove_cutout_nodes(skels, bound_box=bb)
    
    write_navis_skels_tar(out_dir+fname, skels)
    
    
def connect_two_strips(strip1, strip2, overlap, cl=None, sc=None, edge_prop=.9, min_nodes=0, query_dis=10, min_collin=None, prob_thresh=0.1, dis_end=0, replace_file=True):
    if isinstance(strip1, navis.core.neuronlist.NeuronList):
        s1,s2 = strip1,strip2
        strip1,strip2 = 'strip1','strip2'
        pass
    else:
      if strip1.endswith('.gz'):
        s1 = read_navis_neurons_tar(strip1)
        s2 = read_navis_neurons_tar(strip2)
      elif strip1.endswith('.swc'):
        s1 = navis.read_swc(strip1)
        s2 = navis.read_swc(strip2)

    #combine neuronlist pairs and set strip names      
    s1.set_neuron_attributes([strip1]*int(len(s1)), 'strip')
    s2.set_neuron_attributes([strip2]*int(len(s2)), 'strip')
    skels = navis.NeuronList([s1,s2])

    #find bounding box
    edge_dis = [edge_prop if i != 0  else i for i in overlap]
    x,y,z = skels.bbox
    x_dis,y_dis,z_dis = ((np.diff(x)*edge_dis[0])/2)[0], ((np.diff(y)*edge_dis[1])/2)[0], ((np.diff(z)*edge_dis[2])/2)[0]
    bound_box = np.concatenate((x + np.array([x_dis,-x_dis]), y + np.array([y_dis,-y_dis]), z + np.array([z_dis,-z_dis]))).tolist()

    try:
        #run reconnection
        non_merged, merged, id_remap = reconnect(skels=skels, cl=cl, sc=sc, min_nodes=min_nodes, query_dis=query_dis, min_collin=min_collin, 
                                       prob_thresh=prob_thresh, downsample=None, smooth=None, split=False, bound_box=bound_box, dis_end=dis_end)
        merged.set_neuron_attributes(['merged']*int(len(merged)), 'strip')
    
        if replace_file==True:
            #label nodes by strip
            for sk in merged:
                sk.nodes['label']='merge'
            #replace strip files
            for ind,strip in enumerate(list(set(non_merged.strip))):
                strip_sk = [i for i, value in enumerate(non_merged.strip) if value==strip]
                subset = non_merged[strip_sk]
                subset = navis.NeuronList([subset,merged])
                os.remove(strip)
                write_navis_skels_tar(strip, subset)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    
    
###Pull out merged skeletons
def get_merged_skeletons(file):
    print(file)
    if isinstance(file, navis.core.neuronlist.NeuronList):
        skels = file
        pass
    else:
      if file.endswith('.gz'):
        skels = read_navis_neurons_tar(file)
      elif file.endswith('.swc'):
        skels = navis.read_swc(file)
          
    non_merged, merged = navis.NeuronList(None), navis.NeuronList(None)
    for sk in skels:
        if sk.nodes.iloc[0]['label']=="'merge'":
            merged.append(sk)
        else:
            non_merged.append(sk)

    os.remove(file)
    write_navis_skels_tar(file, non_merged)
    return merged
