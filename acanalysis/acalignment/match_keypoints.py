# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:17:11 2023

@author: kevint
"""

import numpy
from acanalysis.acalignment.keypoints import write_keypoints_to_file,read_keypoints
from scipy.spatial import cKDTree
from skimage.transform import matrix_transform


def get_features_from_keypoints(kplist,axes=None,transforms=None):
    #2D rigid transformations input as list (fifo) of ndarrays with scikit for now
    #TODO: need to implement transforms in mpyicbg
    locs = numpy.array([k.location for k in kplist])
    vecs = numpy.array([k.vector for k in kplist])
    if not axes is None:
        locs = locs[:,axes]
    if not transforms is None:
        rigidM = transforms[0]
        if len(transforms) > 1:
            for M in transforms[1:]:
                rigidM = M @ rigidM
        rotM = numpy.eye(3)
        rotM[:2,:2] = rigidM[:2,:2]
        locs = matrix_transform(locs,matrix=rigidM)
        vecs[:,axes] = matrix_transform(vecs[:,axes],matrix=rotM)
    return locs,vecs


def match_keypoint_sets(kpset0,kpset1,axes=[1,2],tforms0=None,tforms1=None,knn=10,kdtreeleafsize=50,mincosine=0.8):
    pyz,pvecs = get_features_from_keypoints(kpset0,axes=axes,transforms=tforms0)
    qyz,qvecs = get_features_from_keypoints(kpset1,axes=axes,transforms=tforms1)
    
    qkdtree = cKDTree(qyz,leafsize=kdtreeleafsize)
    
    matchset0 = []
    matchset1 = []
    distancelist = []
    for ip in range(pyz.shape[0]):
        ploc = pyz[ip]
        qds,qinds = qkdtree.query(ploc,k=knn)
        cosines = numpy.array([numpy.dot(pvecs[ip],-qvecs[i]) for i in qinds])
        if any(cosines>=mincosine):
            iq = numpy.argmax(cosines)
            matchset0.append(kpset0[ip])
            matchset1.append(kpset1[qinds[iq]])
            distancelist.append(qds[iq])
            
    return matchset0,matchset1,distancelist


def combine_tile_keypoints(kpfileList,offsetList):
    kpList = []
    for i,kpfile in enumerate(kpfileList):
        kpList += read_keypoints(kpfile,locfunc=lambda x: x + numpy.array(offsetList[i]))
    return kpList


def run_match(kpfiles0,kpfiles1,tforms0,tforms1,offsets0,offsets1,output0,output1,affines0,affines1):
    kpset0 = combine_tile_keypoints(kpfiles0,offsets0)
    kpset1 = combine_tile_keypoints(kpfiles1,offsets1)
    matches0,matches1,distances = match_keypoint_sets(kpset0,kpset1,tforms0=tforms0,tforms1=tforms1)
    write_keypoints_to_file(matches0,output0)
    write_keypoints_to_file(matches1,output1)