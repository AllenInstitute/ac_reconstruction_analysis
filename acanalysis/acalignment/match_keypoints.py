# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:17:11 2023

@author: kevint
"""

import numpy
from keypoints import write_keypoints_to_file,read_keypoints
from scipy.spatial import cKDTree


def get_features_from_keypoints(kplist,axes=None,transforms=None):
    #TODO: need to implement transforms
    locs = numpy.array([k.location for k in kplist])
    vecs = numpy.array([k.vector for k in kplist])
    if not axes is None:
        locs = locs[:,axes]
    return locs,vecs


def match_keypoint_sets(kpset0,kpset1,axes=[1,2],affines0=None,affines1=None,knn=10,kdtreeleafsize=50,mincosine=0.8):
    pyz,pvecs = get_features_from_keypoints(kpset0,axes=axes,transforms=affines0)
    qyz,qvecs = get_features_from_keypoints(kpset1,axes=axes,transforms=affines1)
    
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
            matchset1.append(kpset1[iq])
            distancelist.append(qds[iq])
            
    return matchset0,matchset1,distancelist


def run_match(kpfile0,kpfile1,output0,output1,affines0,affines1):
    kpset0 = read_keypoints(kpfile0)
    kpset1 = read_keypoints(kpfile1)
    matches0,matches1,distances = match_keypoint_sets(kpset0,kpset1,affines0=affines0,affines1=affines1)
    write_keypoints_to_file(matches0,output0)
    write_keypoints_to_file(matches1,output1)