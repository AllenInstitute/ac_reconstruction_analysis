# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:17:11 2023

@author: kevint
"""

import numpy
from acanalysis.acalignment.keypoints import read_keypoints
from acanalysis.acalignment.transforms import get_matrix,ransac
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
        rigidM = get_matrix(transforms)
        rotM = numpy.eye(3)
        rotM[:2,:2] = rigidM[:2,:2]
        locs = matrix_transform(locs,matrix=rigidM)
        vecs[:,axes] = matrix_transform(vecs[:,axes],matrix=rotM)
    return locs,vecs


def match_keypoint_sets(kpset0,
                        kpset1,
                        axes=[1,2],
                        tforms0=None,
                        tforms1=None,
                        knn=None,
                        rball=None,
                        kdtreeleafsize=50,
                        mincosine=0.8):
    """generate matches between two sets of keypoints
    
    Parameters
    ----------
    kpset0 : list of keypoint.KeyPoint
        keypoints from first section
    kpset1 : list of keypoint.KeyPoint
        keypoints from second section
    tforms0 : list of numpy.ndarray
        list of transforms to apply to kpset0 before pairing
    tforms1 : list of numpy.ndarray
        list of transforms to apply to kpset1 before pairing
    knn : int
        k-nearest neighbors search
    rball : float
        ball search radius
    kdtreeleafsize : int
        leaf size parameter for setting up kdtree
    mincosine : float
        minimum cosine of alignment between pairs

    Returns
    ------
    matchset0 : list of keypoint.KeyPoint
        keypoints matched from kpset0 ordered by pairing
    matchset1 : list of keypoint.KeyPoint
        keypoints matched from kpset1 ordered by pairing
    distancelist : list of float
        euclidean distance between keypoint pairs
    """
    if knn is None and rball is None:
        knn = 10
    pyz,pvecs = get_features_from_keypoints(kpset0,axes=axes,transforms=tforms0)
    qyz,qvecs = get_features_from_keypoints(kpset1,axes=axes,transforms=tforms1)
    qkdtree = cKDTree(qyz,leafsize=kdtreeleafsize)
    
    matchset0 = []
    matchset1 = []
    distancelist = []
    for ip in range(pyz.shape[0]):
        ploc = pyz[ip]
        if not rball is None:
            qinds = numpy.array(qkdtree.query_ball_point(ploc,r=rball)).astype(int)
            qds = numpy.linalg.norm(ploc - qyz[qinds],axis=1)
        else:
            qds,qinds = qkdtree.query(ploc,k=knn)
        cosines = numpy.array([numpy.dot(pvecs[ip],-qvecs[i]) for i in qinds])
        if any(cosines>=mincosine):
            iq = numpy.argmax(cosines)
            matchset0.append(kpset0[ip])
            matchset1.append(kpset1[qinds[iq]])
            distancelist.append(qds[iq])
            
    return matchset0,matchset1,distancelist


def combine_tile_keypoints(kpfileList,offsetList,shuffle=False):
    """combine keypoints across tiles with spatial offsets into section coords
    
    Parameters
    ----------
    kpfileList : list of str or Path str
        filepaths to tile keypoint files
    offsetList : list of list or numpy.ndarray
        spatial offsets to be added to each keypoint location from the corresponding tile
    shuffle : bool
        flag indicating whether to randomly shuffle tile order/offsets

    Returns
    ------
    kpList : list of keypoint.KeyPoint
        list of combined keypoints
    """
    if shuffle:
        print("shuffling offsets")
        i_sh = numpy.random.permutation(len(offsetList))
    kpList = []
    for i,kpfile in enumerate(kpfileList):
        if shuffle:
            offset = offsetList[i_sh[i]]
        else:
            offset = offsetList[i]
        kpList += read_keypoints(kpfile,locfunc=lambda x: x + numpy.array(offset))
    return kpList


# run function not used in notebooks
# def run_match(kpfiles0,kpfiles1,tforms0,tforms1,offsets0,offsets1,output0,output1,affines0,affines1):
#     kpset0 = combine_tile_keypoints(kpfiles0,offsets0)
#     kpset1 = combine_tile_keypoints(kpfiles1,offsets1)
#     matches0,matches1,distances = match_keypoint_sets(kpset0,kpset1,tforms0=tforms0,tforms1=tforms1)
#     write_keypoints_to_file(matches0,output0)
#     write_keypoints_to_file(matches1,output1)