# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:02:13 2023

@author: kevint
"""

import dataclasses
import numpy
import json
from skimage.transform import matrix_transform

@dataclasses.dataclass
class KeyPoint:
    """KeyPoint dataclass
    
    Attributes
    ----------
    name : str
        keypoint name associated with generating skeleton
    location : numpy.ndarray
        3D location (zyx) where z is vertical axis
    vector : numpy.ndarray
        3D unit vector (zyx) where [1,0,0] is flat surface normal
    """
    name: str
    location: numpy.ndarray
    vector: numpy.ndarray
    
    def tojson(self):
        return {
            "name":self.name,
            "location":self.location.tolist(),
            "vector":self.vector.tolist()
        }


class KeypointEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, KeyPoint):
            return obj.tojson()
        # Base class default() raises TypeError:
        return json.JSONEncoder.default(self, obj)


def write_keypoints_to_file(KeyPoints,jsonpath):
    with open(jsonpath,"w+") as f:
        json.dump(KeyPoints,f,cls=KeypointEncoder,indent=4)
        

def read_keypoints(jsonpath,locfunc=None):
    with open(jsonpath) as f:
        js = json.load(f)
    if locfunc is None:
        kpList = [KeyPoint(name=d["name"],
                           location=numpy.array(d["location"]),
                           vector=numpy.array(d["vector"])) for d in js]
    else:
        kpList = [KeyPoint(name=d["name"],
                           location=locfunc(numpy.array(d["location"])),
                           vector=numpy.array(d["vector"])) for d in js]
    return kpList


def filter_angles(keypts,maxangle):
    vecs = [kp.vector for kp in keypts]
    angles = numpy.arccos(numpy.array([numpy.abs(numpy.dot(numpy.array([1,0,0]),v)) for v in vecs]))*180/numpy.pi
    return [keypts[i] for i in numpy.nonzero(angles<maxangle)[0]]


def filter_sub_roi(keypts,coords,tform=None,axes=None):
    coord_axes = [a for a in range(len(coords)) if coords[a]]
    if tform is None:
        locs = numpy.array([k.location for k in keypts])
    else:
        locs,vecs = transform_keypoints(keypts,tform,axes)
    return [kp for i,kp in enumerate(keypts) if all([(locs[i,a]>=coords[a][0])and(locs[i,a]<coords[a][1]) for a in coord_axes])]


def transform_keypoints(keypts,rigidM,axes=None):
    locs = numpy.array([k.location for k in keypts])
    vecs = numpy.array([k.vector for k in keypts])
    rotM = numpy.eye(3)
    rotM[:2,:2] = rigidM[:2,:2]
    locs[:,axes] = matrix_transform(locs[:,axes],matrix=rigidM)
    vecs[:,axes] = matrix_transform(vecs[:,axes],matrix=rotM)
    return locs,vecs