# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:02:13 2023

@author: kevint
"""

import dataclasses
import numpy
import json

@dataclasses.dataclass
class KeyPoint:
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
        

def read_keypoints(jsonpath):
    with open(jsonpath) as f:
        js = json.load(f)
    return [KeyPoint(name=d["name"],
                     location=numpy.array(d["location"]),
                     vector=numpy.array(d["vector"])) for d in js]