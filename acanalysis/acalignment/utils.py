# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:28:09 2023

@author: kevint
"""

import navis
import numpy

def shift_navis_xform(skels,shift_xyz):
    M = numpy.diag([1,1,1,1])
    if shift_xyz:
        M[:3,3] = numpy.array(shift_xyz)
    tr = navis.transforms.AffineTransform(M)
    return navis.xform(skels, tr)