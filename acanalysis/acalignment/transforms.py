# -*- coding: utf-8 -*-
"""
modified mpyicbg ransac code

"""

import logging
import numpy

from mpyicbg.errors import MPYICBGError
from mpyicbg.utils.utils import NullHandler
from mpyicbg.utils.numpy_utils import get_random_state
from mpyicbg.transformations import SimilarityModel2D
from renderapi.transform.leaf import thin_plate_spline

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

def _dynamic_max_trials(*args, **kwargs):
    return numpy.inf

def is_iterable(obj):
    try:
        _ = (i for i in obj)  # noqa: F841
        return True
    except TypeError:
        return False

def _always_true(*args, **kwargs):
    """ALWAYS TRUE, ALWAYS TRUE"""
    return True

def ransac(candidates, epsilon, iterations=None,
           minNumInliers=None, maxNumInliers=numpy.inf,
           data_validator=_always_true, model_validator=_always_true,
           stop_probability=1., random_state=None, **kwargs):
    if not is_iterable(candidates):
        raise ValueError("Invalid candidates of type {}".format(
            type(candidates)))
        
    minNumMatches = 5

    candidates = list(candidates)
    candidate_shapes = {c.shape for c in candidates}
    if len(candidate_shapes) != 1:
        raise ValueError(
            "shape mismatch in candidates! {}".format(candidate_shapes))
    num_samples = list(candidate_shapes)[0][0]
    if num_samples < minNumMatches:
        raise ValueError(
            "need {} points to fit ransac, got {}".format(
                minNumMatches, num_samples))

    # cost = Double.MAX_VALUE
    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = numpy.inf
    best_inliers = None

    random_state = get_random_state(random_state)

    for num_trial in range(iterations):
        #print(str(num_trial) + " out of " + str(iterations))
        indices = random_state.randint(0, num_samples, minNumInliers)
        samples = [d[indices] for d in candidates]

        if not data_validator(*samples):
            continue

        try:
            #sample_model = model.estimate(*samples)
            sample_model = SimilarityModel2D(None,*samples,dim=2,estimate_kwargs={"rigid": True})
        except MPYICBGError:  # TODO custom error handling
            continue

        if not model_validator(sample_model, *samples):
            continue

        #sample_model_residuals = numpy.linalg.norm(
            #sample_model.residual_displacement(*candidates), axis=1)
        sample_model_residuals = sample_model.residual_distance(*candidates)
        sample_model_inliers = sample_model_residuals < epsilon
        sample_model_residuals_sum = numpy.sum(sample_model_residuals ** 2)
        sample_inlier_num = numpy.sum(sample_model_inliers)
        # print(sample_inlier_num)
        # sample_inlier_ratio = numpy.true_division(
        #     sample_inlier_num, sample_model_residuals.shape[0])
        # sample_inlier_cost = max(0., min(1., 1. - sample_inlier_ratio))

        if (sample_inlier_num > best_inlier_num
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)):
            print(sample_inlier_num)
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            # TODO implement cost function for max trials
            # best_inlier_ratio = sample_inlier_ratio
            # best_inlier_cost = sample_inlier_cost
            # TODO actual implementation for maxNumInliers
            if (best_inlier_num >= maxNumInliers
                or num_trial >= _dynamic_max_trials(
                    best_inlier_num, num_samples, minNumInliers,
                    stop_probability)):
                break
            #if best_inliers is not None:
            #    for i, d in enumerate(candidates):
            #        candidates[i] = d[best_inliers]
            #    best_model.estimate(*candidates)
    return best_model, best_inliers


def tps_transform(srcpts,dstpts,computeAffine=True):
    tps = thin_plate_spline.ThinPlateSplineTransform()
    tps.estimate(srcpts,dstpts,computeAffine=computeAffine)
    return tps