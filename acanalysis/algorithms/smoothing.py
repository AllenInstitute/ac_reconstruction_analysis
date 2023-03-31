import numpy


def points_in_path_distance(pts, dist, include_point=True):
    pts_in_distance = []
    dpts_root = numpy.linalg.norm(pts - pts[0], axis=1)
    for idx, root_d in enumerate(dpts_root):
        r = root_d - dist, root_d + dist
        mask = (dpts_root > r[0]) & (dpts_root < r[1])
        if not include_point:
            mask[idx] = False
        pts_in_distance.append(pts[mask])
    return pts_in_distance


def gaussian_smooth_points(
        pts, sigma=3000., min_effect=1e-6):
    max_dist = numpy.sqrt(
        -numpy.log(min_effect) * 2 * sigma * sigma)
    md2 = max_dist * max_dist
    npts = points_in_path_distance(
        pts, md2, include_point=False)
    spts = []
    for (i, pt) in enumerate(pts):
        if i == 0 or i == len(pts) - 1 or len(npts[i]) == 0:
            spts.append(pt)
            continue
        # average npts weighted by distance
        # TODO could use precalculated distances and indices for speed
        deltas = npts[i] - pt
        dists = numpy.linalg.norm(deltas, axis=1)
        # prepend point
        apts = numpy.vstack([pt, npts[i]])
        # weights for apts basedd on distances
        ws = numpy.ones(pts.shape[0])
        ws[1:] = numpy.exp(-(dists ** 2. / (2. * sigma ** 2.)))
        ws /= ws.sum()

        # spt = numpy.sum(apts * nws[:, numpy.newaxis], axis=0)
        spt = numpy.einsum("ij,ij->j", apts, ws[:, numpy.newaxis])
        spts.append(spt)
    return numpy.array(spts)
