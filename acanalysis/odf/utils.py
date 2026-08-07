import numpy as np
import navis


def cart2sph(x, y, z):
    xy = np.sqrt(x**2 + y**2)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arctan2(y, x))
    phi = np.degrees(np.arctan2(xy, z))

    return r, theta, phi


def get_filtered_skeletons(skeletons, bbox):
    """Filter a NeuronList to nodes within a 3D bounding box.

    Parameters
    ----------
    skeletons : navis.NeuronList
    bbox : dict with keys 'x', 'y', 'z', each a (min, max) tuple
    """
    filtered = []

    for skel in skeletons:
        skel.nodes["in_bbox"] = (
            skel.nodes["x"].between(bbox["x"][0], bbox["x"][1])
            & skel.nodes["y"].between(bbox["y"][0], bbox["y"][1])
            & skel.nodes["z"].between(bbox["z"][0], bbox["z"][1])
        )
        skel_inbox = navis.subset_neuron(skel, skel.nodes["in_bbox"])
        if skel_inbox.n_nodes > 2:
            filtered.append(skel_inbox)

    return navis.NeuronList(filtered)
