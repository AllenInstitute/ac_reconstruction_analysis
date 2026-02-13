#!/usr/bin/env python
# coding: utf-8


import navis
import numpy as np
import matplotlib.pyplot as plt
from cloudvolume import Skeleton
from tqdm import tqdm
import os
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from uuid import uuid4
import json
import glob
import navis




def kimi_to_navis(skels):
    out_sk = navis.NeuronList(None)
    try:
        for sk in skels:
            out_sk.append(navis.NeuronList(sk.to_swc()))
    except:
        out_sk.append(navis.NeuronList(skels.to_swc()))
    return out_sk
    



    
    
import os
import numpy as np
import io
import glob
from io import BytesIO
import h5py
from uuid import uuid4



def swc_to_skeleton(swc_bytes, segid):
    """
    Convert SWC bytes (from HDF5) to a CloudVolume Skeleton object.
    
    Parameters
    ----------
    swc_bytes : np.ndarray or bytes
        Flattened uint8 bytes of SWC text.
    segid : int
        Segment ID for the skeleton.
    
    Returns
    -------
    Skeleton object
    """
    # Decode bytes to string
    if isinstance(swc_bytes, np.ndarray):
        swc_text = swc_bytes.tobytes().decode("utf-16")
    else:
        swc_text = swc_bytes.decode("utf-16")
    
    # Split into lines and parse
    lines = [line.strip() for line in swc_text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError(f"SWC for segid {segid} is empty")

    swc_array = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32)

    skel = Skeleton()
    skel.id = segid
    skel.vertices = swc_array[:, 2:5].copy()
    skel.vertex_types = swc_array[:, 1].astype(np.uint32)
    skel.radius = swc_array[:, 5].copy()

    edges = [(int(p)-1, i) for i, p in enumerate(swc_array[:, 6]) if p > 0]
    skel.edges = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    
    return skel


import os
import numpy as np
import io
import glob
from io import BytesIO
import h5py
from uuid import uuid4
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



def swc_to_skeleton(swc_bytes, segid):
    """
    Convert SWC bytes (from HDF5) to a CloudVolume Skeleton object.
    
    Parameters
    ----------
    swc_bytes : np.ndarray or bytes
        Flattened uint8 bytes of SWC text.
    segid : int
        Segment ID for the skeleton.
    
    Returns
    -------
    Skeleton object
    """
    # Decode bytes to string
    if isinstance(swc_bytes, np.ndarray):
        swc_text = swc_bytes.tobytes().decode("utf-16")
    else:
        swc_text = swc_bytes.decode("utf-16")
    
    # Split into lines and parse
    lines = [line.strip() for line in swc_text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError(f"SWC for segid {segid} is empty")

    swc_array = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32)

    skel = Skeleton()
    skel.id = segid
    skel.vertices = swc_array[:, 2:5].copy()
    skel.vertex_types = swc_array[:, 1].astype(np.uint32)
    skel.radius = swc_array[:, 5].copy()

    edges = [(int(p)-1, i) for i, p in enumerate(swc_array[:, 6]) if p > 0]
    skel.edges = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    
    return skel


def read_shard(shard_full_path, segid_filter=None, bbox=None):
    skels = []

    with h5py.File(shard_full_path, "r") as f:
        index_ds = f["index"]
        verts_ds = f["vertices"]
        edges_ds = f["edges"]
        rad_ds   = f["radius"]
        types_ds = f["types"]

        segids = index_ds[:, 0] ###EDIT
  

        for i, segid in enumerate(segids):
            if not np.isfinite(segid):
                continue
            if segid < 0:
                continue
            if segid_filter and segid not in segid_filter:
                continue

            if bbox is not None:
                xmin, xmax = index_ds[i, 1], index_ds[i, 2]
                ymin, ymax = index_ds[i, 3], index_ds[i, 4]
                zmin, zmax = index_ds[i, 5], index_ds[i, 6]
                qxmin, qymin, qzmin, qxmax, qymax, qzmax = bbox
                if (xmax < qxmin or xmin > qxmax or
                    ymax < qymin or ymin > qymax or
                    zmax < qzmin or zmin > qzmax):
                    continue

            skel = Skeleton()
            skel.id = segid

            skel.vertices = verts_ds[i].reshape(-1, 3)
            skel.edges = edges_ds[i].reshape(-1, 2)
            skel.radius = rad_ds[i]
            skel.vertex_types = types_ds[i]

            skels.append(skel)

    return skels



def _load_all_json_matching(shard_dir, pattern):
    """
    Internal helper: load and merge all JSON files matching pattern.
    Returns a combined dict.
    """
    merged = {}
    for path in glob.glob(os.path.join(shard_dir, pattern)):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                merged.update({str(k): v for k, v in data.items()})
        except Exception as e:
            print(f"?? Skipping {path}: {e}")
    return merged


# ---------------------------------------------------------------------
#  GLOBAL + ID INDEX LOADERS
# ---------------------------------------------------------------------

def load_global_index(shard_dir):
    """
    Load and merge *all* global shard index JSON files in a directory.
    (Matches '*global*index.json')
    """
    global_index = _load_all_json_matching(shard_dir, "*global*index.json")
    if not global_index:
        raise FileNotFoundError(f"No global index files found in {shard_dir}")
    return global_index


def load_id_to_shard_index(shard_dir):
    """
    Load and merge *all* id?shard index JSON files in a directory.
    (Matches '*id_to_shard*index.json')
    """
    id_to_shard = _load_all_json_matching(shard_dir, "*id_to_shard*index.json")
    if not id_to_shard:
        raise FileNotFoundError(f"No id_to_shard index files found in {shard_dir}")
    return id_to_shard



def overwrite_skeletons_to_shards(skeletons, shard_to_ids):
    """
    Overwrite updated skeletons into their corresponding SWC-formatted HDF5 shards.

    Parameters
    ----------
    skeletons : list of Skeleton
        Updated CloudVolume Skeleton objects.
    shard_to_ids : dict
        Mapping of shard_path -> list of skeleton IDs to write back in that shard.
    """
    # Build a lookup table from skeleton ID -> Skeleton object
    skel_lookup = {skel.id: skel for skel in skeletons}

    for shard_path, ids in tqdm(shard_to_ids.items(), desc="Writing skeletons back to shards"):
        with h5py.File(shard_path, "r+") as f:
            index_ds = f["index"]
            swc_ds = f["skeletons"]
            segids = index_ds[:, 0].astype(int)

            for skel_id in ids:
                if skel_id not in skel_lookup:
                    continue  # nothing to write for this ID
                skeleton = skel_lookup[skel_id]

                # Find the index of the skeleton in the shard
                idx = np.where(segids == skel_id)[0]
                if len(idx) == 0:
                    raise ValueError(f"Skeleton ID {skel_id} not found in shard {shard_path}")
                idx = idx[0]

                # Rebuild SWC array
                n = len(skeleton.vertices)
                swc = np.zeros((n, 7), dtype=np.float32)
                swc[:, 0] = np.arange(n)                    # Node IDs
                swc[:, 1] = skeleton.vertex_types           # Node types
                swc[:, 2:5] = skeleton.vertices            # X,Y,Z
                swc[:, 5] = skeleton.radius                 # Radius
                parent = np.full(n, -1, dtype=np.float32)
                for p, c in skeleton.edges:
                    parent[c] = p
                swc[:, 6] = parent                          # Parent IDs

                # Overwrite SWC dataset
                swc_ds[idx] = swc.flatten()

                # Update index bounding box & vertex count
                xmin, ymin, zmin = skeleton.vertices.min(axis=0)
                xmax, ymax, zmax = skeleton.vertices.max(axis=0)
                index_ds[idx, 1:7] = [xmin, xmax, ymin, ymax, zmin, zmax]
                index_ds[idx, 7] = n



def save_id_to_shard_h5(id_to_shard, output_path):
    """
    Save a large id_to_shard mapping to HDF5.

    id_to_shard: dict[int, str]  # segid -> shard path
    output_path: str
    """
    # Get unique shard paths
    shard_paths = sorted(set(id_to_shard.values()))
    shard_to_idx = {p: i for i, p in enumerate(shard_paths)}

    segids = np.array(list(id_to_shard.keys()), dtype=np.uint64)
    shard_indices = np.array([shard_to_idx[id_to_shard[k]] for k in segids], dtype=np.uint32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("segids", data=segids, compression="gzip")
        f.create_dataset("shard_indices", data=shard_indices, compression="gzip")
        # Store shard paths as variable-length strings
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("shard_paths", data=np.array(shard_paths, dtype=dt))


def load_id_to_shard_h5(h5_path):
    """
    Load ID -> shard mapping from HDF5.

    Returns:
        id_to_shard : dict[int, str]
    """
    with h5py.File(h5_path, "r") as f:
        segids = f["segids"][:]
        shard_indices = f["shard_indices"][:]
        shard_paths = f["shard_paths"][:]

    # Build mapping
    id_to_shard = {int(segid): shard_paths[idx].decode() if isinstance(shard_paths[idx], bytes) else shard_paths[idx]
                   for segid, idx in zip(segids, shard_indices)}
    return id_to_shard

def write_single_shard(shard_skeleton_items, output_dir):

    shard_dict = dict(shard_skeleton_items)

    shard_id = random.randint(10**13, 10**14 - 1)
    shard_name = f"{shard_id:03d}.h5"
    shard_path = os.path.join(output_dir, shard_name)

    shard_xmin = shard_ymin = shard_zmin = np.inf
    shard_xmax = shard_ymax = shard_zmax = -np.inf

    with h5py.File(shard_path, "w") as f:
        vlen_f32 = h5py.vlen_dtype(np.float32)
        vlen_i32 = h5py.vlen_dtype(np.int32)
        vlen_u32 = h5py.vlen_dtype(np.uint32)

        verts_ds = f.create_dataset("vertices", (len(shard_dict),), dtype=vlen_f32)
        edges_ds = f.create_dataset("edges",    (len(shard_dict),), dtype=vlen_i32)
        rad_ds   = f.create_dataset("radius",   (len(shard_dict),), dtype=vlen_f32)
        types_ds = f.create_dataset("types",    (len(shard_dict),), dtype=vlen_u32)
        index_ds = f.create_dataset("index", (len(shard_dict), 8), dtype="float32")

        for i, (segid, skel) in enumerate(shard_dict.items()):
            v = skel.vertices.astype(np.float32)
            e = skel.edges.astype(np.int32)
            r = skel.radius.astype(np.float32)
            t = skel.vertex_types.astype(np.uint32)

            verts_ds[i] = v.flatten()
            edges_ds[i] = e.flatten()
            rad_ds[i]   = r
            types_ds[i] = t

            xmin, ymin, zmin = v.min(axis=0)
            xmax, ymax, zmax = v.max(axis=0)

            index_ds[i] = [segid, xmin, xmax, ymin, ymax, zmin, zmax, len(v)]

            shard_xmin = min(shard_xmin, xmin)
            shard_ymin = min(shard_ymin, ymin)
            shard_zmin = min(shard_zmin, zmin)
            shard_xmax = max(shard_xmax, xmax)
            shard_ymax = max(shard_ymax, ymax)
            shard_zmax = max(shard_zmax, zmax)

    id_to_shard = {int(segid): shard_name for segid in shard_dict.keys()}

    return shard_name, [
        int(shard_xmin), int(shard_ymin), int(shard_zmin),
        int(shard_xmax), int(shard_ymax), int(shard_zmax)
    ], id_to_shard


def shard_and_write_skeletons(skeletons, output_dir,
                              max_skeletons_per_shard=10000,
                              label="", n_workers=4):

    os.makedirs(output_dir, exist_ok=True)
    global_shard_index = {}
    id_to_shard_index = {}

    if isinstance(skeletons, dict):
        shard_items = list(skeletons.items())
    else:
        dic = {}
        for sk in skeletons:
            dic[sk.id] = sk
        shard_items = list(dic.items())
        
    shards = [
        shard_items[i:i + max_skeletons_per_shard]
        for i in range(0, len(shard_items), max_skeletons_per_shard)
    ]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(write_single_shard,shard_items_chunk, output_dir)
                for i, shard_items_chunk in enumerate(shards)
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Writing shards"):
                shard_name, bbox, idmap = f.result()
                global_shard_index[shard_name] = bbox
                id_to_shard_index.update(idmap)
    else:
        # Serial fallback
        for i, shard_items_chunk in enumerate(shards):
            shard_name, bbox, idmap = write_single_shard(shard_items_chunk, output_dir)
            global_shard_index[shard_name] = bbox
            id_to_shard_index.update(idmap)

    # Save the global index
    with open(os.path.join(output_dir, f"{label}_global_shard_index.json"), "w") as f:
        json.dump(global_shard_index, f, indent=2)

    # Save ID to shard mapping
    save_id_to_shard_h5(
        id_to_shard_index,
        os.path.join(output_dir, f"{label}_id_to_shard.h5")
    )

    return global_shard_index, id_to_shard_index



def load_all_skeletons(shard_dir, n_workers=1):
    global_index = load_global_index(shard_dir)
    shard_names = list(global_index.keys())

    if not shard_names:
        return [], {}

    all_skeletons = []
    shard_to_ids = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    read_shard,
                    os.path.join(shard_dir, shard_name)
                ): shard_name
                for shard_name in shard_names
            }

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading all shards"
            ):
                shard_name = futures[f]
                skels = f.result()
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]
    else:
        for shard_name in tqdm(shard_names, desc="Loading all shards"):
            shard_path = os.path.join(shard_dir, shard_name)
            skels = read_shard(shard_path)
            all_skeletons.extend(skels)
            shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids


def query_skeletons_by_bb(query_bbox, shard_dir, n_workers=1):
    global_index = load_global_index(shard_dir)

    qxmin, qymin, qzmin, qxmax, qymax, qzmax = query_bbox
    candidates = []

    for shard_name, bbox in global_index.items():
        sxmin, symin, szmin, sxmax, symax, szmax = bbox
        if not (sxmax < qxmin or sxmin > qxmax or
                symax < qymin or symin > qymax or
                szmax < qzmin or szmin > qzmax):
            candidates.append(shard_name)

    if not candidates:
        return [], {}

    all_skeletons = []
    shard_to_ids = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    read_shard,
                    os.path.join(shard_dir, shard_name),
                    bbox=query_bbox
                ): shard_name
                for shard_name in candidates
            }

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading shards"
            ):
                shard_name = futures[f]
                skels = f.result()
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]
    else:
        for shard_name in tqdm(candidates, desc="Loading shards"):
            shard_path = os.path.join(shard_dir, shard_name)
            skels = read_shard(shard_path, bbox=query_bbox)
            all_skeletons.extend(skels)
            shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids

def query_skeletons_by_id(segids, shard_dir, n_workers=1):
    segids = set(segids if isinstance(segids, (list, set)) else [segids])
    id_to_shard = load_id_to_shard_index(shard_dir)

    shard_groups = {}
    for segid in segids:
        shard_name = id_to_shard.get(str(segid))
        if shard_name:
            shard_groups.setdefault(shard_name, []).append(segid)

    all_skeletons = []
    shard_to_ids = {}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    read_shard,
                    os.path.join(shard_dir, shard_name),
                    segid_filter=set(ids)
                ): shard_name
                for shard_name, ids in shard_groups.items()
            }

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Reading shards by ID"
            ):
                shard_name = futures[f]
                skels = f.result()
                all_skeletons.extend(skels)
                shard_to_ids[shard_name] = [s.id for s in skels]
    else:
        for shard_name, ids in tqdm(
            shard_groups.items(),
            desc="Reading shards by ID"
        ):
            shard_path = os.path.join(shard_dir, shard_name)
            skels = read_shard(shard_path, segid_filter=set(ids))
            all_skeletons.extend(skels)
            shard_to_ids[shard_name] = [s.id for s in skels]

    return all_skeletons, shard_to_ids


def delete_skeletons_in_shard(shard_path, skeleton_ids_set):
    """
    Load a shard and mark specified skeleton IDs as deleted.
    """
    with h5py.File(shard_path, "r+") as f:
        index_ds = f["index"]
        segids = index_ds[:, 0].astype(int)
        mask = np.isin(segids, list(skeleton_ids_set))
        indices_to_delete = np.where(mask)[0]
        if len(indices_to_delete) == 0:
            return None  # No matching skeletons
        
        # Mark as deleted
        index_ds[indices_to_delete, 0] = -1
        # Optionally, zero out associated datasets here

def delete_skeletons_parallel(shard_dir, skeleton_ids, n_workers=4):
    """
    Load global index, get shard names, and delete specified skeleton IDs in parallel.
    """
    # Load global index to get shard names
    global_index = load_global_index(shard_dir)
    shard_names = list(global_index.keys())

    # Convert skeleton IDs to set for faster lookup
    skeleton_ids_set = set(skeleton_ids)

    # Create full paths for each shard
    shard_paths = [os.path.join(shard_dir, shard_name) for shard_name in shard_names]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(delete_skeletons_in_shard, shard_path, skeleton_ids_set)
            for shard_path in shard_paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Deleting skeletons"):
            future.result()  # To catch exceptions if any




