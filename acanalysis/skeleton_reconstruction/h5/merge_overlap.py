import argschema
import pathlib
import numpy as np
import os
from cloudvolume import Skeleton
from scipy.spatial import cKDTree

from h5_skeletons import *
from h5_reconnect import *


def find_merged_neurons(unmerged, merged):
    log = []
    for sk in unmerged:
        log.append(list(sk.radii))

    unchanged, changed = [], []
    for sk in merged:
        if list(sk.radii) in log:
            unchanged.append(sk)
        else:
            changed.append(sk)

    return unchanged, changed


def find_skel_ids(ref_skels, alt_skels):
    ref_vertices = []
    ref_vertex_skel_ids = []
    for sk in ref_skels:
        ref_vertices.append(sk.vertices[0])
        ref_vertices.append(sk.vertices[-1])
        ref_vertex_skel_ids.append(sk.id)
        ref_vertex_skel_ids.append(sk.id)
    ref_vertices = np.array(ref_vertices)
    ref_vertex_skel_ids = np.array(ref_vertex_skel_ids)
    tree = cKDTree(ref_vertices)

    used_ref_ids = set()
    for sk in alt_skels:
        if len(sk.vertices) == 0:
            continue
        query_pts = np.array([sk.vertices[0], sk.vertices[-1]])
        _, idxs = tree.query(query_pts, k=1)
        candidate_ids = ref_vertex_skel_ids[idxs]
        unique, counts = np.unique(candidate_ids, return_counts=True)
        best_id = unique[np.argmax(counts)]
        sk.id = int(best_id)
        used_ref_ids.add(int(best_id))

    all_ref_ids = {sk.id for sk in ref_skels}
    unused_ref_ids = all_ref_ids - used_ref_ids
    return alt_skels, list(unused_ref_ids)


def get_volume_bounds(shard_dir):
    """
    Derive the full volume bounding box by taking the union
    of all shard bboxes in the global shard index.
    Returns (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    global_index = load_global_index(shard_dir)
    if not global_index:
        raise ValueError(f"No shards found in {shard_dir}")

    bboxes = np.array(list(global_index.values()))  # shape (N, 6): xmin,ymin,zmin,xmax,ymax,zmax
    xmin, ymin, zmin = bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].min()
    xmax, ymax, zmax = bboxes[:, 3].max(), bboxes[:, 4].max(), bboxes[:, 5].max()
    return int(xmin), int(ymin), int(zmin), int(xmax), int(ymax), int(zmax)


class CloudOptions(argschema.schemas.DefaultSchema):
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-east-1')
    bucket = argschema.fields.String(required=False, default=None, allow_none=True)
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)


class FuseSkeletonsParameters(argschema.ArgSchema):
    skel_dir = argschema.fields.String(required=True, metadata={'description': 'Path to skeleton shards directory'})
    chunk_size = argschema.fields.Int(required=False, dump_default=1000, metadata={'description': 'Chunk size in voxels (applied to all three dimensions)'})
    n_workers = argschema.fields.Int(required=False, dump_default=10, metadata={'description': 'Number of parallel workers'})


class FuseSkeletonsModule(argschema.ArgSchemaParser):
    default_schema = FuseSkeletonsParameters

    @property
    def cloud_options(self):
        try:
            return self.args["cloud_options"]
        except:
            return {}

    def run(self):
        skel_dir = self.args['skel_dir']
        chunk = self.args['chunk_size']
        n_workers = self.args['n_workers']

        xmin, ymin, zmin, xmax, ymax, zmax = get_volume_bounds(skel_dir)
        print(f"Volume bounds: x=[{xmin},{xmax}] y=[{ymin},{ymax}] z=[{zmin},{zmax}]")

        # Generate all 3D chunks
        chunks = []
        for x in range(xmin, xmax, chunk):
            for y in range(ymin, ymax, chunk):
                for z in range(zmin, zmax, chunk):
                    cutout = (x, y, z, x + chunk, y + chunk, z + chunk)
                    chunks.append(cutout)

        print(f"Total chunks to process: {len(chunks)}")

        for cutout in chunks:
            skels, shards = query_skeletons_by_bb(cutout, skel_dir, n_workers=n_workers)

            if not skels:
                continue                                

            fused = Skeleton.simple_merge(skels).consolidate().components()

            if len(fused) == len(skels):
                continue

            fused = [x for x in fused if x]

            fused, unused_ids = find_skel_ids(skels, fused)
            unmerged, merged = find_merged_neurons(skels, fused)
            fused = prune_to_furthest_end_path(merged)

            del_ids = unused_ids + [x.id for x in fused]
            print(f"Cutout: {cutout}   # Skels: {len(skels)}   # Fused: {len(fused)}   # Del: {len(del_ids)}")

            delete_skeletons_parallel(skel_dir, del_ids, n_workers=n_workers)
            shard_and_write_skeletons(
                fused, skel_dir,
                max_skeletons_per_shard=10000,
                label=f"fused_{str(cutout)}",
                n_workers=n_workers
            )


if __name__ == "__main__":
    mod = FuseSkeletonsModule()
    mod.run()


__all__ = [
    "FuseSkeletonsModule",
    "FuseSkeletonsParameters"
]

 