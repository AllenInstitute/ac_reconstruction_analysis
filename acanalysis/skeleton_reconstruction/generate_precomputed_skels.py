import numpy as np
import os
from cloudvolume import CloudVolume, Skeleton
import navis
import uuid
import argparse
from acanalysis.skeleton_reconstruction.util import read_navis_neurons_tar
from acanalysis.skeleton_reconstruction.neuroglancer import generate_ngl_segmentation_empty

def generate_precomputed_skeletons(skel_file, outdir, orient=False, match_fname=False, oid=0):
    os.makedirs(outdir, exist_ok=True)
    
    # Create segmentation folder
    generate_ngl_segmentation_empty(outdir)
    
    # Read skeletons
    swcs = read_navis_neurons_tar(skel_file)
    skel_ds = navis.downsample_neuron(swcs, 4)
    
    # Initialize CloudVolume
    vol = CloudVolume(f'file://{outdir}', compress='')
    sk_info = vol.skeleton.meta.default_info()
    sk_info["vertex_attributes"] = [
        { 'id': 'radius',
          'data_type': 'float32',
          'num_components': 1
        }
    ]
    vol.skeleton.meta.info = sk_info
    vol.skeleton.meta.commit_info()
    
    skel_dir = os.path.join(outdir, "skeletons")
    if not os.path.exists(skel_dir):
        os.makedirs(skel_dir)
    
    skel_one = None
    
    for sk in skel_ds:
        if 'label' not in sk.nodes:
            sk.nodes.insert(1, 'label', list(np.zeros(len(sk.nodes))))
        
        # Reformat skeleton node table to SWC
        sk_nodes = navis.TreeNeuron(sk.nodes.copy()).nodes
        if isinstance(sk_nodes.label[0], str):
            sk_nodes.label = 0
        sk_nodes = sk_nodes[['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']]
        sk_nodes[['node_id', 'label', 'parent_id']] = sk_nodes[['node_id', 'label', 'parent_id']].astype(int)
        sk_nodes = list(list(x) for x in zip(*(sk_nodes[x].values.tolist() for x in sk_nodes.columns)))
        sk_nodes = '\n'.join(str(x)[1:-1] for x in sk_nodes).replace(",", "")
        skel = Skeleton.from_swc(sk_nodes)
        
        if skel_one is None:
            skel_one = skel
        else:
            skel_one = skel_one.merge(skel)
    
    skel_one.id = str(oid)
    vol.skeleton.upload(skel_one)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate precomputed skeletons")
    parser.add_argument("skel_file", type=str, help="Path to the skeleton file")
    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument("--orient", action="store_true", help="Include orientation vectors")
    parser.add_argument("--match_fname", action="store_true", help="Match filename")
    parser.add_argument("--oid", type=str, default=str(uuid.uuid4()), help="Object ID")

    args = parser.parse_args()
    generate_precomputed_skeletons(args.skel_file, args.outdir, args.orient, args.match_fname, args.oid)

# python /home/wanqing.yu/AC_Project/ac_visualization/generate_precomputed_skels.py /ACdata/Users/connorl/Skeletons/For_Wan-Qing/Whole_Sections/S32/stitched_interstrip_reconnect/POS52.swcs.tar.gz /ACdata/Users/connorl/Skeletons/For_Wan-Qing/Whole_Sections/S32/stitched_interstrip_reconnect_precomputed/ --orient --match_fname --oid 52