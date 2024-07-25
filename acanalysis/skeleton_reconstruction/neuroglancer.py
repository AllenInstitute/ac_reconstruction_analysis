import numpy as np
import os
from cloudvolume import CloudVolume, Skeleton
import logging
import glob
import json
import neuroglancer
import requests
import tifffile
import navis
import shutil
from acanalysis.skeleton_reconstruction.util import read_navis_neurons_tar, write_navis_skels_tar, swap_dimensions
from joblib import Parallel, delayed, parallel_config
import uuid
from natsort import natsorted


def add_point_annot(viewer, annot_df, point_trans=[0,0,0]):
    with viewer.txn() as s:
        #define an annotation layer
        s.layers['annotation'+str(i)] = neuroglancer.AnnotationLayer()
        annotations = s.layers['annotation'+str(i)].annotations

        #iterate over csv and add annotations
        counter = 1
        for index,row in annot_df.iterrows():
            pt = neuroglancer.PointAnnotation(id=str(counter), point=[row['x'],row['y'],row['z']])
            annotations.append(pt)
            counter += 1
            
    print(viewer)

def add_line_annot(viewer, annot_df, point_trans=[0,0,0]):
    with viewer.txn() as s:
        #define an annotation layer
        s.layers['annotation'+str(i)] = neuroglancer.AnnotationLayer()
        annotations = s.layers['annotation'+str(i)].annotations

        #iterate over csv and add annotations
        counter = 1
        for index,row in annot_df.iterrows():
            pt = neuroglancer.PointAnnotation(point_a=[row['x1'],row['y1'],row['z1']],point_b=[row['x2'],row['y2'],row['z2']], id=str(counter))
            annotations.append(pt)
            counter += 1

    print(viewer)

def zarr_to_ngl_link(zarr_fpath, ip = 'localhost', port='9999', server='http://bigkahuna.corp.alleninstitute.org', aff_transf=[0,0,0,0,0,0], im_pix_range=[0,10000], im_opacity=0.5):
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer=neuroglancer.Viewer()

    zarr_attr = zarr_fpath + '/.zattrs'
    f = open(zarr_attr)
    data = json.load(f)

    # Alter dimension order and set dimension scale
    scale = np.array(data['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][2:5])
    dim_im = neuroglancer.CoordinateSpace(
                names=['z', 'y', 'x', 't'],
                units='nm',
                scales=scale,
            )

    # Create coordinate transform for image
    x1,x2,y1,y2,z1,z2 = aff_transf
    im_matrix = np.array([[1,x1,x2,0],[y1,1,y2,0],[z1,z2,1,0]])
    tr_im = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_im, output_dimensions =  dim_im, matrix = im_matrix) 
    
    with viewer.txn() as s:
        s.dimensions = dim_im
        s.layers['image'] = neuroglancer.ImageLayer(source=['zarr://' + server + zarr_fpath])
        s.layers['image'].layer.source[0].transform  = tr_im
        s.layers['image'].layer.shaderControls = {'normalized': {'range': im_pix_range}}
        s.layers['image'].layer.opacity = im_opacity
        
    view = s.to_json()
    link = make_neuroglancer_url_vneurodata(view)
    return link

    
    
def generate_ngl_tiffs(source_path, out_path, chunk_size, resolution):
    """Create an neuroglancer precomputed volume using tiff files.
       source_path: directory underwhich tiff files will be found.
       out_path: directory for the precomputed volume.
    """
    
    # Read tiffstack into memory
    files = glob.glob(f"{source_path}/*.tif")
    files = sorted(files)
    data = None

    for layer in range(len(files)):
        image = tifffile.imread(files[layer])
        # Allocate array if needed; use the number of files and dimensions of first file
        if data is None:
            data = np.zeros(shape=(image.shape[1], image.shape[0], len(files)), dtype=image.dtype)
        data[:, :, layer] = image.T   # Tiff files have X & Y swapped

    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = 'uint8', # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, compresso
        encoding        = 'png', 
        resolution      = resolution, # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        chunk_size      = chunk_size, # units are voxels
        volume_size     = data.shape # e.g. a cubic millimeter dataset
        )

    vol = CloudVolume(f'file://{out_path}', info=info, compress='', cache=False)
    logging.info(f"Creating cloud volume: {vol.info}")
    vol.commit_info()
    vol.commit_provenance()
    vol[:,:,:] = data.astype(np.uint8)

def generate_ngl_segmentation_empty(out_path):
    """Create an empty neuroglancer precomputed segmentation volume.
       out_path: directory where new the new cloud volume segmentation layer should be generated
    """
    
    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'segmentation',
        data_type       = 'uint64', # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, compresso
        encoding        = 'compressed_segmentation', 
        resolution      = [0,0,0], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        chunk_size      = [0,0,0], # units are voxels
        volume_size     = [0,0,0], # e.g. a cubic millimeter dataset
        skeletons       = 'skeletons'
        )

    vol = CloudVolume(f'file://{out_path}', info=info, compress='', cache=False)
    logging.info(f"Creating cloud volume: {vol.info}")
    vol.commit_info()
    vol.commit_provenance()
        

def transform_ng_layer():
        im_matrix = np.array([[1,x1,x2,0],[y1,1,y2,0],[z1,z2,1,0]])
        skel_matrix = np.array([[1,x1,x2,transl_skel[0]/(1+skel_mip)],[y1,1,y2,transl_skel[1]/(1+skel_mip)],[z1,z2,1,transl_skel[2]/(1+skel_mip)]])
        tr_im = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_im, output_dimensions =  dim_im, matrix = im_matrix) 
        tr_skel = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_skel, output_dimensions = dim_skel, matrix = skel_matrix)

        tr_im = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_im, output_dimensions =  dim_im, matrix = im_matrix) 
        tr_skel = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_skel, output_dimensions = dim_skel, matrix = skel_matrix)
        
        
def make_neuroglancer_url_vneurodata(state,
                                     base_url="http://bigkahuna.corp.alleninstitute.org/neuroglancer",
                                     state_url="https://json.neurodata.io/v1"):
    r = requests.post(state_url, json=state)
    json_url = r.json()["uri"]
    link = f"{base_url}/#!{json_url}"
    print(link)

def tifs_to_ngl_link(source_path, out_path, res=[0,0,0], ip = 'localhost', port='9999'):
    """Create a neuroglancer viewer instance, create a precomputed volume using tiff files, then load the viewer.
       source_path: directory underwhich the tiff files will be found.
       out_path: directory where the precomputed volume will go.
    """
    
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer=neuroglancer.Viewer()
    
    #generate precomputed volume for tif/tifs
    generate_ngl_tiffs(source_path, out_path, [512, 512, 64], res)
    
    with viewer.txn() as s:
        s.layers['Image'] = neuroglancer.ImageLayer(source=['precomputed://http://bigkahuna.corp.alleninstitute.org/' + out_path])
            
    view = s.to_json()
    link = make_neuroglancer_url_vneurodata(view)  
    return link
    
def generate_ngl_skeletons(skels, out_path, match_fname=False, oid=0, tag=[]):
    """Generate skeletons from SWC files.
       This currently assumes the neuroglancer precomputed volume has already been generated by generate_ngl_segmentation.
       source_path: directory underwhich `swc_files_nm` will be found.
       out_path: directory with the previously generated segmentation layer.
    """

    vol = CloudVolume(f'file://{out_path}', compress='')
    vol.skeleton.meta.info.pop("vertex_attributes", None)
    vol.skeleton.meta.commit_info()
        
    skel_dir = os.path.join(out_path, "skeletons")
    if not os.path.exists(skel_dir):
        os.makedirs(skel_dir)
    
    skel_ids, values, skel_length = [],[],[]
    for sk in skels:
        # ..../NNNN.swc -> NNNN
        if 'label' not in sk.nodes:
            sk.nodes.insert(1, 'label', list(np.zeros(len(sk.nodes))))
        #reformat skeleton node table to swc
        sk = navis.TreeNeuron(sk.nodes.copy()).nodes
        sk = sk[['node_id', 'label','x','y','z','radius','parent_id']]
        sk[['node_id', 'label','parent_id']] = sk[['node_id', 'label','parent_id']].astype(int)
        sk = list(list(x) for x in zip(*(sk[x].values.tolist() for x in sk.columns)))
        sk = '\n'.join(str(x)[1:-1] for x in sk).replace(",", "")
        skel = Skeleton.from_swc(sk)

        if match_fname==True:
            skel.id = sk.name
            
        else:
            skel.id = str(int(uuid.uuid4()))[0:15]
 
        vol.skeleton.upload(skel)
        skel_ids.append(skel.id)
        values.append([oid]) 
        skel_length.append(str(skel.cable_length()))

    return [skel_ids,tag,values,skel_length]


def generate_ngl_segproperties(outdir, skel_ids=[], tags=[], values=[], skel_length=[]):
    segprops = {"@type": "neuroglancer_segment_properties",
            "inline" : {
                "ids" : skel_ids,
                "properties" : [
                    {"id": "tags",
                        "type": "tags",
                        "tags" : tags,
                        "values" : values
                    },
                    {"id": "length",
                        "type": "number",
                        "data_type" : "float32",
                        "values" : skel_length
                    }
                ]},
            }

    # Write the segment properties
    segment_info_dir = os.path.join(outdir, "skeletons/segment_properties")
    os.makedirs(segment_info_dir, exist_ok=True)
    with open(os.path.join(segment_info_dir, "info"), "w") as f:
        json.dump(segprops, f)
            
    # Re-write info file with added segment_properties
    with open(f'{os.path.join(outdir, "skeletons")}/info', 'r') as f:
        infofile = json.load(f)
    infofile['segment_properties'] = 'segment_properties'
        
    with open(f'{outdir}skeletons/info', 'w') as f:
        json.dump(infofile, f)

def create_ngl_link_StripsWITHSkels(zarr_dir, strip_range, skels_dir, skel_mip, ip = 'localhost', port='9999', pix_range=[0,10000], trans=[0,0,0]):
    """Create a neuroglancer viewer instance, then load multiple image zarrs and their associated skeleton data into it.
       zarr_path: directory underwhich the zarr image data can be found.
       strip_range: the list range of strips being visualized (EX: [0,10])
       skels_path: directory underwhich the precomputed volume for skeletons can be found (with strip subdirectories).
       skels_mip: mip level of the image data used to generate the skeletons
    """
    
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer=neuroglancer.Viewer()

    # Set image and skeleton variables
    image_source, image_trs = [],[]
    skel_source = 'precomputed://http://bigkahuna.corp.alleninstitute.org' + skels_dir + '/skeletons/'
    
    for ind,strip in enumerate(range(strip_range[0],strip_range[1]+1)):
        image_path = zarr_dir +'highres_Pos' + str(strip)
        image_source.append('zarr://http://bigkahuna.corp.alleninstitute.org' + image_path)
        image_attr = image_path + '/.zattrs'
        f = open(image_attr)
        data = json.load(f)
        
        # Pull translation and scale for image
        transl_im = np.array(data['multiscales'][0]['coordinateTransformations'][0]['translation'])[2:5]
        scale_im = np.array(data['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][2:5])
    
        # Alter dimension order and set dimension scale
        dim_im = neuroglancer.CoordinateSpace(
                    names=['z', 'y', 'x', 't'],
                    units='nm',
                    scales=scale_im*1000)
    
        # Create coordinate transforms to adjust for zarr mip level and translations
        im_matrix = np.array([[1,0,0,trans[0]],[0,1,0,trans[1]],[0,0,1,trans[2]]])
        tr_im = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_im, output_dimensions =  dim_im, matrix = im_matrix) 
        image_trs.append(tr_im)

        if ind == 0:
            # Pull translation and scale for skeletons
            transl_skel = np.array([transl_im[0]/scale_im[0],transl_im[1]/scale_im[1],transl_im[2]/scale_im[2]]).astype(float)
            scale_skel = np.array(data['multiscales'][0]['datasets'][skel_mip]['coordinateTransformations'][0]['scale'][2:5])

            # Alter dimension order and set dimension scale
            dim_skel = neuroglancer.CoordinateSpace(
                    names=['z', 'y', 'x', 't'],
                    units='nm',
                    scales=scale_skel*1000)

            # Create coordinate transforms to adjust for zarr mip level and translations
            skel_matrix = np.array([[1,0,0,transl_skel[0]/(1+skel_mip)],[0,1,0,transl_skel[1]/(1+skel_mip)],[0,0,1,transl_skel[2]/(1+skel_mip)]])
            skel_tr = neuroglancer.CoordinateSpaceTransform(input_dimensions = dim_skel, output_dimensions = dim_skel, matrix = skel_matrix)

    with viewer.txn() as s:
        s.layers['Image'] = neuroglancer.ImageLayer(source=image_source)
        s.layers['Image'].layer.shaderControls = {'normalized': {'range': pix_range}}
        s.layers['Skel'] = neuroglancer.SegmentationLayer(source=skel_source)
        s.dimensions = dim_im
        s.layers['Skel'].layer.source[0].transform  = skel_tr
        #for ind in range(len(image_source)):
            #s.layers['Image'].layer.source[ind].transform  = image_trs[ind]
        view = s.to_json()
    
    #Create shareable neuroglancer link
    return(make_neuroglancer_url_vneurodata(view))
    
    
def create_precompute_skels(file, outpath, pos, oid, swap_dim=False):
    print(pos)
    skels = read_navis_neurons_tar(file)
    if swap_dim==True:
        skels = swap_dimensions(navis.NeuronList(skels))
    skels = navis.NeuronList([x for x in skels if x.n_nodes>=10])
    
    #generate precomputed skeletons
    res = generate_ngl_skeletons(skels, out_path=outpath,  tag = [pos], oid=oid)
    return res