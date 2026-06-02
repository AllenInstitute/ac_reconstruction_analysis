import pymaid
import navis
import numpy as np


def export_neurons(server, api_token, project_id, min_nodes=0, annotations=None, skids=None, reviewed_by=None):
    #Load instance and set project
    rm = pymaid.CatmaidInstance(server=server, api_token=api_token)
    rm.project_id = project_id

    #format annotations
    if annotations:
        annot_list = []
        for ann in annotations:
            annot_list.append('annotation:'+str(ann))
        annotations = annot_list

    skels = pymaid.find_neurons(min_size=min_nodes, annotations=annotations, skids=skids, reviewed_by=reviewed_by)

    return skels
    

def import_neurons(skels, server, api_token, project_id, rad_annot=True, res=[1,1,1], strip=0):
    #Load instance and set project
    rm = pymaid.CatmaidInstance(server=server, api_token=api_token)
    rm.project_id = project_id

    #Import neurons
    for ind,sk in enumerate(skels):
        print(ind)
        nodes = sk.nodes.copy()
        nodes = nodes[['node_id', 'parent_id', 'x', 'y', 'z', 'radius']]
        neu = navis.TreeNeuron(nodes)
        if res != [1,1,1]:
            x,y,z = res
            neu.nodes['x'] = neu.nodes['x']*x
            neu.nodes['y'] = neu.nodes['y']*y
            neu.nodes['z'] = neu.nodes['z']*z
        resp = pymaid.upload_neuron(neu)
        if rad_annot==True:
            rad = round(max(list(neu.nodes['radius'])))
            #resp = pymaid.add_annotations(int(resp['skeleton_id']), "RADIUS:" + str(rad))
            resp = pymaid.add_annotations(int(resp['skeleton_id']), "Pos"+str(strip))
            
            
