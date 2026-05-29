import navis
import os
import acanalysis.skeleton_reconstruction.reconnection.reconnect_navis as rec


def test_reconnect():
    path=os.path.dirname(os.path.realpath(__file__))
    skels = navis.example_neurons(n=10, kind='skeleton')
    nm,m,ids = rec.reconnect(skels, cl=path+'/model/LR_1.joblib', sc=path+'/model/scaler.joblib')
    nm,m,ids = rec.reconnect(skels, cl=None, sc=None)
