import numpy as np
import pyvista as pv
from fiberorient.odf import ODF
from fiberorient.util import make_sphere


def plot_odf(X, points=True):
    """Render an Orientation Distribution Function as a 3-D mesh.

    Parameters
    ----------
    X : array-like, shape (N, 3)
        Unit orientation vectors.
    points : bool
        If True, overlay the raw input points on the mesh.
    """
    pv.set_jupyter_backend("client")
    plotter = pv.Plotter()

    odf = ODF(degree=6)
    odf.fit(X)

    sphere = make_sphere(3000)
    odf_on_sphere = odf.to_sphere(sphere) * 3

    faces = sphere.faces.copy()
    new_faces = np.concatenate(
        (np.full(faces.shape[0], 3).reshape(-1, 1), faces), axis=1
    )
    odf_mesh = pv.PolyData(sphere.vertices * odf_on_sphere[:, None], faces=new_faces)
    odf_mesh["odf"] = odf_on_sphere
    plotter.add_mesh(odf_mesh, scalars="odf")

    if points:
        point_cloud = pv.PointSet(X)
        plotter.add_mesh(point_cloud, point_size=3)

    plotter.show()
