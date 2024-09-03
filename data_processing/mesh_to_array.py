import meshio
import pyvista as pv
import numpy as np
import os

def convert_to_vtk(nas_file_path):
    '''
    Converts a .nas file to a .vtk file
    '''
    mesh = meshio.read(nas_file_path)
    vtk_file_path = nas_file_path.split('.nas')[0]+'.vtk'
    meshio.write(vtk_file_path, mesh)

def mesh_to_voxels(mesh: pv.PolyData, density:float):
    '''
    returns a numpy boolean mask of voxels from mesh
    '''
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(mesh.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)
    mask = selection['SelectedPoints'].view(bool)
    mask = mask.reshape(x.shape, order='F')
    mask = np.array(mask)
    return mask

def mesh_to_image(mesh, resolution=512, z_pos=None, box_size=None, return_mask=False):
    '''
    Extracts an image from the given mesh along the z_pos.
    '''
    if isinstance(mesh, str):
        if mesh.split('.')[1]=='nas':
            print('Convert .nas file to vtk', end='...')
            convert_to_vtk(mesh)
            mesh = mesh.split('.')[0]+'.vtk'
            print(' Complete.')
        elif mesh.split('.')[1]=='vtk':
            print('Loading .vtk file', end='...')
            mesh = pv.read(mesh)
            print(' Complete.')
        else:
            raise TypeError('File format not supported.')
    else:
        None

    x_min, x_max = mesh.bounds[:2]
    y_min, y_max = mesh.bounds[2:4]
    z_min, z_max = mesh.bounds[4:6]
    x_mean = np.mean(np.array([x_max,x_min]))
    y_mean = np.mean(np.array([y_max,y_min]))
    z_mean = np.mean(np.array([z_max,z_min]))
    center_vector = np.array([x_mean, y_mean, z_mean])

    if z_pos==None:
        z_pos = z_mean
        print(f'z_pos set to {z_pos}.')
    if box_size==None:
        z_length = z_max-z_min
        box_size = 0.005*z_length

    # rename cell data
    celldata_name = mesh.cell_data.keys()[0]
    mesh.cell_data['PartId'] = mesh.cell_data[celldata_name]
    # clip mesh to reduce sample times
    z_min = z_pos-box_size
    z_max = z_pos+box_size
    mesh_clipped = mesh.clip_box([x_min, x_max, y_min, y_max, z_min, z_max], invert=False)

    # define coordinates for slice
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=-1)
    xy = xy.reshape(-1,2)
    z = np.ones_like(xy[:,0])*z_pos
    xy = np.concatenate([xy, z.reshape(-1,1)], axis=1)
    points = pv.StructuredGrid(xy)

    # sample values for slice from clipped mesh 
    interpolated = points.sample(mesh_clipped)
    label = interpolated['PartId'].reshape(resolution, resolution, 1)
    if return_mask:
        return np.where(np.asarray(label)>0, 1, 0)    
    points = np.asarray(points.points) #- np.mean(np.asarray(points.points), axis=0)
    return np.asarray(label), points, center_vector
