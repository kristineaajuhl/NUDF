import os
#from data_processing.evaluation import eval_mesh
import traceback
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import SimpleITK as sitk

# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p):

    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)


    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    data_tupels = []
    for i, data in tqdm(enumerate(loader)):


        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])


        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue

        try:
            if len(data_tupels) > 0:
                create_pc(data_tupels)
                data_tupels = []

            surf_points = gen.generate_pc(data)          
            data_tupels.append((surf_points,data, out_path))


        except Exception as err:
            print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))

    try:

        create_pc(data_tupels)
        data_tupels = []
        surf_points = gen.generate_pc(data)
        data_tupels.append((surf_points, data, out_path))


    except Exception as err:
        print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))

# def save_mesh(data_tupel):
#     logits, data, out_path = data_tupel

#     mesh = gen.mesh_from_logits(logits)

#     path = os.path.normpath(data['path'][0])
#     export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

#     if not os.path.exists(export_path):
#         os.makedirs(export_path)

#     mesh.export(export_path + 'surface_reconstruction.stl')

# def create_meshes(data_tupels):
#     p = Pool(mp.cpu_count())
#     p.map(save_mesh, data_tupels)
#     p.close()
        
def save_pc(data_tupels):
    points, data, out_path = data_tupels
    
    path = os.path.normpath(data['path'][0])
    export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])
    
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    np.savetxt(export_path + 'surf_points.txt',points)
 
def save_udf(data_tupel):
    print("SAVE UDF")
    logits, data, out_path = data_tupel
    dim = (np.round(len(logits)**(1/3)).astype(np.int),)*3
    voxel_grid_origin = [-1, -1, -1]
    voxel_size = 2.0 / (dim[0] - 1)

    #mesh = gen.mesh_from_logits(logits)

    path = os.path.normpath(data['path'][0])
    export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    #np.save(export_path + 'logits_reconstruction.npy',logits)
    numpy_3d_sdf_tensor = np.reshape(logits,dim)
    img = sitk.GetImageFromArray(numpy_3d_sdf_tensor)
    img.SetOrigin(voxel_grid_origin)
    img.SetSpacing((voxel_size,voxel_size,voxel_size))
    
    sitk.WriteImage(img,export_path + 'udf_reconstruction.mhd') 
    
def save_udf_gradient(data_tupel):
    print("SAVE UDF")
    logits, data, out_path, gradient = data_tupel
    dim = (np.round(len(logits)**(1/3)).astype(np.int),)*3
    voxel_grid_origin = [-1, -1, -1]
    voxel_size = 2.0 / (dim[0] - 1)

    #mesh = gen.mesh_from_logits(logits)

    path = os.path.normpath(data['path'][0])
    export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    #np.save(export_path + 'logits_reconstruction.npy',logits)
    numpy_3d_sdf_tensor = np.reshape(logits,dim)
    img = sitk.GetImageFromArray(numpy_3d_sdf_tensor)
    img.SetOrigin(voxel_grid_origin)
    img.SetSpacing((voxel_size,voxel_size,voxel_size))
    
    sitk.WriteImage(img,export_path + 'udf_reconstruction.mhd')
    
    np.save(export_path + 'gradient.npy',gradient)
    np.save(export_path + 'distances.npy',logits)

def create_udf(data_tupels):
    p = Pool(mp.cpu_count())
    p.map(save_udf, data_tupels)
    p.close()
    
def create_pc(data_tupels):
    #p = Pool(mp.cpu_count())
    p = Pool(1)
    p.map(save_pc, data_tupels)
    p.close()    