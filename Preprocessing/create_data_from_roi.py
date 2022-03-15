# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:28:41 2021

@author: kajul
"""

#Overview of project
## 1) Predict low-res label with model

## 2) Crop image according to label-COM and save as .nii (heartROI api) in 64/128 resolution

## 3) Convert cropped .nii file to .npy for IF-net input

## 4) Translate surface to make sure it fits with the cropped image

## 5) Scale and sample the surface (main_v2.py)

import os
import vtk
import numpy as np
import nibabel
import subprocess
import methods
import SimpleITK as sitk

res = 64
roi_physical_size = 70 #mm
predict_and_crop_image = True
save_input_img = True
prepare_surface = True
sample_uniform_points = False
sample_UDF_points_shapefunction = True




## -------------  Setting paths ------------------
filelist = 'E:/DATA/TEST/filelist.txt'

config_roi = 'H:/LAAmeshing_and_SSM/SDFregression/configs/configROI.json'
roi_code_base = 'H:/LAAmeshing_and_SSM/SDFregression/'

output_base = 'E:/DATA/IFNET/LAA/prepared_data/'
preproc_root = 'E:/DATA/IFNET/LAA/'
transform_file = 'E:/DATA/IFNET/LAA/transform_file.txt'

## -------------  Loading filelist -------------------
filenames = []
f =  open(filelist,"r")
for x in f:
    filenames.append(x.strip()[0:4])
    
print("Number of files to process: ",len(filenames))
    
    
## --------             Predict and crop all images            ------- ##
# Predict ROI and crop image (Resolution and output-directory are set in configROI.json)
if predict_and_crop_image: 
    subprocess.call(['python', roi_code_base+'predictROI.py', '--c', config_roi, '--n', filelist])


## -----       Convert images and create samples for all images   ---- ##
for fileid in filenames:
    print("Processing file: ",fileid)
    output_dir = os.path.join(output_base, fileid)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    input_file = os.path.join(preproc_root, 'mesh/'+fileid+'.vtk' )
    norm_file = os.path.join(output_dir, 'isosurf_scaled.vtk')
    
    if save_input_img:
        image_filename = os.path.join(preproc_root, 'ROI/img/' + fileid + ".nii")  
        filename = os.path.join(output_dir, 'imageinput_{}.npy'.format(res))
        img = sitk.ReadImage(image_filename)
        origin = np.array(img.GetOrigin())
        
        occ_np = sitk.GetArrayViewFromImage(img)
        occ_np = np.flip(occ_np,2) 
        
        np.save(filename, occ_np)
        
    if prepare_surface: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_file)
        reader.Update()
        surf = reader.GetOutput()  
        
        scale_factor = ((2/roi_physical_size),)*3
        t2 = vtk.vtkTransform()
        t2.Scale(scale_factor)
        
        scale_filter = vtk.vtkTransformFilter()
        scale_filter.SetInputData(surf)
        scale_filter.SetTransform(t2)
        scale_filter.Update()
        surf_scale = scale_filter.GetOutput()
        
        #Translate to fit within [-1,1]
        direction = np.array([img.GetDirection()[0],img.GetDirection()[4],img.GetDirection()[8]])
        pd_direction = direction[::-1]
        trans = (-direction - origin*scale_factor[0])*[-1,-1,1]#pd_direction
        t1 = vtk.vtkTransform()
        t1.Translate(trans)
        
        trans_filter = vtk.vtkTransformFilter()
        trans_filter.SetInputData(surf_scale)
        trans_filter.SetTransform(t1)
        trans_filter.Update()
        surf_transformed = trans_filter.GetOutput()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(norm_file)
        writer.SetInputData(surf_transformed)
        writer.Write()
        
    if sample_uniform_points: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(norm_file)
        reader.Update()
        surf = reader.GetOutput()   
        
        ### sample_uniform_points
        uni_size = 10000
        boundary_points = np.vstack((np.random.uniform(-1,1,uni_size),np.random.uniform(-1,1,uni_size),np.random.uniform(-1,1,uni_size))).T
    
        occupancies = methods.point_to_surf_distance(boundary_points,surf)
        
        # Find point coordinates (in the range [-1;1] with relation to the voxelization)
        #boundary_points[:,[2,0]] = boundary_points[:,[0,2]]
        grid_coords = boundary_points.copy()
        #origin = [0.5,0.5,0.5]#[float(nib_img.header["qoffset_x"]),float(nib_img.header["qoffset_y"]),float(nib_img.header["qoffset_z"])] 
        origin =  [0,0,0]
        for i, point in enumerate(grid_coords):
            grid_coords[i] = (point + origin)#*res
        
        out_file2 = os.path.join(output_dir, 'boundary_{}_samples.npz'.format(0.0))
        np.savez(out_file2, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        
    if sample_UDF_points_shapefunction: 
        #methods.shapediameter_sampling([norm_file,0.5,100,output_dir])
        methods.shapediameter_sampling_cell([norm_file,0.5,100000,500,output_dir])

        