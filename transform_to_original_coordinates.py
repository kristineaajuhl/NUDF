# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:18:25 2023

@author: kajul
"""

import SimpleITK as sitk
import os
import numpy as np 
import vtk 

#input_file = 'C:/Users/kajul/Desktop/Transform_debug/surf_cropped.vtk'
#norm_file = 'C:/Users/kajul/Desktop/Transform_debug/surf_cropped_transformed.vtk'

#vol_num = '15'
#input_file = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v64_mShapeNet64Vox/evaluation_486_@256/generation/prepared_data/KIDNEY_HEALTHY_0016_SERIES0028_volume_'+vol_num+'/surf_cropped.vtk'
#image_file = 'D:/DTUTeams/Kristine2023/kidney_ROI/ROI/img/KIDNEY_HEALTHY_0016_SERIES0028_volume_'+vol_num+'.nii'
#output_file = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v64_mShapeNet64Vox/evaluation_486_@256/generation/prepared_data/KIDNEY_HEALTHY_0016_SERIES0028_volume_'+vol_num+'/surf_cropped_transformed.vtk'

#input_file = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v64_mShapeNet64Vox/evaluation_486_@256/generation/prepared_data/CFA-PILOT_0008_SERIES00'+vol_num+'/surf_cropped.vtk'
#image_file = 'D:/DTUTeams/Kristine2023/CFA_pilot/2mm_ROI/ROI/img/CFA-PILOT_0008_SERIES00'+vol_num+'.nii'
#output_file = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v64_mShapeNet64Vox/evaluation_486_@256/generation/prepared_data/CFA-PILOT_0008_SERIES00'+vol_num+'/surf_cropped_transformed.vtk


base = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_948_@256/generation/prepared_data/'
filelist = os.listdir(base)

if not os.path.exists('D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_948_@256/predicted_meshes/'): 
    os.mkdir('D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_948_@256/predicted_meshes/')


for file in filelist: 
    input_file = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_948_@256/generation/prepared_data/'+file+'/surf_cropped.vtk'
    image_file = 'D:/DTUTeams/Kristine2023/CFA1/ROI/img/'+file+'.nii'
    output_file = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_948_@256/generation/prepared_data/'+file+'/surf_cropped_transformed.vtk'
    output_file2 = 'D:/DTUTeams/Kristine2023/NUDF-main/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_948_@256/predicted_meshes/'+file+'.vtk'


    # ROI image
    img = sitk.ReadImage(image_file)
    size = img.GetSize()
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    roi_physical_size = 140 #mm
    
    
    
    # Surface
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    surf = reader.GetOutput()  
    
    
    scale_factor = (-roi_physical_size/2,-roi_physical_size/2,roi_physical_size/2)
    t2 = vtk.vtkTransform()
    t2.Scale(scale_factor)
    
    scale_filter = vtk.vtkTransformFilter()
    scale_filter.SetInputData(surf)
    scale_filter.SetTransform(t2)
    scale_filter.Update()
    surf_scale = scale_filter.GetOutput()
    
    #Translate 
    #direction = np.array([img.GetDirection()[0],img.GetDirection()[4],img.GetDirection()[8]])
    #pd_direction = direction[::-1]
    #trans = (-direction - origin*scale_factor[0])*[-1,-1,1]#pd_direction
    trans = ((origin[0]+size[0]*spacing[0]/2),
             (origin[1]-size[1]*spacing[1]/2),
             (origin[2]+size[2]*spacing[2]/2))
    #trans = (46.86,15.14,-135.77)
    t1 = vtk.vtkTransform()
    t1.Translate(trans)
    
    print(trans)
    
    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetInputData(surf_scale)
    trans_filter.SetTransform(t1)
    trans_filter.Update()
    surf_transformed = trans_filter.GetOutput()
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(surf_transformed)
    writer.Write()
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file2)
    writer.SetInputData(surf_transformed)
    writer.Write()
