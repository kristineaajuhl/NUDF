# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:18:25 2023

@author: kajul
"""

import SimpleITK as sitk
import os
import numpy as np 
import vtk 

input_file = 'C:/Users/kajul/Desktop/Transform_debug/surf_cropped.vtk'
norm_file = 'C:/Users/kajul/Desktop/Transform_debug/surf_cropped_transformed.vtk'


# ROI image
img = sitk.ReadImage('C:/Users/kajul/Desktop/Transform_debug/KIDNEY_HEALTHY_0016_SERIES0028_volume_0.nii')
size = img.GetSize()
origin = img.GetOrigin()
spacing = img.GetSpacing()
direction = img.GetDirection()
roi_physical_size = 70 #mm



# Surface
reader = vtk.vtkPolyDataReader()
reader.SetFileName(input_file)
reader.Update()
surf = reader.GetOutput()  


scale_factor = (roi_physical_size,roi_physical_size,roi_physical_size)
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
writer.SetFileName(norm_file)
writer.SetInputData(surf_transformed)
writer.Write()