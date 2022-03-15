# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:16:02 2020

@author: kajul
"""

import os
import vtk
import plyfile
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from scipy.ndimage import map_coordinates
import imageio as io
import pymeshlab
import random
    
def shapediameter_sampling(args):
    norm_file = args[0]
    sample_sigma = args[1]
    oversample_param = args[2]
    output_dir = args[3]
    
    shape_diameter_file_out = os.path.split(norm_file)[0] + '/shape_diameter_out.vtk'
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(norm_file)
    reader.Update()
    surf_out = reader.GetOutput()   
   
    if not os.path.exists(shape_diameter_file_out):
        # Save temprary PLY file to use with meshlab
        temp_input_file = os.path.split(norm_file)[0] + '/temp_inputfile.ply'
        temp_output_file= os.path.split(norm_file)[0] + '/temp_outputfile.ply'
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(surf_out)
        writer.SetFileName(temp_input_file)
        writer.Write()        
        
        # Calculate shape diameter function
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_input_file)
        ms.shape_diameter_function(onprimitive='On Faces', coneangle=45)
        ms.save_current_mesh(temp_output_file, binary=False, save_vertex_color = False, save_face_color=False)
        
        # Save shape diameter information on polydata cell scalars
        plydata = plyfile.PlyData.read(temp_output_file)
        shape_diameter = plydata['face'].data['quality']
        
        surf_out.GetCellData().SetScalars(numpy_to_vtk(shape_diameter))
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(shape_diameter_file_out)
        writer.SetInputData(surf_out)
        writer.Write()
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
    else: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shape_diameter_file_out)
        reader.Update()
        surf_out = reader.GetOutput()
        
    # Do the same with flipped normals
    shape_diameter_file_in = os.path.split(norm_file)[0] + '/shape_diameter_in.vtk'
    reverseSense = vtk.vtkReverseSense()
    reverseSense.SetInputData(surf_out)
    reverseSense.ReverseNormalsOn()
    reverseSense.Update()
    surf_in = reverseSense.GetOutput()
    
    if not os.path.exists(shape_diameter_file_in):
        # Save temprary PLY file to use with meshlab
        temp_input_file = os.path.split(norm_file)[0] + '/temp_inputfile.ply'
        temp_output_file= os.path.split(norm_file)[0] + '/temp_outputfile.ply'
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(surf_in)
        writer.SetFileName(temp_input_file)
        writer.Write()        
        
        # Calculate shape diameter function
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_input_file)
        ms.shape_diameter_function(onprimitive='On Faces',coneangle=45)
        ms.save_current_mesh(temp_output_file, binary=False, save_vertex_color = False, save_face_color=False)
        
        # Save shape diameter information on polydata cell scalars
        plydata = plyfile.PlyData.read(temp_output_file)
        shape_diameter = plydata['face'].data['quality']
        
        surf_in.GetCellData().SetScalars(numpy_to_vtk(shape_diameter))
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(shape_diameter_file_in)
        writer.SetInputData(surf_in)
        writer.Write()
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
    else: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shape_diameter_file_in)
        reader.Update()
        surf_in = reader.GetOutput()
          
    # Sample points on the surface    
    resampler = vtk.vtkPolyDataPointSampler()
    #resampler.SetPointGenerationModeToRegular()
    #resampler.InterpolatePointDataOn()
    resampler.GenerateEdgePointsOff()
    resampler.GenerateVertexPointsOff()
    resampler.GenerateInteriorPointsOn()
    resampler.GenerateVerticesOff()
    resampler.SetDistance(0.002)
    resampler.SetInputData(surf_in)
    resampler.Update()
    
    points = vtk_to_numpy(resampler.GetOutput().GetPoints().GetData())
    print("Points inside/outside: ", points.shape[0])
    
    # For eaxh point get normal and the two shape diameters
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(surf_in)    
    cell_locator.BuildLocator()
    normals_in = surf_in.GetPointData().GetNormals()
    normals_out = surf_out.GetPointData().GetNormals()
    boundary_points_in = []
    boundary_points_out = []
    in_count, out_count = 0,0
    for point in points: 
        in_count += 1
        out_count += 1
        ### Interior points
        #Find cell
        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        cell_locator.FindClosestPoint(point, c, cellId, subId, d)
        triangle = surf_in.GetCell(cellId).GetPoints()
        
        # barycentric coordinate        
        bary = np.zeros(3)
        v2_1 = np.array(triangle.GetPoint(0))-np.array(triangle.GetPoint(1))
        v2_3 = np.array(triangle.GetPoint(2))-np.array(triangle.GetPoint(1))
        v2_t = point - np.array(triangle.GetPoint(1))
        d00 =  np.dot(v2_1,v2_1)       
        d01 = np.dot(v2_1,v2_3)
        d11 = np.dot(v2_3,v2_3)
        denom = d00 * d11 - d01 * d01
        
        d20 = np.dot(v2_t,v2_1)
        d21 = np.dot(v2_t,v2_3)
        bary[0] = (d11 * d20 - d01 * d21) / denom
        bary[1] = (d00 * d21 - d01 * d20) / denom
        bary[2] = 1.0 - bary[0] - bary[1]
        
        #Find normals
        p1_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(0)))
        p2_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(1)))
        p3_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(2)))
        normal_in = bary[0] * p1_normal + bary[2] * p2_normal + bary[1] * p3_normal
        p1_normal = np.array(normals_out.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(0)))
        p2_normal = np.array(normals_out.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(1)))
        p3_normal = np.array(normals_out.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(2)))
        normal_out = bary[0] * p1_normal + bary[2] * p2_normal + bary[1] * p3_normal
        
        # Find shape diameter
        sd_in = surf_in.GetCellData().GetScalars().GetTuple(cellId.get())[0]
        sd_out = surf_out.GetCellData().GetScalars().GetTuple(cellId.get())[0]
        if sd_in > sample_sigma: sd_in = sample_sigma
        if sd_out > sample_sigma: sd_out = sample_sigma
        
        #Create interior points
        d = np.abs(random.gauss(0,sd_in/2/2))
        if sd_in < 0.1 or in_count == oversample_param: 
            boundary_points_in.append(point - normal_in*d)
            in_count = 0
        #print(d)
        
        #Create exterior points
        d = np.abs(random.gauss(0,sd_out/2/2))
        if sd_out < 0.1 or out_count == oversample_param: 
            boundary_points_out.append(point - normal_out*d)
            out_count = 0
        #print(d)
        
    print("Final number of points: ",len(boundary_points_in))
    occupancies_in = point_to_surf_distance(np.array(boundary_points_in),surf_in)
    occupancies_out = point_to_surf_distance(np.array(boundary_points_out),surf_out)
    
    out_file_in = os.path.join(output_dir, 'boundary_in3_samples.npz')
    np.savez(out_file_in, points=boundary_points_in, occupancies = occupancies_in, grid_coords= boundary_points_in)
    
    out_file_out = os.path.join(output_dir, 'boundary_out3_samples.npz')
    np.savez(out_file_out, points=boundary_points_out, occupancies = occupancies_out, grid_coords= boundary_points_out)
    
def shapediameter_sampling_cell(args):
    norm_file = args[0]
    sample_sigma = args[1]
    n_samples = args[2]
    volatile_factor = args[3]
    output_dir = args[4]
    
    shape_diameter_file_out = os.path.split(norm_file)[0] + '/shape_diameter_out.vtk'
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(norm_file)
    reader.Update()
    surf_out = reader.GetOutput()   
   
    if not os.path.exists(shape_diameter_file_out):
        # Save temprary PLY file to use with meshlab
        temp_input_file = os.path.split(norm_file)[0] + '/temp_inputfile.ply'
        temp_output_file= os.path.split(norm_file)[0] + '/temp_outputfile.ply'
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(surf_out)
        writer.SetFileName(temp_input_file)
        writer.Write()        
        
        # Calculate shape diameter function
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_input_file)
        ms.shape_diameter_function(onprimitive='On Faces', coneangle=45)
        ms.save_current_mesh(temp_output_file, binary=False, save_vertex_color = False, save_face_color=False)
        
        # Save shape diameter information on polydata cell scalars
        plydata = plyfile.PlyData.read(temp_output_file)
        shape_diameter = plydata['face'].data['quality']
        
        surf_out.GetCellData().SetScalars(numpy_to_vtk(shape_diameter))
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(shape_diameter_file_out)
        writer.SetInputData(surf_out)
        writer.Write()
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
    else: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shape_diameter_file_out)
        reader.Update()
        surf_out = reader.GetOutput()
        
    # Do the same with flipped normals
    shape_diameter_file_in = os.path.split(norm_file)[0] + '/shape_diameter_in.vtk'
    reverseSense = vtk.vtkReverseSense()
    reverseSense.SetInputData(surf_out)
    reverseSense.ReverseNormalsOn()
    reverseSense.Update()
    surf_in = reverseSense.GetOutput()
    
    if not os.path.exists(shape_diameter_file_in):
        # Save temprary PLY file to use with meshlab
        temp_input_file = os.path.split(norm_file)[0] + '/temp_inputfile.ply'
        temp_output_file= os.path.split(norm_file)[0] + '/temp_outputfile.ply'
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(surf_in)
        writer.SetFileName(temp_input_file)
        writer.Write()        
        
        # Calculate shape diameter function
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_input_file)
        ms.shape_diameter_function(onprimitive='On Faces',coneangle=45)
        ms.save_current_mesh(temp_output_file, binary=False, save_vertex_color = False, save_face_color=False)
        
        # Save shape diameter information on polydata cell scalars
        plydata = plyfile.PlyData.read(temp_output_file)
        shape_diameter = plydata['face'].data['quality']
        
        surf_in.GetCellData().SetScalars(numpy_to_vtk(shape_diameter))
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(shape_diameter_file_in)
        writer.SetInputData(surf_in)
        writer.Write()
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
    else: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shape_diameter_file_in)
        reader.Update()
        surf_in = reader.GetOutput()
     
    boundary_points_in = []
    boundary_points_out = []
        
    # Sample "interior" points
    sd_in = vtk_to_numpy(surf_in.GetCellData().GetScalars())
    n_volatile = np.sum(sd_in<0.1)
    n_rest = sd_in.shape[0]-n_volatile
    samples_per_cell = n_samples/2/(n_volatile*volatile_factor+n_rest)
    print(samples_per_cell)
    #print("Number of points inside: ", samples_per_cell*n_rest + samples_per_cell*volatile_factor*n_volatile)
    if samples_per_cell < 1: 
        counter = np.round(1/samples_per_cell)
        count = 0
    else: 
        counter = 0
        count = 0
    
    normals_in = surf_in.GetPointData().GetNormals()
    for cellId in range(surf_in.GetNumberOfCells()):
        triangle = surf_in.GetCell(cellId).GetPoints()
        sd_in = surf_in.GetCellData().GetScalars().GetTuple(cellId)[0]
        
        p1_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(0)))
        p2_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(1)))
        p3_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(2)))
        
        if sd_in>0.1:
            if count == counter: 
                samples = np.ceil(samples_per_cell)
                count = 0
            else:
                count += 1
                continue
            
        else: 
            samples = np.round(samples_per_cell*volatile_factor)
        
        for i in range(int(samples)):
            r = [random.random() for i in range(0,3)]
            bary = [i/sum(r) for i in r]
            
            point_x = bary[0] * triangle.GetPoint(0)[0] + bary[1] * triangle.GetPoint(1)[0] + bary[2] * triangle.GetPoint(2)[0]
            point_y = bary[0] * triangle.GetPoint(0)[1] + bary[1] * triangle.GetPoint(1)[1] + bary[2] * triangle.GetPoint(2)[1]
            point_z = bary[0] * triangle.GetPoint(0)[2] + bary[1] * triangle.GetPoint(1)[2] + bary[2] * triangle.GetPoint(2)[2]
            point = [point_x, point_y, point_z]
            normal = bary[0] * p1_normal + bary[2] * p2_normal + bary[1] * p3_normal
            
            d = np.abs(random.gauss(0,sd_in/2/2))
            boundary_points_in.append(point - normal*d)
            
    print("Interior points: ", len(boundary_points_in))
    
    
    # Sample "exterior" points
    sd_out = vtk_to_numpy(surf_out.GetCellData().GetScalars())
    n_volatile = np.sum(sd_out<0.1)
    n_rest = sd_out.shape[0]-n_volatile
    samples_per_cell = n_samples/2/(n_volatile*volatile_factor+n_rest)
    #print("Number of points inside: ", samples_per_cell*n_rest + samples_per_cell*volatile_factor*n_volatile)   
    if samples_per_cell < 1: 
        counter = np.round(1/samples_per_cell)
        count = 0
    else: 
        counter = 0
        count = 0
    
    normals_out = surf_out.GetPointData().GetNormals()
    for cellId in range(surf_out.GetNumberOfCells()):
        triangle = surf_out.GetCell(cellId).GetPoints()
        
        # Interior
        sd_out = surf_out.GetCellData().GetScalars().GetTuple(cellId)[0]
        
        p1_normal = np.array(normals_out.GetTuple(surf_out.GetCell(cellId).GetPointIds().GetId(0)))
        p2_normal = np.array(normals_out.GetTuple(surf_out.GetCell(cellId).GetPointIds().GetId(1)))
        p3_normal = np.array(normals_out.GetTuple(surf_out.GetCell(cellId).GetPointIds().GetId(2)))
        
        if sd_out>0.1:
            if count == counter:
                samples = np.ceil(samples_per_cell)
                count = 0
            else:
                count += 1
                continue
            
        else: 
            samples = np.round(samples_per_cell*volatile_factor)
        
        for i in range(int(samples)):
            r = [random.random() for i in range(0,3)]
            bary = [i/sum(r) for i in r]
            
            point_x = bary[0] * triangle.GetPoint(0)[0] + bary[1] * triangle.GetPoint(1)[0] + bary[2] * triangle.GetPoint(2)[0]
            point_y = bary[0] * triangle.GetPoint(0)[1] + bary[1] * triangle.GetPoint(1)[1] + bary[2] * triangle.GetPoint(2)[1]
            point_z = bary[0] * triangle.GetPoint(0)[2] + bary[1] * triangle.GetPoint(1)[2] + bary[2] * triangle.GetPoint(2)[2]
            point = [point_x, point_y, point_z]
            normal = bary[0] * p1_normal + bary[2] * p2_normal + bary[1] * p3_normal
            
            d = np.abs(random.gauss(0,sd_out/2/2))
            boundary_points_out.append(point - normal*d)
    
    print("Exterior points: ", len(boundary_points_out))
        
    occupancies_in = point_to_surf_distance(np.array(boundary_points_in),surf_in)
    occupancies_out = point_to_surf_distance(np.array(boundary_points_out),surf_out)
    
    out_file_in = os.path.join(output_dir, 'boundary_in1_samples.npz')
    np.savez(out_file_in, points=boundary_points_in, occupancies = occupancies_in, grid_coords= boundary_points_in)
    
    out_file_out = os.path.join(output_dir, 'boundary_out1_samples.npz')
    np.savez(out_file_out, points=boundary_points_out, occupancies = occupancies_out, grid_coords= boundary_points_out)
    
def close_ostium_hole(meshin):
    """
    This function takes an open mesh and constructs triangles to close the hole. 
    
    Parameters
    ----------
    meshin : vtk polydata
        The open mesh that you wish to close

    Returns
    -------
    meshOut : vtk polydata
        The closed mesh

    """

    #detect holes: 
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(meshin)
    boundaryEdges.SetBoundaryEdges(1)
    boundaryEdges.SetFeatureEdges(0)
    boundaryEdges.SetNonManifoldEdges(0)
    boundaryEdges.SetManifoldEdges(0)
    boundaryEdges.Update()
    
    nombre = boundaryEdges.GetOutput().GetNumberOfLines()
    pointsNumber = nombre
    
    poly = vtk.vtkCleanPolyData()
    poly.SetInputData(meshin)
    poly.Update()
    
    region = vtk.vtkPolyDataConnectivityFilter()
    meshAppend = vtk.vtkAppendPolyData()
    bouchon = vtk.vtkStripper()
    bouchonPoly = vtk.vtkPolyData()
    bouchontri = vtk.vtkTriangleFilter()
    
    meshAppend.AddInputData(poly.GetOutput())
    meshAppend.AddInputData(bouchontri.GetOutput())
    
    region.SetInputData(boundaryEdges.GetOutput())
    region.SetExtractionMode(6)
    
    bouchon.SetInputData(region.GetOutput())
    bouchontri.SetInputData(bouchonPoly)
    
    poly.SetInputData(meshAppend.GetOutput())
    poly.SetTolerance(0.0)
    poly.SetConvertLinesToPoints(0)
    poly.SetConvertPolysToLines(0)
    poly.SetConvertStripsToPolys(0)
    
    boundaryEdges.SetInputData(poly.GetOutput())
    
    while nombre != 0:
        region.Update()
        
        #Creating polygonal patches
        bouchon.Update()
        
        bouchonPoly.Initialize()
        bouchonPoly.SetPoints(bouchon.GetOutput().GetPoints())    
        bouchonPoly.SetPolys(bouchon.GetOutput().GetLines())
        #bouchonPoly.Update()
        
        #triangulate polygonal patch
        bouchontri.Update()
    
        #patch
        meshAppend.Update()
        
        #Remove dublicated edges and points
        poly.Update()
    
        #Update the number of edges
        boundaryEdges.Update()
    
        nombre = ((boundaryEdges.GetOutput()).GetNumberOfLines())
    
    meshOut = vtk.vtkPolyData()
    meshOut.DeepCopy(poly.GetOutput())
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(meshOut)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.Update()
    polydata = normals.GetOutput()
    normalsVTK = polydata.GetPointData().GetArray("Normals")
    meshOut.GetPointData().SetNormals(normalsVTK)
    
    return meshOut

def interpolate_point(nib_img,point_coord):
    img = nib_img.get_fdata()
    
    lower_bounds = [float(nib_img.header["qoffset_x"]),float(nib_img.header["qoffset_y"]),float(nib_img.header["qoffset_z"])]
    spacing = nib_img.header["pixdim"][1]
    sdf_size = nib_img.header["dim"][1:4]
    upper_bounds = lower_bounds + sdf_size*spacing
    
    x = np.linspace(lower_bounds[0],upper_bounds[0],img.shape[0])
    y = np.linspace(lower_bounds[1],upper_bounds[1],img.shape[1])
    z = np.linspace(lower_bounds[2],upper_bounds[2],img.shape[2])
    
    return interp3(x, y, z, img, point_coord[0],point_coord[1],point_coord[2])
    
    
    
    
def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]
    
    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)

def point_to_surf_distance(points,pd):
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(pd)
    cellLocator.BuildLocator()    
    
    all_udf = np.zeros(points.shape[0]).astype(np.float32)
    for i, point in enumerate(points):
        cellId = vtk.reference(0)
        c = [0.0,0.0,0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        cellLocator.FindClosestPoint(point,c,cellId,subId,d)
        all_udf[i] = np.sqrt(d.get())
        
    return all_udf