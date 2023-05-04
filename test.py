# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:06:29 2021

@author: kajul
"""

import pymeshlab
import glob
import os
from shutil import copyfile
import UDFmesh_methods as Methods
import vtk
import numpy as np

if __name__ == '__main__': 
    mesh_point_cloud = True
    evaluate_mesh = True
    evaluate_mesh_SDFpred = False
    epoch = 486
    res = 64
    
    #base = 'H:/if-net-master/experiments/FINAL_'+str(res)+'/evaluation_'+str(epoch)+'_@256/generation/data/'
    base = 'H:/if-net-master/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v128_mShapeNet128Vox/evaluation_158_@256/generation/prepared_data/'
    #base = 'H:/if-net-master/experiments/iimage_dist-0.45_0.45_0.1_sigmas-in1_out1_0.0_v64_mShapeNet64Vox/evaluation_486_@256/generation/prepared_data/'
    true = 'E:/DATA/IFNET/LAA/prepared_data/'
    filelist = os.listdir(base)
    eval_file = os.path.join(base,'evaluation.txt')
    
    if mesh_point_cloud: 
        print("Processing number of files: ", len(filelist))
        for fileid in filelist: 
            print(fileid)
            filename1 = base + fileid + "/" + 'surf_points.txt'
            filename2 = base + fileid + "/" + 'surf_points.xyz'
            output_file = base + fileid + "/" + 'surf.ply'
            output_file2 = base + fileid + "/" + 'subsample.xyz'
            output_file3 = base + fileid + "/" + 'surf_cropped.vtk'
            
            # if os.path.exists(output_file):
            #     continue
            
            #if not os.path.exists(filename2):
            copyfile(filename1,filename2)
        
           # Load pointcloud
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename2)
            print(ms.number_meshes())
        
            # Subsample pointcloud
            ms.generate_sampling_poisson_disk(samplenum = 10000, subsample = True)
            ms.set_current_mesh(1)
            ms.save_current_mesh(output_file2)
            #print(ms.number_meshes())
            
            # Compute Normals
            ms.compute_normal_for_point_clouds()
            
            # Reconstruct surface
            ms.generate_surface_reconstruction_screened_poisson()
            #ms.surface_reconstruction_ball_pivoting(ballradius=0.01)
            starting_point = ms.number_meshes()
            
            #Split in connected components and choose largest
            ms.generate_splitting_by_connected_components()    
            max_vertex_number = 0
            largest_mesh = 0
            
            for i in range(starting_point,ms.number_meshes()):
                if ms.mesh(i).vertex_number() > max_vertex_number: 
                    max_vertex_number = ms.mesh(i).vertex_number()
                    largest_mesh = i
            
            ms.set_current_mesh(largest_mesh)
            m = ms.mesh(largest_mesh)
            print("Input mesh has", m.vertex_number(), 'vertex and', m.face_number(), 'faces' )
        
            # Close small holes
            #ms.close_holes(maxholesize=50,newfaceselected=False)
            
            # Remove parts with large faces
            #ms.select_faces_with_edges_longer_than(threshold=0.02)
            #ms.delete_selected_faces_and_vertices()
            #ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=100)
            
            # Save mesh
            #m = ms.mesh(ms.number_meshes()-1)
            #print("Final mesh has", m.vertex_number(), 'vertex and', m.face_number(), 'faces' )
            ms.save_current_mesh(output_file, binary=False)
            
            del ms
                        
            # Remove unsopported parts
            reader = vtk.vtkPLYReader()
            reader.SetFileName(output_file)
            reader.Update()
            pd = reader.GetOutput()
            
            p = np.loadtxt(filename1)
            points = vtk.vtkPoints()
            for c in p: 
                points.InsertNextPoint(c)
            kDTree = vtk.vtkKdTree()
            kDTree.BuildLocatorFromPoints(points)
            
            remove_ids = vtk.vtkIdTypeArray()
            for i in range(pd.GetNumberOfCells()):
                triangle = pd.GetCell(i).GetPoints()
                com = [0.0,0.0,0.0]
                pd.GetCell(i).TriangleCenter(triangle.GetPoint(0),triangle.GetPoint(1),triangle.GetPoint(2),com)

                neighborIds = vtk.vtkIdList()
                kDTree.FindPointsWithinRadius(0.01, com, neighborIds)
                
                if not neighborIds.GetNumberOfIds()>1: 
                    remove_ids.InsertNextValue(i)
                    pd.DeleteCell(i)
            
            pd.RemoveDeletedCells()
            print("Number of cells deleted: ", remove_ids.GetNumberOfValues())
            
            filler = vtk.vtkFillHolesFilter()
            filler.SetInputData(pd)
            filler.SetHoleSize(0.1)
            filler.Update()
            
            connFilter = vtk.vtkPolyDataConnectivityFilter()
            connFilter.SetInputData(filler.GetOutput())
            connFilter.SetExtractionModeToLargestRegion()
            connFilter.Update()
                    
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(connFilter.GetOutput())
            writer.SetFileName(output_file3)
            writer.Write()
                    
            # selectionNode = vtk.vtkSelectionNode()
            # selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
            # selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
            # selectionNode.SetSelectionList(remove_ids)
        
            # selection = vtk.vtkSelection()
            # selection.AddNode(selectionNode)
            
            # extractSelection = vtk.vtkExtractSelection()
            # extractSelection.SetInputData(0, pd)
            # extractSelection.SetInputData(1, selection)
            # extractSelection.Update()
            
            # selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
            # extractSelection.Update()   
            # extractSelection.GetOutput()

            # keep = 
    
    if evaluate_mesh:    
        final_list = []
        for fileid in filelist: 
            #final_list.append(os.path.join(base,fileid + "/surf.ply"))
            final_list.append(os.path.join(base,fileid + "/surf_cropped.vtk"))
            
        true_list = []
        for fileid in filelist: 
            true_list.append(os.path.join(true, fileid + '/isosurf_scaled.vtk'))
            
        n_names = len(filelist)
        sample_points = 0
        n_threads = 12
        scale_factor = 0.02857142857142857#0.009105477855477856
        
        print("Number of predictions: ",n_names)
        
        evalArgs = list(zip(final_list, true_list, [eval_file]*n_names,[sample_points]*n_names,[scale_factor]*n_names))
        Methods.imap_unordered_bar(Methods.eval_reconstruction, evalArgs, n_threads)

    if evaluate_mesh_SDFpred:
        #base = 'E:/DATA/TEST/Predictions/surf_model/'
        base = 'H:/surf/'
        true = 'E:/DATA/LAA100_Annotations/LAA_mesh/'
        filelist = os.listdir(base)
        eval_file = os.path.join(base,'evaluation.txt')
        
        final_list = []
        for fileid in filelist: 
            final_list.append(os.path.join(base,fileid ))
            
        true_list = []
        for fileid in filelist: 
            true_list.append(os.path.join(true, fileid[:4] + '.vtk'))
            
        n_names = len(filelist)
        sample_points = 0
        n_threads = 1
        scale_factor = 1
        
        evalArgs = list(zip(final_list, true_list, [eval_file]*n_names,[sample_points]*n_names,[scale_factor]*n_names))
        Methods.imap_unordered_bar(Methods.eval_reconstruction, evalArgs, n_threads)


