# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:12:17 2021

@author: kajul
"""
import subprocess
import vtk
import multiprocessing
from tqdm import tqdm
import os
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import math

def mesh_UDF(args):
    input_file = args[0]
    output_file = args[1]
    dir_mrf = args[2]
    
    if os.path.exists(input_file):
        #if not os.path.exists(output_file):
        quiet = True
        if quiet:
            output_pipe = open(os.devnull, 'w')       # Ignore text output from MRF.exe.
        else:
            output_pipe = None
            
        subprocess.call([dir_mrf, '-i', input_file, '-o', output_file, '-u', '-I', '0.02'], stdout=output_pipe)

def rotate_vtk(args):
    input_file = args[0]
    output_file = args[1]
    
    if os.path.exists(input_file):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_file)
        reader.Update()
        pred = reader.GetOutput()
        
        transFilter = vtk.vtkTransformPolyDataFilter()
        transform = vtk.vtkTransform()
        transform.Scale(1,1,-1)
        #transform.RotateX(-90)
        transform.RotateY(90)
        transFilter.SetInputData(pred)
        transFilter.SetTransform(transform)
        transFilter.Update()
        
        pred_flip = transFilter.GetOutput()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pred_flip)
        writer.SetFileName(output_file)
        writer.Write()
        
def convert_to_org_coords(args):
    in_file = args[0]
    out_file = args[1]
    transform_file = args[2]
    fileid = os.path.split(os.path.split(in_file)[0])[-1]
    
    with open(transform_file, "r") as fp:
        for line in fp: 
            fields = line.split('\t')
            if fields[0] == fileid: 
                scale_factor = [1/float(fields[1]),]*3
                com = np.array([float(fields[2]),float(fields[3]),float(fields[4])])
                
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(in_file)
    reader.Update()
    pd_in = reader.GetOutput()
    
    t1 = vtk.vtkTransform()
    t1.Translate(com)
    t1.Scale(scale_factor)
        
    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetInputData(pd_in)
    trans_filter.SetTransform(t1)
    trans_filter.Update()
    pd_out = trans_filter.GetOutput()
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(pd_out)
    writer.SetFileName(out_file)
    writer.Write()
        
def eval_reconstruction(args):
    pred_file = args[0]
    true_file = args[1]
    eval_file = args[2]
    sample_points = args[3]
    scale_factor = args[4]
    filename = os.path.split(os.path.split(pred_file)[0])[-1]
    dataset = os.path.split(os.path.split(os.path.split(pred_file)[0])[0])[-1]
    
    print(pred_file)
    
    if os.path.exists(pred_file):
        if pred_file[-4::] == '.ply':
            print('ply')
            reader1 = vtk.vtkPLYReader()
            reader1.SetFileName(pred_file)
            reader1.Update()
            pred = reader1.GetOutput()
        else: 
            reader1 = vtk.vtkPolyDataReader()
            reader1.SetFileName(pred_file)
            reader1.Update()
            pred = reader1.GetOutput()
        
        print(true_file)
        reader2 = vtk.vtkPolyDataReader()
        reader2.SetFileName(true_file)
        reader2.Update()
        true = reader2.GetOutput()
        
        # Chamfer distance:
        distanceFilter = vtk.vtkDistancePolyDataFilter()
        distanceFilter.SetInputData(0,pred)
        distanceFilter.SetInputData(1,true)
        distanceFilter.ComputeSecondDistanceOn()
        distanceFilter.SignedDistanceOff()
        distanceFilter.Update()
        
        try: 
            vtk_d = distanceFilter.GetOutput().GetPointData().GetScalars()
            if sample_points != 0: 
                np_d = np.random.choice(vtk_to_numpy(vtk_d),sample_points,replace=True)
            else: 
                np_d = vtk_to_numpy(vtk_d)
            mean_chamfer = np.mean(np_d)/scale_factor
            median_chamfer = np.median(np_d)/scale_factor
        except: 
            print("Something is wrong with the distance filter...")
            return
            
        #MeshAccuracy
        PREDpoints0 = vtk_to_numpy(pred.GetPoints().GetData())
        if sample_points != 0: 
            chosen_points = np.random.choice(np.arange(PREDpoints0.shape[0]),sample_points,replace=True)
            PREDpoints = PREDpoints0[chosen_points,:]
        else: 
            PREDpoints = PREDpoints0
        
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(true)
        cellLocator.BuildLocator()
        
        dist = np.zeros(PREDpoints.shape[0])
        for idx in range(PREDpoints.shape[0]):
            testPoint = PREDpoints[idx]
            
            #Find the closest points to TestPoint
            cellId = vtk.reference(0)
            c = [0.0, 0.0, 0.0]
            subId = vtk.reference(0)
            d = vtk.reference(0.0)
            cellLocator.FindClosestPoint(testPoint, c, cellId, subId, d)
            
            dist[idx] = d
            
        meshAcc = math.sqrt(np.percentile(dist,90))/scale_factor

        # Mesh completion: 
        delta = 2*scale_factor #2mm
            
        GTpoints = vtk_to_numpy(true.GetPoints().GetData())
        
        ## Create the tree
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(pred)
        cellLocator.BuildLocator()
        
        dist = np.zeros(GTpoints.shape[0])
        for idx in range(GTpoints.shape[0]):
            testPoint = GTpoints[idx]
            
            #Find the closest points to TestPoint
            cellId = vtk.reference(0)
            c = [0.0, 0.0, 0.0]
            subId = vtk.reference(0)
            d = vtk.reference(0.0)
            cellLocator.FindClosestPoint(testPoint, c, cellId, subId, d)

            # Rasmus 9/4-2020...do remember that d is the squared distance.
            dist[idx] = math.sqrt(d)
        
        MeshCompl = np.sum(dist<delta)/dist.shape[0]
        
        with open(eval_file,'a+') as f: 
            f.write(dataset + "\t")
            f.write(filename+ "\t")
            f.write(str(mean_chamfer) + "\t")
            f.write(str(median_chamfer) + "\t")
            f.write(str(meshAcc) + "\t")
            f.write(str(MeshCompl) + "\t")
            f.write("\n")
        
        return
        
def evalLAA_reconstruction(args):
    pred_file = args[0]
    true_file = args[1]
    eval_file = args[2]
    filename = os.path.split(os.path.split(pred_file)[0])[-1]
    
    if os.path.exists(pred_file):
        reader1 = vtk.vtkPolyDataReader()
        reader1.SetFileName(pred_file)
        reader1.Update()
        pred = reader1.GetOutput()
        
        reader2 = vtk.vtkPolyDataReader()
        reader2.SetFileName(true_file)
        reader2.Update()
        true = reader2.GetOutput()
        
        # Chamfer distance:
        distanceFilter = vtk.vtkDistancePolyDataFilter()
        distanceFilter.SetInputData(0,true)
        distanceFilter.SetInputData(1,pred)
        distanceFilter.ComputeSecondDistanceOff()
        distanceFilter.SignedDistanceOff()
        distanceFilter.Update()
        
        try: 
            vtk_d = distanceFilter.GetOutput().GetPointData().GetScalars()
            mean_chamfer = np.mean(vtk_to_numpy(vtk_d))
            median_chamfer = np.median(vtk_to_numpy(vtk_d))
        except: 
            print("Something is wrong with the distance filter...")
            return
            
        #MeshAccuracy
        # PREDpoints = vtk_to_numpy(pred.GetPoints().GetData())
        
        # cellLocator = vtk.vtkCellLocator()
        # cellLocator.SetDataSet(true)
        # cellLocator.BuildLocator()
        
        # dist = np.zeros(PREDpoints.shape[0])
        # for idx in range(PREDpoints.shape[0]):
        #     testPoint = PREDpoints[idx]
            
        #     #Find the closest points to TestPoint
        #     cellId = vtk.reference(0)
        #     c = [0.0, 0.0, 0.0]
        #     subId = vtk.reference(0)
        #     d = vtk.reference(0.0)
        #     cellLocator.FindClosestPoint(testPoint, c, cellId, subId, d)
            
        #     dist[idx] = d
            
        # meshAcc = math.sqrt(np.percentile(dist,90))/0.019758613037983232

        # Mesh completion: 
        delta = 2 #0.039517226075966465 #2mm
            
        GTpoints = vtk_to_numpy(true.GetPoints().GetData())
        
        ## Create the tree
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(pred)
        cellLocator.BuildLocator()
        
        dist = np.zeros(GTpoints.shape[0])
        for idx in range(GTpoints.shape[0]):
            testPoint = GTpoints[idx]
            
            #Find the closest points to TestPoint
            cellId = vtk.reference(0)
            c = [0.0, 0.0, 0.0]
            subId = vtk.reference(0)
            d = vtk.reference(0.0)
            cellLocator.FindClosestPoint(testPoint, c, cellId, subId, d)

            # Rasmus 9/4-2020...do remember that d is the squared distance.
            dist[idx] = math.sqrt(d)
        
        MeshCompl = np.sum(dist<delta)/dist.shape[0]
        
        with open(eval_file,'a+') as f: 
            f.write(filename+ "\t")
            f.write(str(mean_chamfer) + "\t")
            f.write(str(median_chamfer) + "\t")
            #f.write(str(meshAcc) + "\t")
            f.write(str(MeshCompl) + "\t")
            f.write("\n")
        
        return
    

def imap_unordered_bar(func, args, n_processes = 2):
    p = multiprocessing.Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list