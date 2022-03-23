#import data_processing.implicit_waterproofing as iw
import torch
from torch.nn import functional as F
import os
from glob import glob
import numpy as np

class Generator(object):
    def __init__(self, model, threshold, exp_name, checkpoint = None, device = torch.device("cuda"), resolution = 16, batch_points = 1000000):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.resolution = resolution
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format( exp_name)
        #self.checkpoint_path = 'H:/if-net-master/experiments/{}/checkpoints/'.format( exp_name)
        self.load_checkpoint(checkpoint)
        self.batch_points = batch_points

        self.min = -1
        self.max = 1
        
        ## START FROM SPARSE BBOX
        grid_coords = torch.rand((batch_points,3))*2-1
        grid_coords = grid_coords.unsqueeze(0).to(self.device, dtype=torch.float)
        #self.init_points = grid_coords
        
        ## FROM UDF GRID GENERATOR
        # x_ = np.linspace(0,resolution-1,resolution)
        # y_ = np.linspace(0,resolution-1,resolution)
        # z_ = np.linspace(0,resolution-1,resolution)
        
        #x_ = np.linspace(self.min,self.max,resolution)
        #y_ = np.linspace(self.min,self.max,resolution)
        #z_ = np.linspace(self.min,self.max,resolution)
        
        #x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        
        #grid_points = np.vstack((np.reshape(x,-1),np.reshape(y,-1),np.reshape(z,-1))).T

        #grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        #grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        #a = self.max + self.min
        #b = self.max - self.min

        #grid_coords = 2 * grid_points - a
        #grid_coords = grid_coords / b

        #grid_coords = torch.from_numpy(grid_points).to(self.device, dtype=torch.float)
        #grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)

    def generate_pc(self,data):
        inputs = data['inputs'].to(self.device)
        num_steps = 5
        for i,points in enumerate(self.grid_points_split): 
            for iteration in range(num_steps): 
                points_grad = points.clone().detach().requires_grad_(True) #torch.tensor(points,requires_grad=True)
                d2 = self.model(points_grad,inputs)
                d2.sum().backward()
                #d2 = dist.clone().detach()

                gradient = points_grad.grad.detach()
                norm_grad = F.normalize(gradient,dim=2)
                
                points2 = points[d2<0.1] - torch.mul(d2[d2<0.1],norm_grad[d2<0.1].T).T
                del points
                points = points2.unsqueeze(0)
                del points2
        
        n_points = 1e6
        print("N init: ", points.shape[1])
        batches = np.ceil(n_points/self.batch_points).astype(int)
        if batches == 0:
            batches = 1
        print("Batches: ", batches)
        
        final_points = []
        for b in range(batches):
            #print(points.shape[1])
            indices = torch.randint(0,points.shape[1],(int(n_points/batches),))
            #print(indices.shape)
            samples_b = points[0,indices,:]
            #print(samples_b.shape)
            points = samples_b + (0.1 / 3) * torch.randn(samples_b.shape).to(self.device).unsqueeze(0)  # 3 sigma rule
            del indices, samples_b
            #print(points.shape)
            
            for iteration in range(num_steps):
                points_grad = points.clone().detach().requires_grad_(True) #torch.tensor(points,requires_grad=True)
                d2 = self.model(points_grad,inputs)
                d2.sum().backward()
                #d2 = dist.clone().detach()
                
                gradient = points_grad.grad.detach()
                norm_grad = F.normalize(gradient,dim=2)
                    
                points = points[d2<0.1] - torch.mul(d2[d2<0.1],norm_grad[d2<0.1].T).T
                
                points = points.unsqueeze(0)
                
                
            surf_points = points[d2[d2<0.1].unsqueeze(0)<0.05].squeeze(0).detach().cpu().numpy()
            surf_points = surf_points[np.logical_and(np.all(surf_points > -1,axis=1),np.all(surf_points < 1,axis=1))]
            
            
            if len(final_points) < 1: 
                final_points = surf_points
            else: 
                final_points = np.vstack((final_points,surf_points))
        
        #print(np.min(surf_points,axis=0))
        #print(np.max(surf_points,axis=0))
        print("N final: ",final_points.shape)
        return final_points
            

    def generate_udf(self,data):
        inputs = data['inputs'].to(self.device)

        logits_list = []
        gradient_list = []
        for points in self.grid_points_split:
            #with torch.no_grad():
            points_grad = torch.tensor(points,requires_grad=True)
            logits = self.model(points_grad,inputs)
                
            logits.sum().backward()
            gradient = points_grad.grad.detach()
                
            logits_list.append(logits.squeeze(0).detach().cpu())
            #print(gradient.shape)
            gradient_list.append(gradient.squeeze(0).cpu())

        logits = torch.cat(logits_list, dim=0)
        gradients = torch.cat(gradient_list,dim=0)

        return logits.numpy(), gradients.numpy()
    
    # def generate_mesh(self, data):


    #     inputs = data['inputs'].to(self.device)


    #     logits_list = []
    #     for points in self.grid_points_split:
    #         with torch.no_grad():
    #             logits = self.model(points,inputs)
    #         logits_list.append(logits.squeeze(0).detach().cpu())

    #     logits = torch.cat(logits_list, dim=0)

    #     return logits.numpy()
    #     logits = np.reshape(logits.numpy(), (self.resolution,)*3)
        
    #     #padding to be able to retrieve object close to bounding box bondary
    #     logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
    #     threshold = np.log(self.threshold) - np.log(1. - self.threshold)
    #     vertices, triangles = mcubes.marching_cubes(
    #         logits, threshold)

    #     #remove translation due to padding
    #     vertices -= 1

    #     #rescale to original scale
    #     step = (self.max - self.min) / (self.resolution - 1)
    #     vertices = np.multiply(vertices, step)
    #     vertices += [self.min, self.min, self.min]

    #     mesh = trimesh.Trimesh(vertices, triangles)
    #     return mesh

    # def mesh_from_logits(self, logits):
    #     logits = np.reshape(logits, (self.resolution,) * 3)

    #     # padding to ba able to retrieve object close to bounding box bondary
    #     logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
    #     threshold = np.log(self.threshold) - np.log(1. - self.threshold)
    #     vertices, triangles = mcubes.marching_cubes(
    #         logits, threshold)

    #     # remove translation due to padding
    #     vertices -= 1

    #     # rescale to original scale
    #     step = (self.max - self.min) / (self.resolution - 1)
    #     vertices = np.multiply(vertices, step)
    #     vertices += [self.min, self.min, self.min]

    #     return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path+'/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])