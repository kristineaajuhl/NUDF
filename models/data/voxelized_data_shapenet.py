from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from scipy.ndimage import rotate, shift
from scipy.spatial.transform import Rotation as R



class VoxelizedDataset(Dataset):


    #def __init__(self, mode, res = 32,  input_type="voxel", pointcloud_samples = 3000, data_path = '/scratch/kajul/shapenet/data/', split_file = '/scratch/kajul/shapenet/full_split.npz',
    #             batch_size = 64, num_sample_points = 1024, num_workers = 12, sample_distribution = [1], sample_sigmas = [0.015], **kwargs):
    def __init__(self, mode, res = 32,  input_type="voxel", pointcloud_samples = 3000, data_path = 'D:/data/IMM/Kristine/prepared_data/', split_file = 'H:/if-net-master/shapenet/full_split.npz',
                 batch_size = 64, num_sample_points = 1024, num_workers = 1, sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.mode = mode
        self.data = self.split[mode]
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        #self.voxelized_pointcloud = voxelized_pointcloud
        self.input_type = input_type
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]

        if self.input_type == "voxel":
            occupancies = np.load(path + '/voxelization_{}.npy'.format(self.res))
            occupancies = np.unpackbits(occupancies)
            input = np.reshape(occupancies, (self.res,)*3)       
        elif self.input_type == "image":
            img = np.load(path + '/imageinput_{}.npy'.format(self.res))
            input = np.clip(img,-1000,1000)/1000 #Heart normalization
            #input = (np.clip(img,1050,1450)-1250)/200 #Soft tissue normalization
            #input = (np.clip(img,0,2000)-1000)/1000 #DEBUG
            
            # img = np.float32(img)
            # mean_x = np.mean(img)
            # std_x = np.std(img)
            # input = (img - mean_x)/std_x
            #print(np.min(input),np.max(input))

        else:
            voxel_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
            occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            input = np.reshape(occupancies, (self.res,)*3)
          
        if self.mode == 'test':
            points = []
            coords = []
            occupancies = []
            return {'grid_coords':np.array(coords, dtype=np.float32),'occupancies': np.array(occupancies, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}
        
        else: 
            points = []
            coords = []
            occupancies = []

            for i, num in enumerate(self.num_samples):
                boundary_samples_path = path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_coords = boundary_samples_npz['grid_coords']
                boundary_sample_occupancies = boundary_samples_npz['occupancies']
                subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
                points.extend(boundary_sample_points[subsample_indices])
                coords.extend(boundary_sample_coords[subsample_indices])
                occupancies.extend(boundary_sample_occupancies[subsample_indices])

            assert len(points) == self.num_sample_points
            assert len(occupancies) == self.num_sample_points
            assert len(coords) == self.num_sample_points

            # augmentation: TODO Make sure it is only done at training time
            if self.mode == 'train':
                coords, points, input = self.data_augmentation(coords,input,p=0.7,r_range=[-10,10],t_range=[20,20],erasing=True,g_std=0.05)

            return {'grid_coords':np.array(coords, dtype=np.float32),'occupancies': np.array(occupancies, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

    def get_loader(self, shuffle =True):
        # X = torch.utils.data.DataLoader(
        #         self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
        #         worker_init_fn=self.worker_init_fn)
        X = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)
        return X

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
        
    def data_augmentation(self, bsc, image, p = 0, r_range = [0,0],t_range=[0,0],erasing=False,g_std=0):
        '''Data augmentation of image and point coordinates
        Input:  bsc:        Coordinates to be augmented
                image:      Image to be augmented
                p:          Probability of augmentation
                r_range:    Rotation range in degress
                t_range:    Translation range in pixels
                erasing:    Random erasing (not implemented yet!!)
                g_std:      Standard deviation of gaussian noise on image
        
        Return: bsc:        Augmented point coordinates
                bsp:        Augmented points (same as coordinates)
                image:      Augmented image
        
        '''
        if np.random.random() < p:
            # translation
            if any(t_range)!=0 and np.random.random() < p: 
                t_img = np.random.uniform(t_range[0],t_range[1],3)   
                t_points = t_img[[2,1,0]]
                image = shift(image,t_img,cval=-1)
                bsc = bsc+2*t_points/image.shape
            
            # rotation        
            if any(r_range)!=0 and np.random.random() < p:            
                angles = np.random.uniform(r_range[0],r_range[1],3)
                directions = [[1,2],[0,2],[0,1]]
                sign = [1,-1,1]
                
                for i,angle in enumerate(angles):    
                    rotation = np.zeros(3)
                    rotation[i] = sign[i]*angle
                    rot = R.from_euler('zyx',rotation,degrees=True).as_matrix()
                    image= rotate(image,angle,axes=directions[i],cval = -1,reshape=False)
                    bsc = np.matmul(bsc,rot)
            
            # random erasing
            #TODO: Investigate and implement if relevant                
                    
            # Gaussian noise
            if np.random.random() < p:
                image += np.random.normal(0,g_std,image.shape)
                    
        bsp = bsc.copy()
        return bsc,bsp,image
                
            
