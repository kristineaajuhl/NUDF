# NUDF: NEURAL UNSIGNED DISTANCE FIELDS FOR HIGH RESOLUTION 3D MEDICAL IMAGE SEGMENTATION
by Kristine Sørensen, Ole de Backer, Klaus Kofoed, Oscar Camara and Rasmus Paulsen

Paper accepted for oral presentation at ISBI 2022 

[![reconstruction](results_v3_cropped.png)](results_v3_cropped.png)

This repository is currently under construction

# Data preparation
For each training example we need a set consisting of the input image (.npy) and a collection of point-distance samples (.npz).
An example of how we prepared the data used for the paper can be seen in 

```
/Preprocessing/create_data_from_roi.py
```

We use an automatic ROI-detection network - for more information on this see https://github.com/kristineaajuhl/LAAmeshing_and_SSM.

An example of an artificial training example can be seen in ```Data_example```.

# Training 
To initiate training the following command is run
```
python train.py -std_dev in1 out1 0.0 -dist 0.45 0.45 0.1 -res 64 -m ShapeNet64Vox -batch_size 5
```
```-std_dev``` indicates the distribution of the sampling points (in/out indicates shape diameter sampling and 0.0 indicates random sampling in the full image space). 

```-dist``` indicates the percentage of points to use from each sampling category. 

```-res``` is the input image resolution and should match the chosen model in ```-m``` which can either ShapeNet64Vox, ShapeNet128Vox, ShapeNet256Vox or ShapeNet512Vox.

```-batch_size``` is the batch size and is chosen as high as possible without exceeding the memory capacity of the GPU.

# Dense Point Cloud Prediction
To create a dense point cloud based on trained model the following command is run
```
python PC_generate.py -std_dev in1 out1 0.0 -dist 0.45 0.45 0.1 -res 64 -m ShapeNet64Vox -checkpoint 260 -batch_points 100000
```

```-checkpoint``` indicates which epoch during training that is used for prediction

```-batch_points``` decides how many points is predicted per batch. For fastest processing the number should be chosen as high as possible without exceeding the GPU memory capacity. 

The point coordinates are saved as a .txt file

# Meshing the point cloud
To convert the coordinates from the above .txt file into the predicted mesh, we use a combination of poisson resampling, poisson reconstruction and removal of mesh parts that are not supported by the point cloud. 

Code for doing this can be seen in 
```
test.py
```

The script also holds code for evaluating the predictions.

# Acknowledgements & Licence
Some scripts take base in code from "Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion" from Chibane et. al. https://github.com/jchibane/if-net




