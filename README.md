# NUDF: NEURAL UNSIGNED DISTANCE FIELDS FOR HIGH RESOLUTION 3D MEDICAL IMAGE SEGMENTATION
by Kristine Sørensen, Ole de Backer, Klaus Kofoed, Oscar Camara and Rasmus Paulsen

Paper accepted for oral presentation at ISBI 2022 

This repository is currently under construction

# Data preparation
For each training example we need a set consisting of the input image (.npy) and a collection of point-distance samples (.npz).
An example of how we prepared the data used for the paper can be seen in 

´´´
/Preprocessing/create_data_from_roi.py
´´´

We use an automatic ROI-detection network - for more information on this see https://github.com/kristineaajuhl/LAAmeshing_and_SSM.

# Training 

# Evaluation

# Acknowledgements
Some scripts take base in code from "Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion" from Chibane et. al. 


