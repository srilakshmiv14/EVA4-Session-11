# EVA4-Session-11

1.	Importing all the packages.
2.	Modules are stored in a library which is available in Session 8 repository folder named Library.

For Session 11 the new modules are Data Transforms which includes some features of albumentataions Transforms and New Resnet architecture as per mentioned in assignment Question. (A11.py)

Modules are 1. Loading Data 2. Transforming the data (including albumentation transforms) 3. Building the Basic Model 4. New Model 5. Including LR finder to find the learning rate 6. Training and testing the model 7. Plotting One Cycle LR 8. Plotting the Triangular Curve 9. Plotting the Training AND Test accuracies. 10. Plotting the Misclassified images 

Modules Description:   

I.	Loading CIFAR 10 Datasets.

II.	Transformations done: Albumentations : 

1.	Rotate, 
2.	Adding PadIfneeded(40), 
3.	Horizontal flip, 
4.	RandomCrop(32), 
5.	RGB Shift, Normalize, 
6.	Cutout strategy of 8 Holes

III.	Loading and running Train dataset and Test dataset:


We have used SGD Optimizer with learning rate 0.03 and Momentum 0.9 and included One Cycle LR.

Model Parameters are 6,573,130

We used LR finder here : To find lr max used exponential steps.
Benefits of Learning rates:
Converge faster to local minima
To get higher accuracy. The training should start from a relatively large learning rate in the beginning, as random weights are far from optimal, and then the learning rate can be decreased during training to allow for more fine-grained weight updates. The trick is to train a network starting from a low learning rate and increase the learning rate exponentially for every batch. Record the learning rate and training loss for every batch. Then, plot the loss and the learning rate. Typically, it looks like this:

Ran the model with 25 epochs with batch size 512 and 10000 iterations.

Train set: Average loss: 0.0031, Accuracy: 86.32
Test set: Average loss: 0.2684, Accuracy: 90.99

Final Accuracy of the model is 90.99

Albumentations:
We can improve the performance of the model by augmenting the data we already have.
Albumentations Package
Albumentations package is written based on numpy, OpenCV, and imgaug. It is a very popular package written by Kaggle masters and used widely in Kaggle competitions. Moreover, this package is very efficient. You may find the benchmarking results here and the full documentation for this package here. Albumentations package is capable of: • Over 60 pixel-level and spatial-level transformations; • Transforming images with masks, bounding boxes, and keypoints; • Organizing augmentations into pipelines; • PyTorch integration.

PyTorch Integration
When using PyTorch you can effortlessly migrate from torchvision to Albumentatios because this package provides specialized utilities to use with PyTorch. Migrating to Albumentations helps to speed up the data generation part and train deep learning models faster.
