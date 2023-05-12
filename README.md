# 3DRGAN
This is the version of our model that handles 2D data, the version that handles 3D data will be uploader later.
1.requirementns

torch == 1.7.0

To run the code,an NVIDIA GeForce RTX 3080 GPU with 10G video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

Programming language: Python 3.6. 

2.How to use？

First, preprocess the images: cut the shale images into 80*80 size png images and store them in the folder for the training data and the folder for the test data

Secondlly,set the parameters in the train.py, such as the path to the training data,the scale factor and the save location. After configuring the parameters and enviorment,you can run directly: python train.py

Finally, set the parameters in the recon.py, such as the path to the test data, the scale factor and the model location. After configuring, you can run: python recon.py and then get the reconstructed results