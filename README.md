# CTC-2022-IISc-MIG

# Helsinki Tomography Challenge 2022 (HTC 2022) Submission

#### *Ashish Verma, Raji Susan Mathew, Phaneendra K. Yalavarthy*

##### *Department of Computational and Data Sciences, IISc Bangalore, Karnataka, INDIA- 560012*


## A Deep Learning based Back Project Filter (BPF) Method for Limited Angle Computed Tomography

<p align="justify" markdown="1">
Computed tomography (CT) is an efficient imaging tool that plays an important role in medical diagnoses, industrial inspection, and security checks. The accurate reconstruction of a CT image requires the test object to be at least scanned under consecutive 180° or 180°+fan angles for parallel-beam or fan-beam geometries, respectively. In order to reduce the harmful radiation doses, the limited angle (LA) acquisition with decreased number of projection views becomes a more preferable choice in medical applications. However, with such acquisitions, the conventional CT reconstruction approaches such as filtered back projection (FBP) [1] and iterative reconstruction approaches exhibit limited angle artifacts. To overcome the difficulties associated with the above mentioned conventional CT reconstruction approaches, we propose a backproject filter based reconstruction wherein the deconvolution operation is performed using a convolutional neural network (CNN). 
</p>



### Proposed Approach

<p align="justify" markdown="1">
The work flow of this approach can be explained as follows. Initially the limited angle sinogram was back projected to obtain a blurred CT image. The back projected CT image was then fed to a deep symmetric encoder decoder architecture (UNet) [2] to obtain a CT image with reduced blurring. This image was forward projected to obtain a sinogram with an extended number of views (for example views corresponding to 180°+60°, i.e., 481 views were used throughout this approach). Then a simultaneous iterative reconstruction technique (SIRT) [3] was performed on the extended sinogram to obtain the final reconstruction. A schematic representation of the proposed approach is shown in Figure 1.  
</p>


<p align="center">
  <img src="https://github.com/rajisusan/CTC-2022-IISc-MIG/blob/main/Picture1.png">
  <em>Figure 1: Schematic representation of the proposed reconstruction approach.</em>
</p>  


## Helsinki Tomography Challenge 2022 (HTC 2022)

<p align="justify" markdown="1">
The objective of the Helsinki Tomographic Challenge is to recover the shapes of 2D targets imaged with LA acquisitions, collected in the Industrial Mathematics Computed Tomography Laboratory at the University of Helsinki, Finland [4]. The targets are homogenous acrylic disc phantoms of 70mm in diameter, with a different number of irregular holes in random locations. The expected outcome of the challenge should be an algorithm which takes in the X-ray data, i.e., the sinogram and its associated metadata about the measurement geometry, and produces a reconstruction which has been segmented into two components: air and plastic. The challenge data have been scanned using full-angle tomography, and have been appropriately subsampled to create the training data for the different difficulty groups ranging from 181 views to 61 views.   
</p>
 
### Training Details

<p align="justify" markdown="1">
Data: The given challenge data was partitioned into training and validation with the first four datasets (solid_disc_full, ta, tb, tc) for training and the last dataset (td) for validation. We have subsampled each dataset to create the training data for the different difficulty groups ranging from 181 views to 61 views. 
</p>
<p align="justify" markdown="1">
Training: For the model with 181 views, training was done from scratch. For all other models, transfer learning was performed from this model. The model used was a deep UNet with 8 encoders and decoders. The training was performed with MSE loss and Adam optimizer with a batch size of 24. The checkpoints were shared at <a href="https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/rajisusanm_iisc_ac_in/EtC0RNiFsT1Kn_qCIQ3GD6kBNJwRYLqUYfx0bjcKjgNB5Q?e=qZGZh2
">[Checkpoints]</a>. Each checkpoints were named corresponding to the respective difficulty level/ group category. For example, the checkpoint for difficulty level 1 with 181 views was named as ‘1.tar’. 
</p>

## Installation Instructions
<p align="justify" markdown="1">
The environment.yml file used for testing the datasets was added to the repository as ‘environment.yml’. The testing code was shared in the repository as main.py
</p>

## Usage:
```md
python3 main.py --path_to_input_folder '/path/to/input/folder' --path_to_output_folder '/path/to/output/folder' --group_category 1 --load_path_netG '/path/to/Checkpoints'
```
## Sample Reconstructions

<p align="justify" markdown="1">
Two examples of the reconstructions from the training set are shown here. Figure 2 shows the segmentated images obtained using the proposed approch for dataset 'ta' (for seven difficulty levels) with the view angles beginning from 270 <sup></0>. Figure 3 shows the segmentated images obtained using the proposed approch for dataset 'tc' (for the seven difficulty levels) with the view angles beginning from 0 degrees. The scores obtained using Matthews correlation coefficient for each reconstruction are shown in the insets. 
</p>

<p align="center">
  <img src="https://github.com/rajisusan/CTC-2022-IISc-MIG/blob/main/Picture2.png">
  <em>Figure 2: Segmented images obtained using the proposed reconstruction approach for dataset 'ta' with view angles beginning from 270 degrees.</em>
</p>


<p align="center">
  <img src="https://github.com/rajisusan/CTC-2022-IISc-MIG/blob/main/Picture3.png">
  <em>Figure 3: Segmented images obtained using the proposed reconstruction approach for dataset 'tc' with view angles beginning from 0 degrees.</em>
</p>


#### References
<p align="justify" markdown="1">
[1]  L. A. Feldkamp, L. C. Davis, and J. W. Kress, “Practical cone-beam algorithm,” Journal of the Optical Society of America A, vol. 1, no. 6, pp. 612–619, 6 1984.
</p>

<p align="justify" markdown="1">
[2] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation. InInternational Conference on Medical image computing and computer-assisted intervention 2015 Oct 5 (pp. 234-241). Springer, Cham.
</p>

<p align="justify" markdown="1">
[3] Gilbert, P.: ‘Iterative methods for the three-dimensional reconstruction of an object from projections’, J. Theor. Biol., 1972, 36, (1), pp. 105–117.
</p>

<p align="justify" markdown="1">
[4] Salla Latva-¨Aij¨o, Alexander Meaney, Siiri Rautio, Samuli Siltanen, Fernando Silva de Moura, Tommi Heikkil¨a “Helsinki Tomography Challenge 2022 (HTC 2022)”, 28th of October, 2022.
</p>
