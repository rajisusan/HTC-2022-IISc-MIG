# Program to test/do inference from several inputs

# Imports
from genericpath import exists
import os
import argparse
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.image as image
from pathlib import Path
from skimage import filters

# Other imports

from utils.load_net_from_checkpoint import get_inference_net_from_checkpoint
from utils.list_of_files_paths import list_paths_for_files_in_dir_and_subdir
from utils.util import load_single_sino_tensor_frm_path_ext_mat
from utils.util import convert_batchsize_1_channel_1_tensor_to_2D_npy_array
from astra_utils.backprojection import get_backprojection
from astra_utils.forward_projection import get_forward_projection
from astra_utils.reconstruction import get_reconstruction

#############################
# Argparse arguments/options
#############################

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_input_folder', required=True,
                    help='path to test images parent dir')
parser.add_argument('--path_to_output_folder',
                    required=True,
                    help='parent folder for subfolder for saving images from testing')
parser.add_argument('--group_category', type=int, required=True,
                    help='Group Category')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nef', type=int, default=64)      # Used in network model
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_index_string', default="cuda:0",  help='index of GPU to use')
parser.add_argument('--gpu_index_1_string', default="cuda:0",  help='index 1 of GPU to use')
parser.add_argument('--load_path_netG',
                    required=True,
                    help="path to load netG")
parser.add_argument('--cuda', action='store_false', help='enables cuda')
parser.add_argument('--save_err_surfplot_figpng', action='store_true', help="if option not used, 'save_err_surfplot_figpng' defaults to false")
parser.add_argument('--do_scaling_of_data', action='store_true', help="if option not used, 'do_scaling_of_data' defaults to false")
parser.add_argument('--inp_scaling_factor', type=float, default=1, help='scaling factor for input data')
parser.add_argument('--row_idx', type=int, default=280, help='row index for plotting line profile')
parser.add_argument('--num_det_pxls', type=int, default=560, help='No. of det. pixels')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--size_recon_img_tuple', default=(512, 512), help='size_recon_img_tuple')
parser.add_argument('--num_rows_ct_img', type=int, default=512, help='the height of the input image to network')
parser.add_argument('--num_cols_ct_img', type=int, default=512, help='the width of the input image to network')

parser.add_argument('--inp_ct_blurry_max_value_scaling_factor', type=float, default=7.00e-3, help='scaling factor for input data')
parser.add_argument('--inp_ct_blurry_division_factor', type=float, default=5.00e2, help='division factor for input data')

parser.add_argument('--num_extnd_views', type=int, default=481, help='No. of extended views of sino')
parser.add_argument('--num_sirt_iterations', type=int, default=40, help='num sirt iterations to use')

opt = parser.parse_args()
print(opt)


# Get number of limited views from category

if opt.group_category==1:
   num_ltd_views=181
elif opt.group_category==2:
    num_ltd_views=161
elif opt.group_category==3:
    num_ltd_views=141
elif opt.group_category==4:
    num_ltd_views=121
elif opt.group_category==5:
    num_ltd_views=101
elif opt.group_category==6:
    num_ltd_views=81
else:
    num_ltd_views=61

opt.num_ltd_views=num_ltd_views

print('Group Category:', opt.group_category, 'Number of limited angle views:', opt.num_ltd_views)

# option variables
gpu_index_string = opt.gpu_index_string
ngpu = opt.ngpu

# Decide which device we want to run on
device = torch.device(gpu_index_string if (torch.cuda.is_available() and ngpu > 0) else "cpu")

##############
# Paths for saving outputs

outp_dir = opt.path_to_output_folder

# Creating the required directories
if not os.path.exists(outp_dir):
    os.makedirs(outp_dir)


#########
# Main()
#########

if __name__ == "__main__":

    ##############################################
    # Configure paths and directories for inputs
    ##############################################
       
    # Get full dir path for test files
    path_dir_test_sino = opt.path_to_input_folder
    
    # Get a list of paths of sino files
    list_paths, files = \
        list_paths_for_files_in_dir_and_subdir(path_top_dir=path_dir_test_sino, ext='.mat')
    #print('list_paths:', list_paths)

    ##################
    # Loading network
    ##################

    # with torch.no_grad():
    netG_instance = get_inference_net_from_checkpoint(opt, device)

    # serial number; it will be incremented and displayed as we iterate with the for-loop
    serial_num = 0

    # initialize a variable to save  the scores
    sc_temp=[]

    ###############################
    # Iterate over test sino files
    ###############################
    for path_fname_ext in list_paths:

        serial_num = serial_num + 1
        # (1) Load the test input sinogram and init_view
        with torch.no_grad():               # required?
            test_gtrue_sino_tensor, init_angle = \
                load_single_sino_tensor_frm_path_ext_mat(path_fname_ext=path_fname_ext,
                                                                 device=device)
            # We don't need grad
            test_gtrue_sino_tensor.requires_grad_(requires_grad=False)
            init_angle = int(init_angle)

        #print('init_angle:',init_angle)
        init_view = init_angle*2    
        #print('init_view:',init_view)
        # Get file name
        fname_groundtruth_sino = Path(path_fname_ext).stem

        ####################################
        # Backproject test sino using ASTRA
        ####################################

        # Get init_view of LA sino
        #init_view = get_init_view_frm_fname(fname_str=fname_groundtruth_sino)
        
        #test_gtrue_sino_tensor=test_gtrue_sino_tensor[:,init_view:init_view+opt.num_ltd_views]
        print('test_gtrue_sino_tensor Shape', test_gtrue_sino_tensor.shape)

        lamino_2d_np_arr = get_backprojection(input_sino_npy_arr=test_gtrue_sino_tensor.cpu().detach().numpy(),
                                              idx_first_view=init_view,
                                              idx_last_view=init_view+opt.num_ltd_views,
                                              size_recon_ct_img=[512, 512],
                                              num_det_channels=560,
                                              GPU_index=0,
                                              floating_point_precision="float32")

        # Convert to torch tensor
        lamino_2d_tensor = torch.from_numpy(lamino_2d_np_arr)

        # Scale input  data
        with torch.no_grad():
            lamino_nrmlzd_2d_tensor = torch.div(lamino_2d_tensor,
                                                opt.inp_ct_blurry_division_factor)
            lamino_scld_2d_tensor = torch.mul(input=lamino_nrmlzd_2d_tensor,
                                              other=opt.inp_ct_blurry_max_value_scaling_factor)

        # Convert to 4D tensor
        lamino_scld_4d_tensor = lamino_scld_2d_tensor.unsqueeze(0).unsqueeze(0)
        lamino_scld_4d_tensor = lamino_scld_4d_tensor.to(device)

        print(serial_num, '   Performing Reconstruction of ', files[serial_num-1])

        ############
        # Inference:
        ############

        # Get output from the network
        
        # with torch.no_grad():
        outp_ct_less_blurry_4d_tnsr = netG_instance(lamino_scld_4d_tensor)
        
        
        #convert to numpy array
        outp_ct_less_blurry_2d_np = \
            convert_batchsize_1_channel_1_tensor_to_2D_npy_array(tensor_batchsize_1_channel_1=outp_ct_less_blurry_4d_tnsr)

        ###########################
        # Project sino using ASTRA
        ###########################

        sino_projected_2d_np =  get_forward_projection(input_recon_ct_img_npy_arr=outp_ct_less_blurry_2d_np,
                                                       idx_first_view=0,
                                                       idx_last_view=721,
                                                       num_det_channels=560,
                                                       GPU_index=0,
                                                       # min_proj_angle_degrees=0,
                                                       # max_proj_angle_degrees=360,
                                                       floating_point_precision="float32")

        # For:
        # ValueError: At least one stride in the given numpy array is negative, and tensors with
        # negative strides are not currently supported. (You can probably work around this by 
        # making a copy of your array  with array.copy().) 
        sino_projected_2d_np = sino_projected_2d_np.copy()

        # Convert to torch tensor
        sino_projected_2d_tensor = torch.from_numpy(sino_projected_2d_np)

        ####################
        # Get extended sino
        ####################

        # Both tensors should be on same device
        sino_projected_2d_tensor = sino_projected_2d_tensor.to(torch.device(opt.gpu_index_string))
                
        sino_extended_2d_tensor_tmp = torch.hstack((sino_projected_2d_tensor[:, 0:init_view],
                                               test_gtrue_sino_tensor[:,0:opt.num_ltd_views]))
        
        sino_extended_2d_tensor = torch.hstack((sino_extended_2d_tensor_tmp[:, 0:init_view+opt.num_ltd_views],
                                               sino_projected_2d_tensor[:,init_view+opt.num_ltd_views:721]))
        
        #############################################
        # SIRT reconstruction for extended sinogram:
        #############################################
        
        # to adjust the init_view+extension if it goes beyond 721
        temp = init_view+opt.num_extnd_views
        
        if temp>721:
            temp1 = temp-721
            init_view=init_view-temp1

        # get extended sinogram input to SIRT
        sino_extended_2d_tensor=sino_extended_2d_tensor[:,init_view:init_view+opt.num_extnd_views]

        recon_sirt = get_reconstruction(input_sino_npy_arr=sino_extended_2d_tensor.cpu().detach().numpy(),
                                        idx_first_view=init_view,
                                        idx_last_view=init_view+opt.num_extnd_views,
                                        num_det_channels=560,
                                        # num_proj_per_degree=2,
                                        size_recon_ct_img=[512, 512],
                                        recon_algorithm='SIRT_CUDA',
                                        num_iterations=opt.num_sirt_iterations,
                                        GPU_index=0,
                                        # min_proj_angle_degrees=0,
                                        # max_proj_angle_degrees=240,
                                        floating_point_precision="float32",
                                        short_scan_bool=False)

        # Perform the segmentation using otsu thresholding
        im = recon_sirt
        th = filters.threshold_otsu(im)
        seg_im = np.zeros(im.shape)
        seg_im[im>th] =1

        print(serial_num, '   Saving Reconstruction of ', files[serial_num-1])

        # Save the segmented image

        with torch.no_grad():

            save_path_w_fname=outp_dir + '/' + files[serial_num-1]
            image.imsave(save_path_w_fname+'.png', seg_im, cmap='gray')
               
