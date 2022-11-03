import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from matplotlib import cm
import torch.fft as torch_fft
import os
import scipy.io

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size[1], size[0]), Image.ANTIALIAS)
        # print('size[0], size[1]:', size[0], size[1])
        # breakpoint()
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def load_single_sino_tensor_frm_path_ext_mat(path_fname_ext, device):
    """"
    Function to load (and return) a sinogram
    from a directory storing it in .npy format
    """
    # filename_with_ext = filename + '.npy'
    # path_w_fname_sino = os.path.join(path_dir, filename_with_ext)

    with torch.no_grad():
        data_dict = scipy.io.loadmat(path_fname_ext)
        #print('data keys:', list(data_dict.keys()))

        # Get CtDataLimited and angles
        CtData = data_dict['CtDataLimited']
        #CtData = data_dict['CtDataFull']
        sino = CtData[0, 0][1]
        sino = np.rot90(sino, 1)

        params = CtData[0, 0][2]
        angles= params['angles'][0][0][0]
        #print('angle',angles.shape)
        
        #print('init_angle type:', init_angle.type)
        init_angle = angles[[0]] 
        print('init_angle',init_angle)
        
        sino = sino.copy()
        sino_tensor = torch.from_numpy(sino)

        
        if torch.cuda.is_available():
            sino_tensor = sino_tensor.to(device)

    return sino_tensor, init_angle




def convert_batchsize_1_channel_1_tensor_to_2D_npy_array(tensor_batchsize_1_channel_1):
    two_D_npy_array = tensor_batchsize_1_channel_1[0, 0, :, :].cpu().detach().numpy()
    return two_D_npy_array
