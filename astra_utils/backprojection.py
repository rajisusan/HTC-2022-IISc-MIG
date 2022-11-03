import os
import astra
import numpy as np
from pathlib import Path

def get_backprojection(input_sino_npy_arr,
                       idx_first_view,
                       idx_last_view,
                       size_recon_ct_img=[512, 512],
                       num_det_channels=560,
                       GPU_index=0,
                    #    min_proj_angle_degrees=0,
                    #    max_proj_angle_degrees=360,
                       floating_point_precision="float32"):
    """
    
    """

    # Some settings for sinogram generation

    # Set default GPU index to use for sinogram generation with ASTRA
    GPU_index = GPU_index
    astra.astra.set_gpu_index(GPU_index)

    # size of DICOM images in 'number of pixel' units
    # size_recon_ct_img = input_recon_ct_img_npy_arr.shape  # [512, 512]

    # floating point precision for saving sinogram npy arrays
    floating_point_precision = floating_point_precision

    # # for projection angle range
    # min_proj_angle_degrees = min_proj_angle_degrees
    # max_proj_angle_degrees = max_proj_angle_degrees

    # for projection angle range string
    # proj_angle_range_string = str(min_proj_angle_degrees) + "_degrees" + "_to_" + str(max_proj_angle_degrees) + "_degrees"

    ###################
    # ASTRA parameters
    ###################

    # Please see: https://github.com/Diagonalizable/HelTomo/blob/main/create_ct_operator_2d_fan_astra_cuda.m
    xDim = size_recon_ct_img[1]
    yDim = size_recon_ct_img[0]

    # Create shorthands for needed variables
    DSD = 553.74
    DSO = 410.66
    M = 1.3484  # CtData.parameters.geometricMagnification
    numDetectors = num_det_channels  # CtData.parameters.numDetectorsPost
    effPixel = 0.1483 # CtData.parameters.effectivePixelSizePost

    # Distance from origin to detector
    DOD = DSD - DSO

    # Distance from source to origin specified in terms of effective pixel size
    DSO = DSO / effPixel

    # Distance from origin to detector specified in terms of effective pixel size
    DOD = DOD / effPixel

    vol_geom = astra.create_vol_geom(yDim, xDim)

    # Parameters for projection geometry

    # minimum/initial projection angle (i.e. initial angle made by the nominal source-detector direction with horizontal)
    # f.orbit_start = 0;
    # In radians
    # min_proj_angle = min_proj_angle_degrees * (np.pi/180)
    min_proj_angle = (idx_first_view/2) * (np.pi/180)

    # maximum/final projection angle (i.e. final angle made by the nominal source-det. direction with horizontal)
    # max_proj_angle = (5/6)*pi ;
    # f.orbit = 360;                    # value from ctsys.m in MIRT
    # In radians
    # max_proj_angle = max_proj_angle_degrees * (np.pi/180)
    max_proj_angle = (idx_last_view/2) * (np.pi/180)

    # number of angular samples taken for all projections from min_proj_angle to max_proj_angle
    # f.na = 984; # angular samples   # value from ctsys.m in MIRT
    # n_angl_samples = max_proj_angle_degrees * (984/360)

    # n_angl_samples = ( (max_proj_angle_degrees - min_proj_angle_degrees) * 2 ) + 1
    n_angl_samples = idx_last_view - idx_first_view

    fanbeam_angles = np.linspace(min_proj_angle, max_proj_angle, int(n_angl_samples), True)

    # proj_geom = astra_create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
    # f.ray_spacing = 1.0;	% Approximate detector pitch
    # f.dis_src_det = 949.;	% Approximate
    # f.dis_iso_det = 408.;	% Approximate
    # f.nb = 888;	% detector channels
    # proj_geom = astra_create_proj_geom('fanflat', 1.0, 888, fanbeam_angles, (949-408), 408);
    # proj_geom = astra.create_proj_geom('fanflat', 1.0, 888, fanbeam_angles, (949-408), 408)

    proj_geom = astra.create_proj_geom('fanflat',
                                       M,  # 1
                                       numDetectors, # 888,
                                       fanbeam_angles,
                                       DSO,  # (949-408),
                                       DOD # 408
                                       )

    # Create a "projector" object  using the GPU.
    # Note that the first time the GPU is accessed, there may be a delay
    # of up to 10 seconds for initialization.
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    sino_numpydata_rotated = np.rot90(input_sino_npy_arr, 3)

    ############################
    # Generating backprojection
    ############################

    backprojection_id, backprojection = astra.create_backprojection(sino_numpydata_rotated,
                                                                    proj_id)

    # Free memory
    astra.data2d.delete(backprojection_id)
    astra.projector.delete(proj_id)

    return backprojection