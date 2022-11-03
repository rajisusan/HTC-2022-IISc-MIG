import os
import astra
import numpy as np
from pathlib import Path


def get_forward_projection(input_recon_ct_img_npy_arr,
                           idx_first_view,
                           idx_last_view,
                           num_det_channels,
                           GPU_index=0,
                           # min_proj_angle_degrees=0,
                           # max_proj_angle_degrees=360,
                           floating_point_precision="float32"):
    """
    Using this function is perhaps inefficient when many sinograms need to be
    generated since astra.projector.delete(proj_id) is called in it for every
    call to this function.
    """

    # Some settings for sinogram generation

    # Set default GPU index to use for sinogram generation with ASTRA
    GPU_index = GPU_index
    astra.astra.set_gpu_index(GPU_index)

    # size of DICOM images in 'number of pixel' units
    size_recon_ct_img = input_recon_ct_img_npy_arr.shape  # [512, 512]

    # floating point precision for saving sinogram npy arrays
    floating_point_precision = floating_point_precision

    # # for projection angle range
    # min_proj_angle_degrees = min_proj_angle_degrees
    # max_proj_angle_degrees = max_proj_angle_degrees

    # # for projection angle range string
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

    # Parameters for volume geometry for available DICOM images

    # For head anatomy, we assume FOV (both in x and y directions) to be 25 cm (== 250 mm).
    # Further, we assume input image to be of size
    # (n_Y = n_rows = size_recon_ct_img(1) = 512)  * (n_X = n_cols = size_recon_ct_img(2) = 512) pixels.
    # Therefore, min_X = -12.5 cm, max_X = (12.5 - 12.5/512), and
    # min_Y = -12.5 cm, max_Y = (12.5 - 12.5/512)

    # f.pixel_size = 500/512; % = 0.9765625 = fov/nx [from ct_sys() of MIRT]
    # vol_geom = astra_create_vol_geom(rows, cols, min_x, max_x, min_y, max_y);
    # vol_geom = astra.create_vol_geom(size_recon_ct_img[0], size_recon_ct_img[1], -125, (125 - 125/512),  -125, (125 - 125/512))

    # Note: It was found later during reconstruction with FBP_CUDA algorithm for fan-flat projection geometry that:
    # Following doesn't work for GPU code
    # vol_geom = astra.create_vol_geom(size_recon_ct_img[0], size_recon_ct_img[1], -125, (125 - 125/512),  -125, (125 - 125/512))
    # since: [https://www.astra-toolbox.com/docs/geom2d.html#volume-geometries]
    # "Note: For usage with GPU code, the volume must be centered around the origin and pixels must be square.
    # This is not always explicitly checked in all functions, so not following these requirements may have unpredictable results."
    # and following worked in the reconstrution code:
    # vol_geom = astra.create_vol_geom(size_recon_ct_img[0], size_recon_ct_img[1], -125, 125,  -125, 125)

    vol_geom = astra.create_vol_geom(yDim, xDim)

    # Parameters for projection geometry

    # minimum/initial projection angle (i.e. initial angle made by the nominal source-detector direction with horizontal)
    # f.orbit_start = 0;
    # min_proj_angle = min_proj_angle_degrees * (np.pi/180)
    min_proj_angle = (idx_first_view/2) * (np.pi/180)

    # maximum/final projection angle (i.e. final angle made by the nominal source-det. direction with horizontal)
    # max_proj_angle = (5/6)*pi ;
    # f.orbit = 360;                    # value from ctsys.m in MIRT
    # max_proj_angle = max_proj_angle_degrees * (np.pi/180)
    max_proj_angle = (idx_last_view/2) * (np.pi/180)

    # number of angular samples taken for all projections from min_proj_angle to max_proj_angle
    # f.na = 984; # angular samples   # value from ctsys.m in MIRT
    # n_angl_samples = max_proj_angle_degrees * (984/360)

    # n_angl_samples = ( (max_proj_angle_degrees - min_proj_angle_degrees) * 2 ) + 1
    n_angl_samples = idx_last_view - idx_first_view

    #print('n_angl_samples: ', n_angl_samples)

    # define measuring angles (i.e. What are those angular sampling angles?)
    # linspace(start: Union[ndarray, Iterable, int, float], stop: Union[ndarray, Iterable, int, float], num: Optional[int] = 50, endpoint: Optional[bool] = True, ...
    # fanbeam_angles = np.linspace(min_proj_angle, max_proj_angle, int(n_angl_samples), True)

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

    ######################
    # Generating sinogram
    ######################

    # sinogram_id, sinogram = astra.create_sino(pixel_array_dicom, proj_id)
    sinogram_id, sinogram = astra.create_sino(input_recon_ct_img_npy_arr, proj_id)

    # Rotate the generated sinograms counter-clockwise by 90 degrees
    sinogram_np_arr = np.rot90(sinogram)

    # Free memory
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    # Clear all used memory of the ASTRA Toolbox
    astra.functions.clear()

    return sinogram_np_arr
