import os
import astra
import numpy as np
from pathlib import Path


def get_reconstruction(input_sino_npy_arr,
                       idx_first_view,
                       idx_last_view,
                       num_det_channels,
                       # num_proj_per_degree,
                       size_recon_ct_img=[512, 512],
                       recon_algorithm='FBP_CUDA',
                       num_iterations=1,
                       GPU_index=0,
                    #    min_proj_angle_degrees=0,
                    #    max_proj_angle_degrees=360,
                       floating_point_precision="float32",
                       short_scan_bool='True'):
    """
    Function to perform reconstruction from a sinogram
    """
    # Clear all used memory of the ASTRA Toolbox
    astra.functions.clear()

    # Set default GPU index to use.
    astra.astra.set_gpu_index(GPU_index)

    # For reconstruction:
    recon_algorithm = recon_algorithm

    # for iterative algorithms
    num_iterations = num_iterations

    # floating point precision for saving reconstructed ct images as npy arrays
    floating_point_precision = floating_point_precision

    # # for projection angle range
    # min_proj_angle_degrees = min_proj_angle_degrees
    # max_proj_angle_degrees = max_proj_angle_degrees

    # # for projection angle range string
    # proj_angle_range_string = str(min_proj_angle_degrees) + "_degrees" + "_to_" + str(max_proj_angle_degrees) + "_degrees"

    ###################
    # ASTRA parameters
    ###################

    # size of output recon image in 'number of pixel' units
    size_recon_ct_img = size_recon_ct_img

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

    #######################
    # For Volume geometry:
    #######################
    # Define and initialize parameters for volume geometry for available DICOM images

    # For this pathology, we assume FOV (both in x and y directions) to be 25 cm.
    # Further, we assume input 'physical space' image to be of size
    # (n_Y = n_rows = size_dcm_image(1) = 512)  * (n_X = n_cols = size_dcm_image(2) = 512) pixels.
    # Therefore, min_X = -12.5 cm, max_X = (12.5 - 12.5/512), and
    # min_Y = -12.5 cm, max_Y = (12.5 - 12.5/512)

    # f.pixel_size = 500/512; % = 0.9765625 = fov/nx [from ct_sys()]
    # vol_geom = astra_create_vol_geom(rows, cols, min_x, max_x, min_y, max_y);

    # Following doesn't work for GPU code
    # vol_geom = astra.create_vol_geom(size_dcm_image[0], size_dcm_image[1], -125, (125 - 125/512),  -125, (125 - 125/512))
    # since: [https://www.astra-toolbox.com/docs/geom2d.html#volume-geometries]:
    # Note: For usage with GPU code, the volume must be centered around the origin and pixels must be square.
    # This is not always explicitly checked in all functions, so not following these requirements may have unpredictable results.
    # vol_geom = astra.create_vol_geom(size_recon_ct_img[0], size_recon_ct_img[1], -125, 125,  -125, 125)

    # vol_geom = astra.create_vol_geom(size_recon_ct_img[0], size_recon_ct_img[1], -35, 35,  -35, 35)
    vol_geom = astra.create_vol_geom(yDim, xDim)

    ###########################
    # For Projection geometry:
    ###########################
    # Define and initialize parameters for projection geometry

    # minimum/initial projection angle (i.e. initial angle made by the nominal source-det. direction with the object)
    # f.orbit_start = 0;
    # min_proj_angle = min_proj_angle_degrees * (np.pi/180)
    min_proj_angle = (idx_first_view/2) * (np.pi/180)

    # maximum/final projection angle (i.e. final angle made by the nominal source-det. direction with the object)
    # max_proj_angle = (5/6)*pi ;
    # f.orbit = 360;                    % value from ctsys.m in MIRT
    # max_proj_angle = max_proj_angle_degrees * (np.pi/180)
    max_proj_angle = (idx_last_view/2) * (np.pi/180)

    # number of angular samples taken for all projections from min_proj_angle to max_proj_angle
    # f.na = 984;	% angular samples   % value from ctsys.m in MIRT
    # n_angl_samples = 230 * (984/360);
    # n_angl_samples = ( (max_proj_angle_degrees - min_proj_angle_degrees) * 2 ) + 1  # (984/360)
    n_angl_samples = idx_last_view - idx_first_view

    #print('n_angl_samples: ', n_angl_samples)

    # define measuring angles (i.e. What are those angular sampling angles?)
    fanbeam_angles = np.linspace(min_proj_angle, max_proj_angle, int(n_angl_samples), True)

    # print('fanbeam_angles: ', fanbeam_angles)

    # proj_geom = astra_create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det);

    # det_width: distance between the centers of two adjacent detector pixels
    # det_count: number of detector pixels in a single projection
    # angles: projection angles in radians
    # source_origin: distance between the source and the center of rotation
    # origin_det: distance between the center of rotation and the detector array

    # proj_geom = astra_create_proj_geom('fanflat', 1, 400, fanbeam_angles, 64, 64);
    # f.ray_spacing = 1.0;	% Approximate detector pitch
    # f.dis_src_det = 949.;	% Approximate
    # f.dis_iso_det = 408.;	% Approximate
    # f.nb = 888;	% detector channels
    # proj_geom = astra_create_proj_geom('fanflat', 1.0, 888, fanbeam_angles, (949-408), 408);

    # proj_geom = astra.create_proj_geom('fanflat',
    #                                    0.1483,  # 1
    #                                    num_det_channels, # 888,
    #                                    fanbeam_angles,
    #                                    410.66,  # (949-408),
    #                                    (553.74-410.66) # 408
    #                                    )

    proj_geom = astra.create_proj_geom('fanflat',
                                       M,  # 1
                                       numDetectors, # 888,
                                       fanbeam_angles,
                                       DSO,  # (949-408),
                                       DOD # 408
                                       )

    # For projector object (not required for GPU code??)
    # Seems to be required when using astra.creators.create_reconstruction()

    # Create a "projector" object  using the GPU.
    # Note that the first time the GPU is accessed, there may be a delay
    # of up to 10 seconds for initialization.
    projector_id = astra.create_projector('cuda', proj_geom, vol_geom)

    # For CPU-based algorithms, a "projector" object specifies the projection
    # model used. In this case, we use the "strip" model.
    # projector_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    sino_numpydata_rotated = np.rot90(input_sino_npy_arr, 3)

    sinogram_id = astra.data2d.create('-sino', proj_geom, data=sino_numpydata_rotated)

    # # # Trying for minc
    # rec_id, rec_data = astra.creators.create_reconstruction(rec_type=recon_algorithm, proj_id=projector_id, sinogram=sinogram_id,
    #                                                         iterations=num_iterations, use_mask='no', use_minc='yes',
    #                                                         minc=0.0, use_maxc='yes', maxc=4095.0, returnData=True, filterType='Ram-Lak',
    #                                                         filterData=None)

    ################
    # Check
    if recon_algorithm == 'FBP_CUDA':
        print('reconstructing with FBP_CUDA')
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectorId'] = projector_id
        cfg['ShortScan'] = short_scan_bool
        cfg['FilterType'] = 'Ram-Lak'

        algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algo_id)

    elif recon_algorithm == 'SIRT_CUDA':
        #print('reconstructing with SIRT_CUDA')
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ReconstructionDataId'] = rec_id
        # cfg['ProjectorId'] = projector_id
        #cfg['MinConstraint'] = 0
        #cfg['MaxConstraint'] = 0.01
        #cfg['GPUindex'] = GPU_index

        algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algo_id, num_iterations)

    elif recon_algorithm == 'CGLS_CUDA':
        print('reconstructing with CGLS_CUDA')
        cfg = astra.astra_dict('CGLS_CUDA')
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ReconstructionDataId'] = rec_id
        # cfg['ProjectorId'] = projector_id
        # cfg['MinConstraint'] = 0
        # cfg['MaxConstraint'] = 4095
        cfg['GPUindex'] = GPU_index

        algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algo_id, num_iterations)

    

    elif recon_algorithm == 'SIRT':
        print('SIRT algorithm for recon')
        # Set up the parameters for a reconstruction algorithm using the CPU
        # The main difference with the configuration of a GPU algorithm is the
        # extra ProjectorId setting.
        cfg = astra.astra_dict('SIRT')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = projector_id

        algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algo_id, num_iterations)

    else:
        print('No algorithm for recon')


    rec_data = astra.data2d.get(rec_id)
    ##################

    return rec_data
