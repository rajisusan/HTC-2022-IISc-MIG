import os
import torch
from models.unet_based_models_kern7_pddng_3_for_H512_W512_w_disc_fr_512_512 import Generator as netG


def get_inference_net_from_checkpoint(opt, device):
    """
    Get a network model instance and intialize it with a saved checkpoint
    for inference
    """
    # Loading the Generator network from saved checkpoint:
    file_name_gen_checkpoint = str(opt.group_category) + '.tar'
    load_path_and_name_netG = os.path.join(opt.load_path_netG, file_name_gen_checkpoint)

    # Getting an instance/object of the Generator class
    netG_instance = netG(opt)
    if opt.cuda:
        netG_instance.cuda(device=device)

    # To load a pretrained netG model for inference:
    # netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    # checkpoint = torch.load(opt.load_path_and_name_netG)
    # https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/2
    checkpoint = torch.load(load_path_and_name_netG, map_location='cpu')
    netG_instance.load_state_dict(checkpoint['model_state_dict'])
    netG_instance.to(device)
    netG_instance.eval()

    return netG_instance
