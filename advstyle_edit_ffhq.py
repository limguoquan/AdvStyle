import os
import argparse
import torch
import sys
import numpy as np
import math
from tqdm import tqdm
# import cv2
from functions import save_img, read_image_path
from visualization import adjust_pixel_range
from models.stylegan_generator import StyleGANGenerator
from utils.util import torch_to_numpy, manipulation


def parse_attr(attribute):
    attr_list = attribute.split(",")
    return attr_list

def manipulate_test(attribute, output_dir, image_name, noise_path, resolution, gan_model, latent_type):
    attr_list = parse_attr(attribute)
    boundary = []
    manipulate_layers = []
    shift_range = []

    for attr in attr_list:
        direction_path = os.path.join("./boundaries", f"{attr}.npy")
        direction_dict = np.load(direction_path, allow_pickle=True)[()]
        boundary.append(direction_dict["boundary"])
        # direction vector
        manipulate_layers.append(direction_dict["manipulate_layers"])
        # specific operation layers
        shift_range.append(direction_dict["shift_range"])

        '''
        recommand range
        direction_dict["shift_range"]: [-10, 10] 
        represents that the negative direction step is -10 and the positive direction step is 10
        '''

    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(f"{output_dir}/{attr_list[0]}", exist_ok=True)

    step = 7
    gan = StyleGANGenerator(gan_model)
    gan.net.eval()

    num_layers = int((math.log(resolution, 2) -1)*2)

    latent = np.load(noise_path)
    noise_torch = torch.from_numpy(latent).float().cuda()

    if latent_type == "ws":
        ws = noise_torch
    elif latent_type == "z":
        w = gan.net.mapping(noise_torch)
        ws = gan.net.truncation(w)
    if latent_type == "w":
        ws = gan.net.truncation(noise_torch)

    output_images = []
    wp_np = torch_to_numpy(ws)
    shift_range = np.array(shift_range)
    wp_mani = manipulation(latent_codes=wp_np,
            boundary=boundary,
            start_distance=shift_range[:,0],
            end_distance=shift_range[:,1],
            steps=step,
            layerwise_manipulation=True,
            num_layers=num_layers,
            manipulation_layers=manipulate_layers,
            is_code_layerwise=True,
            is_boundary_layerwise=False)
    '''
    When generating one image,
    please set step to 1,
    set end_distance to x,
    where shift_range[:,0] <= x <= shift_range[:,1] is recommended,
    set start_distance randomly.

    when generating multi images(multi steps),
    please set end_distance to shift_range[:,1],
    set start_distance to shift_range[:,0]
    
    wp_np shape: [batch_size, steps, num_layers, *code_shape]
    '''
    for step_idx in range(step):
        test_torch = torch.from_numpy(wp_mani[:,step_idx,:,:])
        test_torch = test_torch.type(torch.FloatTensor).cuda()
        images = gan.net.synthesis(test_torch)
        
    save_img(images, f"{output_dir}/{image_name}.png",is_torch=True, is_map=False, trans_type=None)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--attribute", type=str, default="supermodel", help="manipulate attribute name")
    parser.add_argument("--output_dir", type=str, default="/home/FYP/limg0038/ials/invertedImages/img_advstyle")
    parser.add_argument("--latent_code_path", type=str, default="/home/FYP/limg0038/ials/invertedImages/latent_code_ials")
    parser.add_argument("--resolution", default=1024, type=int)
    parser.add_argument("--gan_model", default="stylegan_ffhq", type=str)
    parser.add_argument("--latent_type", default="w", type=str)

    opt, _ = parser.parse_known_args()

    latent_codes = os.listdir(opt.latent_code_path)

    for latent_code in latent_codes:
        image_name = latent_code[:-4]
        noise_path = opt.latent_code_path + '/' + latent_code
        manipulate_test(opt.attribute, opt.output_dir, image_name, noise_path, opt.resolution, opt.gan_model, opt.latent_type)