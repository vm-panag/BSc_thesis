import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.lafin import Lafin

# function that applies the lafin inpainting method on a set of images
def lafin_func(model_type, input_path=None, mask_path=None, landmark_path=None, output_path=None, config_path=None): 
    # load config file
    config = Config(config_path)

    config.MODE = 2
    config.MODEL = model_type

    if input_path is not None:
        config.TEST_INPAINT_IMAGE_FLIST = input_path

    if mask_path is not None:
        config.TEST_MASK_FLIST = mask_path

    if landmark_path is not None:
        config.TEST_INPAINT_LANDMARK_FLIST = landmark_path

    if output_path is not None:
        config.RESULTS = output_path


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = Lafin(config)
    model.load()

    print('\nstart testing...\n')
    model.test()
