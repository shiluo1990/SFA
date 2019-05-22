# ------------------------------------------
# SFA: Small Faces Attention Face Detector
# Demo
# by Shi Luo
# ------------------------------------------

from __future__ import print_function
from SFA.test import detect
from argparse import ArgumentParser
import os
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe

def parser():
    parser = ArgumentParser('SFA Demo!')
    parser.add_argument('--im',dest='im_path',help='Path to the image',
                        default='data/demo/demo.jpg',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SFA caffe test prototxt',
                        default='SFA/models/test_sfa.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SFA trained caffemodel',
                        default='data/SFA_models/SFA.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='data/demo',type=str)
    parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
                        default='SFA/configs/wider_wide_scale_testing.yml',type=str)
    return parser.parse_args()

if __name__ == "__main__":

    # Parse arguments
    args = parser()

    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # Print config file
    cfg_print(cfg)

    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    print('Loading the network...', end="")
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SFA'
    print('Done!')

    # Read image
    assert os.path.isfile(args.im_path),'Please provide a path to an existing image!'
    pyramid = True if len(cfg.TEST.SCALES)>1 else False

    # Perform detection
    cls_dets,_ = detect(net,args.im_path,visualization_folder=args.out_path,visualize=True,pyramid=pyramid)





