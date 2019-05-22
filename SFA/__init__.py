# --------------------------------------------------------
# SFA: Small Faces Attention Face Detector
# This file is a modified version from https://github.com/mahyarnajibi/SSH
# Modified by Shi Luo
# --------------------------------------------------------

import sys
# Add caffe and lib to the paths
if not 'caffe-sfa/python' in sys.path:
    sys.path.insert(0,'caffe-sfa/python')
if not 'lib' in sys.path:
    sys.path.insert(0,'lib')
from utils.get_config import cfg

if not cfg.DEBUG:
    import os
    # Suppress Caffe (it does not affect training, only test and demo)
    os.environ['GLOG_minloglevel']='3'
