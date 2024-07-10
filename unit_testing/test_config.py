import sys
sys.path.insert(0, ".")
import argparse
from configs import *

class TestORChiDConfigs:

    def __init__(self, config):
        self.test_cfg = eval(opts.config)



