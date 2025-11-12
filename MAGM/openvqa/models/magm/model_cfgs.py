# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_AUX_FEAT = False
        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = True
        
        self.SRWSA_MIDSIZE = 4
        self.SGRWSA_SA_MIDSIZE = 16
        self.SGRWSA_GA_MIDSIZE = 64
        self.HIDDEN_SIZE_HEAD = self.HIDDEN_SIZE / self.MULTI_HEAD
        self.SRWSA_LAYER = 3
        self.SGRWSA_LAYER = 3
        self.IMG_FEAT_SIZE = 2048
