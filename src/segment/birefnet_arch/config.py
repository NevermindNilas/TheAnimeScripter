"""Fixed inference configuration for the Swin Tiny BiRefNet Lite checkpoint."""


class Config:
    def __init__(self):
        self.batch_size = 8
        self.SDPA_enabled = True
        self.bb = "swin_v1_t"
        self.freeze_bb = False
        self.lateral_channels_in_collection = [1536, 768, 384, 192]
        self.cxt = [192, 384, 768]
        self.auxiliary_classification = False
        self.squeeze_block = "BasicDecBlk_x1"
        self.dec_blk = "BasicDecBlk"
        self.lat_blk = "BasicLatBlk"
        self.dec_channels_inter = "fixed"
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.mul_scl_ipt = "cat"
        self.dec_att = "ASPPDeformable"
        self.ms_supervision = True
        self.out_ref = True
