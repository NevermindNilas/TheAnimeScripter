def build_generator_arch(version):
    if version.lower() == "v00":
        from .PFlowVFI_V0 import Network

        model = Network(dilate_size=9)

    ################## ABLATION ##################
    elif version.lower() == "ab_b_n":
        from .PFlowVFI_ablation import Network

        model = Network(dilate_size=7, mask_type="binary", noise=True)
    elif version.lower() == "ab_b_nf":
        from .PFlowVFI_ablation import Network

        model = Network(dilate_size=7, mask_type="binary", noise=False)
    elif version.lower() == "ab_qb_nf":
        from .PFlowVFI_ablation import Network

        model = Network(dilate_size=7, mask_type="quasi-binary", noise=False)
    elif version.lower() == "ab_a":
        from .PFlowVFI_adaptive import Network

        model = Network(dilate_size=7)
    ################## ABLATION ##################

    elif version.lower() == "vb":
        from .PFlowVFI_Vb import Network

        model = Network(9)

    elif version.lower() in ["v20_nll", "v20_laper"]:
        from .PFlowVFI_V2 import Network_flow

        model = Network_flow(5)
    elif version.lower() == "v2b":
        from .PFlowVFI_V2 import Network_base

        model = Network_base(5)

    return model
