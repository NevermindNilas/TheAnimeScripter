import torch

CATEGORIES = [
    {"id": 0, "name": "object", "isthing": 1}
]

IMAGE_ID_ZFILL = 12

COLOR_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
    (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
    (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
    (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
    (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
    (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
    (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
    (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
    (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
    (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
    (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
    (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
    (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
    (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
    (246, 0, 122), (191, 162, 208), (255, 255, 128), (147, 211, 203),
    (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),
    (92, 136, 89), (218, 88, 184), (241, 129, 0), (217, 17, 255),
    (124, 74, 181), (70, 70, 70), (255, 228, 255), (154, 208, 0),
    (193, 0, 92), (76, 91, 113), (255, 180, 195), (106, 154, 176),
    (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
    (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
    (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
    (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
    (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
    (146, 139, 141), (70, 130, 180), (134, 199, 156), (209, 226, 140),
    (96, 36, 108), (96, 96, 96), (64, 170, 64), (152, 251, 152),
    (208, 229, 228), (206, 186, 171), (152, 161, 64), (116, 112, 0),
    (0, 114, 143), (102, 102, 156), (250, 141, 255)
]

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF1010', '10FF10', 'FFF010', '100FFF', '0018EC', 'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=True):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    
colors = Colors()
def get_color(idx):
    if idx == -1:
        return 255
    else:
        return colors(idx)


MULTIPLE_TAGS = {'2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls', 
'2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys',
'2others', '3others', '4others', '5others', '6+others', 'multiple_others'}

if hasattr(torch, 'cuda'):
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEFAULT_DEVICE = 'cpu'

DEFAULT_DETECTOR_CKPT = 'models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'
DEFAULT_DEPTHREFINE_CKPT = 'models/AnimeInstanceSegmentation/kenburns_depth_refinenet.ckpt'
DEFAULT_INPAINTNET_CKPT = 'models/AnimeInstanceSegmentation/kenburns_inpaintnet.ckpt'
DEPTH_ZOE_CKPT = 'models/AnimeInstanceSegmentation/ZoeD_M12_N.pt'