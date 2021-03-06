'''
video mode setting: {
    "display": Random colors, While ink would fade, like tails,
    "styleTransfer": Stylize the whole input from webcam or only your clothes,
    "grabCut": Move you from the whole scene to a new video,
    "glass": Help you wear a pair of glasses,
    "AR": Build a roof on the plane you choose,
}
'''
from collections import deque
import tensorflow as tf
import numpy as np
import cv2
from modes.styleTransfer_mode.im_transf_net import create_net
from modes.AR_mode.plane_ar import App
print(__doc__)
MODE = "display"

# argument settings
args = {
    "video_source": 0,
    "deque_buffer": 16,
    "green_lower": (19, 89, 64),   # color range of ball in the HLS color space
    "green_upper": (69, 250, 255),
}

# menu_top settings
# order of items: right -> left
args_menu = {
    "menu_dict": {
        "color_blue": (253, 1, 1),
        "color_green": (1, 253, 1),
        "color_red": (1, 1, 253),
        "plus_icon": "images/plus_icon.png",
        "minus_icon": "images/minus_icon.png",
        "scissors_icon": "images/scissors.jpg",
        "art_icon": "images/theStarryNight.jpg",
        "glass_icon": "images/glass.jpg",
    },
    "icon_len_side": 80,
}

# arguments for tracking
pts = deque(maxlen=args["deque_buffer"])
for i in range(len(pts)):
    pts.appendleft((0, 0))
args_display = {
    "pts": pts,
    "counter": 0,
    "dXY": (0, 0),
    "direction": "",
    "drawing_color": (0, 255, 255),
    "is_start": 1,
    "thick_coeff": 2.5,
}

# arguments for style transfer
args_styleTransfer = {
    'model_path': './models/starry_final.ckpt',
    'upsample_method': 'resize',
    'content_target_resize': (1, 240, 320, 3),
    'skin_lower': (0, 28, 60),
    'skin_upper': (50, 255, 255),
    'fgbg': cv2.bgsegm.createBackgroundSubtractorLSBP(),
    'kernel_open': cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
    'show_detail': False,
    'whole_scene': True,
}

# add rgb and bgr differences
args_styleTransfer["mask_rgb"] = np.ones(args_styleTransfer["content_target_resize"][1:], dtype=np.bool)
args_styleTransfer["mask_rgb"][:, :args_styleTransfer["mask_rgb"].shape[1]//2, :] = 0

# add sess
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
sess = tf.Session(config=config)
args_styleTransfer["sess"] = sess

# add X, Y
with tf.variable_scope('img_t_net'):
    args_styleTransfer["X"] = tf.placeholder(
        tf.float32,
        shape=args_styleTransfer["content_target_resize"],
        name='input'
    )
    args_styleTransfer["Y"] = create_net(
        args_styleTransfer["X"],
        args_styleTransfer['upsample_method']
    )
# add model_saver
model_saver = tf.train.Saver()
model_saver.restore(args_styleTransfer["sess"], args_styleTransfer['model_path'])


# arguments for wearing glasses
args_glass = {
    'eye_cascade': cv2.CascadeClassifier('./models/haarcascade_eye.xml'),
    'glass_img': cv2.imread('./images/glass_image.jpg'),
}

# args for grabCut
args_grabCut = {
    "bg_capture": cv2.VideoCapture('./images/oasis'),
    "ratio": 4,
}

cv2.namedWindow('Amuse_park')
cv2.moveWindow('Amuse_park', 300, 20)
# arguments for plane_ar
args_AR = {
    'app_ar': App(),
}
