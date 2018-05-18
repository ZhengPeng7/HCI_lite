from collections import deque
import tensorflow as tf
import numpy as np
import cv2
from im_transf_net import create_net


# video mode setting: {
# "display": Default value, of which ink would fade, with tracking effect,
# "writing": Ink would not fade,
# "gaming": Join in a shabby Arkanoid,
# "calc": Do math evaluation from handwritten formula by OCR tech.
#       "evaluation": evaluate the result of handwritten formula.
# "glass": Help you wear a pair of glasses.
# }
MODE = "glass"

# argument settings
args = {
    "video_source": 0,
    "deque_buffer": 32,
    "orange_lower": (29, 86, 6),#(3, 103, 178),  # color range of ball in the HSV color space
    "orange_upper": (64, 255, 255),#(234, 255, 255),
}

# menu_top settings
# order of items: right -> left
args_menu = {
    "menu_dict": {
        "color_blue": (253, 1, 1),
        "color_green": (1, 253, 1),
        "color_red": (1, 1, 253),
        "color_purple": (161, 0, 161),
        "plus_icon": "images/plus_icon.png",
        "minus_icon": "images/minus_icon.png",
        "color_redaa": "images/theStarryNight.jpg",
        "color_purpleaa": "images/calculator.jpg",
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
    'waiting_sec': 5,
    'whole_scene': True,
}

# add rgb and bgr differences
args_styleTransfer["mask_rgb"] = np.ones(args_styleTransfer["content_target_resize"][1:], dtype=np.bool)
args_styleTransfer["mask_rgb"][:, :args_styleTransfer["mask_rgb"].shape[1]//2, :] = 0

# add sess
sess = tf.Session()
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

# arguments for calc
args_calc = {
    'font_color': (151, 0, 64),
    'font_thickness': 3,
    'font_style': cv2.FONT_HERSHEY_SIMPLEX,
    "orange_lower": (29, 86, 6),        # (3, 103, 178),
    "orange_upper": (64, 255, 255),     # (234, 255, 255),
    'pts': args_display['pts'].copy(),
    'thick_coeff': 2.5,
    'deque_buffer': 32,
    'res': '',    # 10,
    'kernel_open': cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
}

# arguments for wearing glasses
args_glass = {
    'eye_cascade': cv2.CascadeClassifier('./models/haarcascade_eye.xml'),
    'glass_img': cv2.imread('./images/glass_image.jpg'),
}
