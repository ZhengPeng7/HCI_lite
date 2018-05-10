from collections import deque


# argument settings
args = {
    "video_source": 0,
    "deque_buffer": 32,
    "green_lower": (3, 103, 198),#(29, 86, 6),  # color range of ball in the HSV color space
    "green_upper": (234, 255, 255),#(64, 255, 255),
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
        "color_redaa": (0, 0, 0),
        "color_purpleaa": (255, 255, 255),
    },
    "icon_len_side": 80,
}


# video mode setting: {
# "display": Default value, of which ink would fade, with tracking effect,
# "writing": Ink would not fade,
# "gaming": Join in a shabby Arkanoid,
# "calc": 
# }
MODE = "display"

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