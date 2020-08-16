import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

KERNEL_DICT = {
    'normal': np.array([
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    ]),
    'gaussianBlur': np.array([
        0.045, 0.122, 0.045,
        0.122, 0.332, 0.122,
        0.045, 0.122, 0.045
    ]),
    'gaussianBlur2': np.array([
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    ]),
    'gaussianBlur3': np.array([
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    ]),
    'unsharpen': np.array([
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    ]),
    'sharpness': np.array([
        0,-1, 0,
        -1, 5,-1,
        0,-1, 0
    ]),
    'sharpen': np.array([
        -1, -1, -1,
        -1, 16, -1,
        -1, -1, -1
    ]),
    'edgeDetect': np.array([
        -0.125, -0.125, -0.125,
        -0.125,  1,     -0.125,
        -0.125, -0.125, -0.125
    ]),
    'edgeDetect2': np.array([
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    ]),
    'edgeDetect3': np.array([
        -5, 0, 0,
        0, 0, 0,
        0, 0, 5
    ]),
    'edgeDetect4': np.array([
        -1, -1, -1,
        0,  0,  0,
        1,  1,  1
    ]),
    'edgeDetect5': np.array([
        -1, -1, -1,
        2,  2,  2,
        -1, -1, -1
    ]),
    'edgeDetect6': np.array([
        -5, -5, -5,
        -5, 39, -5,
        -5, -5, -5
    ]),
    'sobelHorizontal': np.array([
        1,  2,  1,
        0,  0,  0,
        -1, -2, -1
    ]),
    'sobelVertical': np.array([
        1,  0, -1,
        2,  0, -2,
        1,  0, -1
    ]),
    'previtHorizontal': np.array([
        1,  1,  1,
        0,  0,  0,
        -1, -1, -1
    ]),
    'previtVertical': np.array([
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
    ]),
    'boxBlur': np.array([
        0.111, 0.111, 0.111,
        0.111, 0.111, 0.111,
        0.111, 0.111, 0.111
    ]),
    'triangleBlur': np.array([
        0.0625, 0.125, 0.0625,
        0.125,  0.25,  0.125,
        0.0625, 0.125, 0.0625
    ]),
    'emboss': np.array([
        -2, -1,  0,
        -1,  1,  1,
        0,  1,  2
    ]),
    'sepia' : np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
}

KERNEL = dotdict(KERNEL_DICT)