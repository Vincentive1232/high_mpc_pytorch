import os
import random
import numpy as np
import torch

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


#
def set_global_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if env is not None:
        env.seed(seed)