import numpy as np

def img_slice(arr, dim):
    """
    turns an array arr into a list of dim x dim blocks.
    Remark: input array should have dimensions which are multiples of dim.
    """
    total_nb_of_blocks = arr.size // (dim ** 2)
    sliced = np.zeros((total_nb_of_blocks, dim, dim))
    max_x, max_y = arr.shape
    max_x, max_y = max_x // dim, max_y // dim
    i = 0
    for x_block_coord in range(max_x):
        x_px = dim * x_block_coord
        for y_block_coord in range(max_y):
            y_px = dim * y_block_coord
            sliced[i] = arr[x_px:x_px+dim, y_px:y_px+dim].copy()
            i += 1
            
    return sliced
