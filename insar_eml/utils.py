#imports
import numpy as np
import tensorflow as tf

def normalize(data):
    normalized_data = data / np.max(np.abs(data))
    normalized_data += 1
    return normalized_data

# Loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

# code for creating an image: disk centered at [ix, iy], with
# pixel values 1, the rest of the image has pixel values zero
def getimg(ix, iy, radius = 4, num_pixels = 40):
    # create matrices of x and y coordinates
    xx = np.arange(num_pixels)
    yy = np.arange(num_pixels)
    X, Y = np.meshgrid(xx, yy)

    return np.array((X - ix) ** 2 + (Y - iy) ** 2 <= radius ** 2, dtype=float)

# code for creating x examples: start with a disk at a given center
# (the "start" parameter specifies the initial x,y coordinates).
# get shifted versions of the disk. the total amount of shift from
# beginning to end specified by the "delta" parameter, which is a vector.

def get_shifts(start, delta, num_img = 9, radius = 4, num_pixels = 40):
    start = np.array(start, dtype=float)
    # delta1: the amount of shift in one step
    delta1 = np.array(delta, dtype=float)/(num_img-1)

    # mat will contain all the images
    mat = np.zeros([num_img, num_pixels, num_pixels])
    for i in range(num_img):
        center = start + delta1 * i
        mat[i,:,:] = getimg(ix=center[0],
                            iy=center[1],
                            radius=radius,
                            num_pixels=num_pixels)
    return mat

# for y values, we create a disk centered at the starting point of the
# x disk. this time we set the pixel values to the total amount of displacement
# of the x disk in the first coordinate direction.

# note that this y disk is not moving, this is a single image. this y image
# represents the amount of displacement of the x disk.

def get_y(start, deltax, radius = 2, num_pixels = 40):
    # note the multiplication by deltax: each pixel value in the disk represents
    # the total amount of movement in the x direction

    return getimg(ix=start[0], iy=start[1], radius=radius, num_pixels=num_pixels) * deltax

#this function will generate train_x, train_y and topology datasets.
def create_dataset(num_samples = 500, shift_mag = 8, num_instants = 9, num_pixels = 40 ):
    # select a set of random starting centers
    centers = np.random.uniform(10, 30, [num_samples, 2])

    # select a set of random shifts in the x direction
    shifts = np.random.uniform(-shift_mag, shift_mag, num_samples)

    x = np.zeros([num_samples, num_instants, num_pixels, num_pixels, 1])
    y = np.zeros([num_samples, num_pixels, num_pixels])
    topology = []

    for i in range(num_samples):
        x[i, :, :, :, 0] = get_shifts(start=centers[i, :],
                                                  delta=[shifts[i], 0],
                                                  num_img=num_instants)
        y[i, :, :] = get_y(start=centers[i, :],
                                       deltax=shifts[i],)

        new_shift = np.ones((num_pixels, num_pixels)) * shifts[i]
        topology.append(new_shift)

    topology = np.dstack(topology)
    # To get the shape to be Nx(num_pixels)x(num_pixels), use rollaxis:
    topology = np.rollaxis(topology, -1)
    topology = topology.reshape((500, 1, 40, 40, 1))

    return [x, y, topology]