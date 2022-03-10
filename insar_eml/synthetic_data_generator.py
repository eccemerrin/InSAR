#imports
import numpy as np
from insar_eml import models
from insar_eml import metrics

# default values we will use below
NUM_SAMPLES  = 100
NUM_INSTANTS = 9
NUM_PIXELS   = 40

# default radius of "x" disk
RADIUS = 4
# default radius of "y" disk
Y_RADIUS = 2


# code for creating an image: disk centered at [ix, iy], with
# pixel values 1, the rest of the image has pixel values zero
def getimg(ix, iy, radius=RADIUS, num_pixels=NUM_PIXELS):
    # create matrices of x and y coordinates
    xx = np.arange(num_pixels)
    yy = np.arange(num_pixels)
    X, Y = np.meshgrid(xx, yy)

    return np.array((X - ix) ** 2 + (Y - iy) ** 2 <= radius ** 2, dtype=float)

# code for creating x examples: start with a disk at a given center
# (the "start" parameter specifies the initial x,y coordinates).
# get shifted versions of the disk. the total amount of shift from
# beginning to end specified by the "delta" parameter, which is a vector.

def get_shifts(start, delta, num_img=9, radius= RADIUS, num_pixels=NUM_PIXELS):
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

def get_y(start, deltax, radius=Y_RADIUS, num_pixels=NUM_PIXELS):
    # note the multiplication by deltax: each pixel value in the disk represents
    # the total amount of movement in the x direction

    return getimg(ix=start[0], iy=start[1], radius=radius, num_pixels=num_pixels) * deltax


num_samples = 500
shift_mag = 8
num_instants = 9
num_pixels = NUM_PIXELS
num_test_samples = 100


# select a set of random starting centers
centers = np.random.uniform(10,30,[num_samples, 2])

# select a set of random shifts in the x direction
shifts = np.random.uniform(-shift_mag, shift_mag, num_samples)


# the final "1" is for the number of channels
train_x = np.zeros([num_samples, num_instants,num_pixels, num_pixels, 1])
train_y = np.zeros([num_samples, num_pixels, num_pixels])

for i in range(num_samples):
    train_x[i, :, :, :, 0] = get_shifts(start   = centers[i,:],
                                        delta   = [shifts[i], 0],
                                        num_img = num_instants,
                                        radius  = RADIUS,
                                        num_pixels = NUM_PIXELS)
    train_y[i, :, :]       = get_y(start  = centers[i,:],
                                   deltax = shifts[i],
                                   radius = Y_RADIUS,
                                   num_pixels = NUM_PIXELS)


test_x = np.zeros([num_samples, num_instants,num_pixels, num_pixels, 1])
test_y = np.zeros([num_samples, num_pixels, num_pixels])

for i in range(num_test_samples):
    test_x[i, :, :, :, 0] = get_shifts(start   = centers[i,:],
                                        delta   = [shifts[i], 0],
                                        num_img = num_instants,
                                        radius  = RADIUS,
                                        num_pixels = NUM_PIXELS)
    test_y[i, :, :]       = get_y(start  = centers[i,:],
                                   deltax = shifts[i],
                                   radius = Y_RADIUS,
                                   num_pixels = NUM_PIXELS)


topo = []

for i in range(0, num_samples):
    new_shift = np.ones((40,40)) * shifts[i]

    topo.append(new_shift)

topo = np.dstack(topo)


# To get the shape to be Nx40x40, you could  use rollaxis:
topo = np.rollaxis(topo,-1)
topo = topo.reshape((500, 1, 40, 40, 1))


test_topo = []
for i in range(0, num_samples):
    new_shift = np.ones((40,40)) * shifts[i]

    test_topo.append(new_shift)

test_topo = np.dstack(test_topo)


# To get the shape to be Nx40x40, you could  use rollaxis:
test_topo = np.rollaxis(test_topo, -1)
test_topo = test_topo.reshape((500, 1, 40, 40, 1))

model = models.insar_model()

normalized_train_x = metrics.normalize(train_x)
normalized_train_y = metrics.normalize(train_y )

model.summary()
model.fit(normalized_train_x, normalized_train_y, epochs =5)
