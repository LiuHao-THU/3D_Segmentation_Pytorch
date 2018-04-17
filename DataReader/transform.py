#basic operation flip transpose random crop 
import random
import numpy as np
from scipy.ndimage.interpolation import rotate

def random_boolean():
	random_bool = np.random.choice([True, False])
	return random_bool

def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis

def Flip_image(image):
	n_dim = len(image.shape)
	flip_axis = random_flip_dimensions(n_dim)
	try:
		new_data = np.copy(image)
		for axis_index in flip_axis:
			new_data = np.flip(new_data, axis=axis_index)

	except TypeError:
		new_data = np.flip(image, axis=flip_axis)

	return new_data

def Random_crop(image, label, crop_size):
	"""generate random croped cube according to 
		crop_size = [crop_size_x,...crop_size_z]"""
	w,h,z = image.shape

	random_x_l = w - crop_size[0]
	random_y_l = h - crop_size[1]
	random_z_l = z - crop_size[2]
	#generate random nums
	random_x = random.randint(0, random_x_l - 1)
	random_y = random.randint(0, random_y_l - 1)
	random_z = random.randint(0, random_z_l - 1)
	croped_image = image[random_x: random_x + crop_size[0], 
							random_y: random_y + crop_size[1],
							random_z: random_z + crop_size[2]]
	croped_label = label[random_x: random_x + crop_size[0],
							random_y: random_y + crop_size[1],
							random_z: random_z + crop_size[2]]							
	return croped_image, croped_label