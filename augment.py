import numpy as np
import random
from skimage.transform import resize


class augmentImages(object):
    def __init__(self):
        self.cameras = ['left', 'center', 'right']
        self.steering_correction = [.25, 0, -0.25]
        self.images = []
        self.steering_angles = []

    def flip_image_left_right(self, image):
        return np.fliplr(image)

    def flip_image_up_down(self, image):
        return np.flipud(image)

    def crop_image_resize(self, image, top_crop_range=(0.325, 0.425), bottom_crop_range=(0.075, 0.175),
                          resize_dim=(32, 128, 3)):
        """
        In the crop_image_resize method I am randomly selecting the a upper crop range and lower crop range and
        then cropping the image and resizing it so that all the images are of the same dimension.
        """
        top = int(random.uniform(top_crop_range[0], top_crop_range[1]) * image.shape[0])
        bottom = int(random.uniform(bottom_crop_range[0], bottom_crop_range[1]) * image.shape[0])
        image = image[top:-bottom, :]
        image = resize(image,resize_dim)
        return image

    def generate_shadows(self, image):
        """
        I am going to randomly select a point in the width of the image and reduce the brightness of the image until
        that point.
        The idea of adding shadows to the images to make the model more robust has been taken from Alex Staravoitau's
        blog. http://navoshta.com/
        """
        h, w = image.shape[0], image.shape[1]
        x1 = np.random.choice(w,
                              1)  # x1 here is an array. Since its an array we need to use the oth element for comparison
        x2 = np.random.choice([0, w])
        if (x1[0] < x2):
            image[:, x1[0]:x2, :] = (image[:, x1[0]:x2, :] * .5).astype(
                np.uint8)  # Reducing the brightness of the image
        elif (x1[0] > x2):
            image[:, x2:x1[0], :] = (image[:, x2:x1[0], :] * .5).astype(
                np.uint8)  # Reducing the brightness of the image

        return image