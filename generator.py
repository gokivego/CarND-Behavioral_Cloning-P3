from sklearn.utils import shuffle
from augment import *
import matplotlib.image as mpimg


class dataGenerator(object):
    def __init__(self, data_dir, augment=True):

        self.__aI = augmentImages()
        self.data_dir = data_dir  # This is the directory where all the data resides
        self.augment = augment

    def dataGen(self, data):
        while True:
            # Shuffling the train data
            data = shuffle(data)
            data_indices = data.index.tolist()
            batch_size = 128
            for offset in range(0, len(data), batch_size):
                # Output arrays
                images = np.empty([0, 32, 128, 3], dtype=np.float32)
                steering = np.empty([0], dtype=np.float32)
                for idx in data_indices[offset:offset + batch_size]:
                    if self.augment:
                        i = np.random.randint(len(self.__aI.cameras))
                        path = data[self.__aI.cameras[i]][idx]
                        file_name = path.split('/')[-1]
                        local_path = self.data_dir + file_name
                        img = mpimg.imread(local_path)
                        steer_angle = data['steering'][idx] + self.__aI.steering_correction[i]

                        if np.random.choice([True, False]):
                            img = self.__aI.generate_shadows(img)  # Generating the shadows

                        img = self.__aI.crop_image_resize(img)
                        images = np.append(images, [img], axis=0)  # Appending these to the images dataset
                        steering = np.append(steering, [steer_angle])  # Appending the steering angles to the dataset

                        img = self.__aI.flip_image_left_right(img)
                        steer_angle = (-1 * steer_angle)

                        images = np.append(images, [img], axis=0)
                        steering = np.append(steering, [steer_angle])

                yield (images, steering)

    def validationGen(self, data):
        while True:
            data = shuffle(data)
            data_indices = data.index.tolist()
            batch_size = 128
            for offset in range(0, len(data), batch_size):
                # Output arrays
                images = np.empty([0, 32, 128, 3], dtype=np.float32)
                steering = np.empty([0], dtype=np.float32)
                for idx in data_indices[offset:offset + batch_size]:
                    i = 1
                    path = data[self.__aI.cameras[i]][idx]
                    file_name = path.split('/')[-1]
                    local_path = self.data_dir + file_name
                    img = mpimg.imread(local_path)
                    steer_angle = data['steering'][idx] + self.__aI.steering_correction[i]

                    # Cropping the image to match the input that the convolution takes
                    img = self.__aI.crop_image_resize(img)
                    images = np.append(images, [img], axis=0)
                    steering = np.append(steering, [steer_angle])

                yield (images, steering)