# Udacity - Behavioral Cloning Project
## End to End Learning for Self-Driving Car

[//]: # (Image References)

[image1]: ./images_for_writeup/data_form.png "Data Format"
[image2]: ./images_for_writeup/frame_example.png "Random image of a frame"
[image3]: ./images_for_writeup/histogram_examples.png "Histogram of Total Dataset"
[image4]: ./images_for_writeup/histogram_balanced.png "Histogram of Balanced Dataset"
[image5]: ./images_for_writeup/flipped_examples.png "Examples of Flipped images"
[image6]: ./images_for_writeup/shadow.png "Example of shadow image"

[video1]: ./video.mp4 "Video"

The goal of the project is to build an end-to-end deep learning model to train a car to drive itself in a simulator.

I followed a set a of steps that I thought could make a robust model able to generalize on different roads and conditions on which the model was not trained on. In the process I brainstormed a little and also took ideas from different blogs that I thought would be helpful in making a good model.

## Steps:
1) Understanding the Data

2) Data Collection and Balancing

3) Data preprocessing and Augmentation

4) Model Design

## Understanding the Data

The Simulator designed by Udacity has two tracks on which data can be collected and used to train a deep neural network. The whole idea is to collect data on track 1 which is an easier track and train a robust model to account for situations that can potentially be presented which have not been accounted for by track 1. The process of training included preprocessing and augmenting the dataset and then building a model.

The dataset contains two types of files, a _driving log.csv_ file and an _IMG folder_, the csv file contains information such as location of each frame(left, right and center) and the steering angle recorded for that frame. This information is used to train the model.

The steering angles range between (-1,1) with -1 corresponding to left and +1 corresponding to right and 0 corresponding to center.

### The format of the data is in the picture below:

![alt text][image1]

One of the images in the data is depicted below. 

### The image is from the left camera of a frame

![alt text][image2]


## Data Collection and Balancing

I began by collecting data on track 1 but the data I collected seemed a bit jittery because I was not able to keep the car in the middle of the track for very long. So, I decided to take the data uploaded by Udacity to train the model.

The dataset contains **8036 samples** with each sample comprising of 3 images(left, center and right camera images for each frame). This is the whole data present in the dataset. 

### The histogram of the whole dataset

![alt text][image3]

We can see from the histogram that most of the samples belong to a steering angle of 0. It means that most of the time the tires are stationary and not leaning to the left or the right. These examples constitute about 80% of the data and needs to be corrected. This brings about a situation where the trained model is biased towards predicting a steering angle of 0. To correct this we need to balance the sample across various steering angle.

To balance the data I divided the dataset into 100 bins, size of each bin is 0.02. I decided to put 200 samples in each bin. In those bins where the number of samples is less than 200 I counted all the samples present. Now, the dataset looks more balanced. The code and histogram are below:

    balanced_df = pd.DataFrame()
    n_bins = 100
    samples_per_bin = 200

    begin = -1
    for end in np.linspace(-1,1,n_bins):
    range_df = image_df[(image_df.steering >= begin) & (image_df.steering < end)] # Finding all those steering angles in that range
    num_samples = min(samples_per_bin, len(range_df))
    if (num_samples == 0):
    continue

    balanced_df = pd.concat([balanced_df,range_df.sample(num_samples)])
    begin = end

    # balanced_df.to_csv('data/driving_log_balanced.csv', index=False)

    print ('The number of examples in the new Balanced Dataframe are:',len(balanced_df))

### Histogram of Balanced Dataset
![alt text][image4]

## Data Preprocessing and Augmentation

After balancing the dataset I ended up with a balanced dataset with ** 3478 samples **. These samples are not enough to generalize well on different tracks as they don't account for unforseen instances. So, to make more images I created an augmentImages class in which are different methods that can create new images from the existing images. We will also be using the Left and Right camera images too.

The code for the augmentImages class is below

    class augmentImages(object):

    def __init__(self):
    self.cameras = ['left', 'center', 'right']
    self.steering_correction = [.25,0, -0.25]
    self.images = []
    self.steering_angles = []

    def flip_image_left_right(self,image):
    return np.fliplr(image)

    def flip_image_up_down(self,image):
    return np.flipud(image)

    def crop_image_resize(self,image,top_crop_range = (0.325, 0.425), bottom_crop_range = (0.075,0.175), resize_dim = (32,128,3)):
    """
    In the crop_image_resize method I am randomly selecting the a upper crop range and lower crop range and 
    then cropping the image and resizing it so that all the images are of the same dimension.
    """
    top = int(random.uniform(top_crop_range[0],top_crop_range[1]) * image.shape[0])
    bottom = int(random.uniform(bottom_crop_range[0],bottom_crop_range[1]) * image.shape[0])
    image = image[top:-bottom,:]
    image = resize(image,resize_dim) 
    return image


    def generate_shadows(self,image):
    """
    I am going to randomly select a point in the width of the image and reduce the brightness of the image until
    that point.
    The idea of adding shadows to the images to make the model more robust has been taken from Alex Staravoitau's
    blog. http://navoshta.com/
    """
    h, w = image.shape[0], image.shape[1]
    x1 = np.random.choice(w,1) # x1 here is an array. Since its an array we need to use the 0th element for comparison
    x2 = np.random.choice([0,w])
    if (x1[0] < x2):
    image[:, x1[0]:x2, :] = (image[:, x1[0]:x2, :]*.5).astype(np.uint8) # Reducing the brightness of the image
    elif (x1[0] > x2):
    image[:, x2:x1[0], :] = (image[:, x2:x1[0], :]*.5).astype(np.uint8) # Reducing the brightness of the image

    return image

As part of data augmentation, I flipped a few of the imagess randomly and a added shadows to the images so that the model can generalize well on a track that the model has not seen before. The examples for flipped and shadow examples are below.

### Flipped Images
![alt text][image5]

### Shadow Image
![alt text][image6]

Data augmentation is done in real time using the keras ** fit_generator ** method. A generator object is created which yields augmented data in chunks and this augmented data is passed to the model below as input and while model is trained on the GPU using the chunk of data another chunk of data is augmented parallely on the CPU. This is the most efficient utilization of the CPU and GPU resources.

## Model Design

An End to End deep learning model has been presented in a paper by NVIDIA. This model is very complex as it is trained on more complex data(real world data). Applying the model directly to our example might result in overfitting of the model given that we are training our model on much smaller data, this leaves scope for the network to store in its internal representation each and every example. This would give excellent results on the train data but would perform abysmally poorly on the test data in which we are really interested in. 

The architecture of the network is defined using the keras module which is a wrapper over tensorflow. It simplifies the way a neural network architecture is defined.

### The architecture is as follows:

Layer (type) |
-------------|
convolution2d_1 (Convolution2D - (16,3,3))|
MaxPooling2d_1 (MaxPooling2D - (2,2))|
convolution2d_2 (Convolution2D - (32,3,3))|
MaxPooling2d_2 (MaxPooling2D - (2,2))|
convolution2d_3 (Convolution2D - (64,3,3))|
MaxPooling2d_3 (MaxPooling2D - 2,2)|
flatten_1 (Flatten)|
Dropout_1 (Dropout - 0.5)|
dense_1 (Dense - 200)|
Activation (relu)|
Dropout_2 (Dropout - 0.25)|
dense_2 (Dense - 50)|
Activation (relu)|
Dropout_2 (Dropout - 0.25)|
dense_3 (Dense - 10)|
Activation (relu)|
dense_4 (Dense - 1)|


Adam Optimizer has been used to train the model, mean squared error loss function with learning rate of 0.0001 for 5 epochs and for 25 epochs

### Result

The car drives endlessly on the track it was trained but fails to complete the challenge track.

![alt text][video1]





