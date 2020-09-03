# import the necessary packages
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Activation
from keras import backend as k


class SallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be "channels last"
        model = Sequential()
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if k.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
