import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from sklearn.manifold import TSNE
import numpy as np
import cv2
import os

class Vectorizer:
    
    #init
    def __init__(self):
        self.img_row=28
        self.img_column=28
        self.img_channel=1
        #deep_learning/3_api/bartender/src/modules/Triplet/1.30001-0.0990-04.hdf5
        self.hdf5_path = os.path.dirname(os.path.realpath(__file__)) + './modules/Triplet/1.30001-0.0990-04.hdf5'
        self.anchor_input = Input((self.img_row, self.img_column, self.img_channel, ), name='anchor_input')
        self.Shared_DNN =self.create_base_network()
        self.encoded_anchor = self.Shared_DNN(self.anchor_input)
        self.model = Model(inputs=self.anchor_input, outputs=self.encoded_anchor)
        self.model.load_weights(self.hdf5_path)
        self.tsne = TSNE()


    def create_base_network(self):
        model = Sequential()
        model.add(Conv2D(128,(7,7),padding='same',input_shape=(self.img_row,self.img_column,self.img_channel),activation='relu',name='conv1'))
        model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
        model.add(Conv2D(256,(10,10),padding='same',activation='relu',name='conv2'))
        model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(4,name='embeddings'))
        return model

    
    #image shape(1,28,28.1)
    def get_vector(self, image):
        image_resized = cv2.resize(np.array(image), (28,28), interpolation = cv2.INTER_AREA )

        # check if grayscale
        if len(image_resized.shape)==3 and image_resized.shape[2]==3:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        trm = self.model.predict(image_resized.reshape(-1,28,28,1))
        return np.array(trm)
