import glob
import sys
import time
import cv2
import json

# load image with url type
import requests
import numpy as np
from PIL import Image
from io import BytesIO

#sys.path.insert(0, glob.glob('C:\\Users\\DajinHan\\Anaconda3\\envs\\ai_bartender\\Lib\\*')[0])


# thrift modules
sys.path.append('../thrift_modules/gen-py')
sys.path.append('../bartender_module/src')
from bartender_rmi import Bartender_rmi
from bartender_rmi.ttypes import InvalidOperation
from shared.ttypes import SharedStruct
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# Detector, Vectorizer, Classifier
# from Detector import Detector
# from Vectorizer import Vectorizer
# from Classifier import Classifier
from Bartender import Bartender


class BartenderHandler:

    def __init__(self):
        self.log = {}
        # self.detector = Detector()
        # self.vectorizer = Vectorizer()
        # self.classifier = Classifier()
        self.bartender = Bartender()
        self.database = 'database'
        self.main_server = 'http://localhost:11000/'


    # internal functions
    def get_objects(self, image):
        objects = self.bartender.get_objects(image)
        return objects

    def get_vector(self, image):
        vector = self.bartender.get_vector(image)
        return vector

    def get_label(self, vector):
        label = self.bartender.get_label(vector)
        return label

    def get_wine(self, label):
        wine = self.database.get_wine(label)
        return wine



    # functions to check thrift connection
    def ping(self):
        print('>>>ping_server')

    def test_function_string(self, input):
        print('>>> string input : '+input)
        return input

    def test_function_maplist(self, input):
        print('>>> maplist input : '+input)
        wine = {'r':'1', 'c':'2', 'len_r':'3', 'len_c':'4', 'label':'10'}
        wines = []
        for i in range(10):
            wines.append(wine)
        return wines



    # thrift api
    def proto_get_objects(self, filename):
        # load image
        response = requests.get(self.main_server+'static/images/'+filename)
        src = np.array(Image.open(BytesIO(response.content)))
        #src = cv2.imread(filename, cv2.IMREAD_COLOR)
        objects = self.get_objects(src)

        # convert data to string format
        for object in objects:
            for key in object.keys():
                # avoid the problem with np.int32 is not serializable
                object[key]=object[key].item()
        json_objects = json.dumps(objects)
        return json_objects

    def proto_get_vectors(self, filename):
        # load image
        response = requests.get(self.main_server+'static/images/'+filename)
        src = np.array(Image.open(BytesIO(response.content)))
        #src = cv2.imread(filename, cv2.IMREAD_COLOR)
        objects = self.get_objects(src)

        for object in objects:
            # get roi portion of image and resize it
            roi = src[object['r']:object['r'] + object['len_r'], object['c']:object['c'] + object['len_c']]
            dst = cv2.resize(roi, dsize=(28,28), interpolation=cv2.INTER_AREA)
            dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

            # get vector
            vector = self.get_vector(dst_gray)
            object['vector'] = vector.tolist()


        # convert data to string format
        for object in objects:
            # avoid the problem with np.int32 is not serializable
            for key in object.keys():
                if key is 'vector':
                    continue
                object[key]=object[key].item()
        json_objects = json.dumps(objects)
        return json_objects

    def proto_get_labels(self, filename):
        # load image
        response = requests.get(self.main_server+'static/images/'+filename)
        src = np.array(Image.open(BytesIO(response.content)))
        #src = cv2.imread(filename, cv2.IMREAD_COLOR)
        objects = self.get_objects(src)

        results = []
        for object in objects:
            if object['label'] == 0:
                continue

            # get roi portion of image and resize it
            roi = src[object['r']:object['r'] + object['len_r'], object['c']:object['c'] + object['len_c']]
            dst = cv2.resize(roi, dsize=('triplet_input_r', 'triplet_input_c'), interpolation=cv2.INTER_AREA)

            # get vector
            vector = self.get_vector(dst)
            object['vector'] = self.vector2string(vector)

            # get label
            label = self.get_label(vector)
            object['label'] = label

            results.append(object)

        return results

    def get_wines(self, filename):
        # load image
        src = cv2.imread(filename, cv2.IMREAD_COLOR)
        objects = self.get_objects(src)

        wines = []
        for object in objects:
            if object['label'] == 0:
                continue

            # get roi portion of image and resize it
            roi = src[object['r']:object['r'] + object['len_r'], object['c']:object['c'] + object['len_c']]
            dst = cv2.resize(roi, dsize=('triplet_input_r', 'triplet_input_c'), interpolation=cv2.INTER_AREA)

            # get vector
            vector = self.get_vector(dst)
            object['vector'] = self.vector2string(vector)

            # get label
            label = self.get_label(vector)
            object['label'] = label

            # get wine info
            wine = self.get_wine(label)
            for key in wine.keys:
                object[key]=wine[key]
            wines.append(object)

        return wines




if __name__ == '__main__':
    print(">>>main function start...")
    handler = BartenderHandler()
    processor = Bartender_rmi.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=12000)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    server.serve()