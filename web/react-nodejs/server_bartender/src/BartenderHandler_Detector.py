import glob
import sys
import time
import cv2
import json

#sys.path.insert(0, glob.glob('C:\\Users\\DajinHan\\Anaconda3\\envs\\ai_bartender\\Lib\\*')[0])


# thrift modules
sys.path.append('../thrift_modules/gen-py')
from bartender_api import Bartender
from bartender_api.ttypes import InvalidOperation
from shared.ttypes import SharedStruct
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# Detector, Vectorizer, Classifier
from Detector import Detector




class BartenderHandler:

    def __init__(self):
        self.log = {}
        self.detector = Detector()
        self.vectorizer = 'vectorizer'
        self.classifier = 'classifier'
        self.database = 'database'



    # internal functions
    def get_objects(self, image):
        objects = self.detector.get_objects(image)
        return objects

    def get_vector(self, image):
        vector = self.vectorizer.get_vector(image)
        return vector

    def get_label(self, vector):
        label = self.classifier.get_label(vector)
        return label

    def get_wine(self, label):
        wine = self.database.get_wine(label)
        return wine

    def vector2string(self, vector):
        vector_str = "asdf"
        return vector_str



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
        src = cv2.imread(filename, cv2.IMREAD_COLOR)
        objects = self.get_objects(src)

        # convert data to string format
        for object in objects:
            for key in object.keys():
                object[key]=object[key].item()
        json_objects = json.dumps(objects)
        return json_objects

    def proto_get_vectors(self, filename):
        # load image
        src = cv2.imread(filename, cv2.IMREAD_COLOR)
        objects = self.get_objects(src)

        results = []
        for object in objects:
            if object['label'] != 'Bottle':
                continue

            # get roi portion of image and resize it
            roi = src[object['r']:object['r'] + object['len_r'], object['c']:object['c'] + object['len_c']]
            dst = cv2.resize(roi, dsize=('triplet_input_r','triplet_input_c'), interpolation=cv2.INTER_AREA)

            # get vector
            vector = self.get_vector(dst)
            object['vector'] = self.vector2string(vector)

            results.append(object)

        return results

    def proto_get_labels(self, filename):
        # load image
        src = cv2.imread(filename, cv2.IMREAD_COLOR)
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
    processor = Bartender.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=12000)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    server.serve()