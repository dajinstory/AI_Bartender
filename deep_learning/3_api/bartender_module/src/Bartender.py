from Detector import Detector
from Vectorizer import Vectorizer
import cv2

class Bartender:

    def __init__(self):
        self.log = {}
        self.detector = Detector()
        self.vectorizer = Vectorizer()
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
        return objects

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
    bartender = Bartender()
    result = bartender.proto_get_objects('../../../images/4.png')
    print(result)
    # result = bartender.proto_get_vectors('../../../images/4.png')
    # print(result)
    # result = bartender.proto_get_labels('../../../images/4.png')
    # print(result)