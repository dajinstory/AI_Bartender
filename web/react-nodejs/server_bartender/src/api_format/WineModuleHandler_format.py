class WineModuleHandler:

    def __init__(self):
        self.log = {}
        self.detector = load_models()
        self.classifier = load_classifier()
        self.clusterer = load_clusterer()
        self.dataframe, self.center_vectors = self.load_dataframe()

    # load models
    def load_detector(self):
        # load ssd
        detector = Model(inputs=anchor_input, outputs=anchor_output)
        detector.load_weights('models/detector.h5')
        return detector

    def load_classifier(self):
        # load triplet loss
        classifier = Model(inputs=anchor_input, outputs=anchor_output)
        classifier.load_weights('models/classifier.h5')
        return classifier

    def load_clusterer(self):
        # load annoy
        dimension=4
        clusterer=AnnoyIndex(dimension,'euclidean')
        clusterer.load('models/clusterer.ann')
        return clusterer


    def search_wines(self, filename):
        src = cv2.imread("../image/"+filename, cv2.IMREAD_COLOR)
        objects = get_objects(src)
        #[{'x':x, 'y':y, 'len_x':len_x, 'len_y':len_y, 'label':label}]

        results=[]
        for object in objects:
            if object['label']==0:
                continue

            # get roi portion of image and resize it
            roi = src[object['x']:object['x']+object['len_x'], object['y']:object['y']+object['len_y']]
            dst = cv2.resize(roi, dsize=(500, 500), interpolation=cv2.INTER_AREA)

            # get vector and label
            vector = get_vector(dst)
            label = get_label(vector)
            object['label']=label
            results.append(object)

         return results

    def get_objects(self, image):
        # laod image and convert to 4-dimension
        objects = self.detector.predict(image)
        return objects

    def get_vector(self, image):
        vector = self.classifier.predict(image)
        return vector

    def get_label(self, vector):
        start_time=time.time()
        nns_num=20
        nns = self.clusterer.get_nns_by_vector(vector, nns_num)
        label = self.data_frame[nns].value_counts().argmax()
        end_time=time.time()
        print(">>>get_label : ",label)
        print(">>>get_label : total_time : ", end_time-start_time)
        return label

    def ping(self):
        print('>>>ping_server')
