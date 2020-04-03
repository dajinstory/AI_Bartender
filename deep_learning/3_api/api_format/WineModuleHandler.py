import cv2
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility


class WineModuleHandler:
    def __init__(self):
        #self.log = {}
        self.detector = load_detector()
        #self.classifier = load_classifier()
        #self.clusterer = load_clusterer()
        #self.dataframe, self.center_vectors = self.load_dataframe()

        #Detector Variable
        self.input_shape=(300, 300, 3)
        self.voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        self.grids=[1,2,4,7]
        self.margin_rate=0.3
        self.bbox_util = BBoxUtility(len(voc_classes)+1)
        self.batch_size=1
        self.verbose=1
        self.accuracy=0.5

    # load models
    def load_detector(self):
        #config
        np.set_printoptions(suppress=True)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        set_session(tf.Session(config=config))
        # load ssd
        NUM_CLASSES = len(self.voc_classes) + 1
        detector = SSD300(self.input_shape, num_classes=NUM_CLASSES)
        detector.load_weights('..\0_detection\ssd_keras\weights_SSD300.hdf5')
        return detector

    '''
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
    '''

    def search_wines(self, filename):
        src = cv2.imread("../image/"+filename, cv2.IMREAD_COLOR)
        images = np.array([src])
        objects = get_objects(images)
        print(objects)
        #[{'x':x, 'y':y, 'len_x':len_x, 'len_y':len_y, 'label':label}]

        results=[]
        return results
        '''
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
        '''

    def divide_by_grid(self, image):
        img_w, img_h = image.shape[1:3]
        grids=self.grids
        img_resolution=self.input_shape
        margin_rate=self.margin_rate
        
        #result
        new_data=[]
        new_coord=[]

        for image in data:
            image=image.reshape(img_w, img_h, 3)
            for grid in grids:
                len_x = int(img_w/(grid-margin_rate*(grid-1)))
                len_y = int(img_h/(grid-margin_rate*(grid-1)))
                stride_x = int(len_x*(1-margin_rate))
                stride_y = int(len_y*(1-margin_rate))
                for idx_x in range(grid):
                    for idx_y in range(grid):
                        p_x = idx_x*stride_x
                        p_y = idx_y*stride_y
                        new_coord.append([ p_x ,p_y,p_x+len_x,p_y+len_y])
                        image_sliced = image[p_x:p_x+len_x, p_y:p_y+len_y:]
                        image_sliced_resized=cv2.resize(image_sliced, img_resolution)
                        new_data.append(image_sliced_resized)
        return np.array(new_data),np.array(new_coord)

    def nms(bounding_boxes, confidence_score,  labels , threshold):
        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], [], []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        #each box's labels
        label = np.array(labels)

        # Picked bounding boxes
        picked_boxes = []
        picked_score = []
        picked_label = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])
            picked_label.append(label[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]
        return picked_boxes, picked_score,picked_label

    def transform_detection_result(self,divided_images,coord):
        boxs=[]
        scores=[]
        labels=[]
        for i, img in enumerate(divided_images):
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]
            # Get detections with confidence higher than accuracy
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= accuracy]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            x_rate = (coord[i][2]-coord[i][0])/300
            y_rate = (coord[i][3]-coord[i][1])/300
            for j in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[j] * img.shape[1])*x_rate)+coord[i][1]
                ymin = int(round(top_ymin[j] * img.shape[0])*y_rate)+coord[i][0]
                xmax = int(round(top_xmax[j] * img.shape[1])*x_rate)+coord[i][1]
                ymax = int(round(top_ymax[j] * img.shape[0])*y_rate)+coord[i][0]
                score = top_conf[j]    
                label = int(top_label_indices[j])
                label_name = voc_classes[label - 1]
        
                boxs.append([xmin,ymin,xmax,ymax])
                scores.append(score)
                labels.append(label_name)
        box_nms,score_nms,label_nms=nms(boxs,scores,labels,iou_threshold)
    
        #[{'x':x, 'y':y, 'len_x':len_x, 'len_y':len_y, 'label':label}]
        result=[]
        for i in range(len(scores)):
            result.append(pack('iiiis',box_nms[i][0],box_nms[i][1],box_nms[i][2]-box_nms[i][0],box_nms[i][3]-box_nms[i][1],label_nms[i]))
        return result
    
    def get_objects(self, image):
        divided_images,coord=self.divide_by_grid(image)
        preds =self.detector.predict(divided_images, self.batch_size, self.verbose)
        results = self.bbox_util.detection_out(preds)
        # laod image and convert to 4-dimension
        return self.transform_detection_result(results,coord)
    '''
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
        '''


test = WineModuleHandler()
print(test)
#test.search_wines('1.jpg')