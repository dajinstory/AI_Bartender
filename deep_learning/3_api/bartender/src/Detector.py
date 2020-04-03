import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import cv2

from SSD import * # 이렇게하면 바로 SSD.ssd.SSD300() 식으로 호출가능
from SSD.ssd import SSD300
from SSD.ssd_utils import BBoxUtility


class Detector:
    # init
    def __init__(self):
        self.log = {}

        # classes able to detect
        self.classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

        # variables for preprocessing input image
        self.resolution=(300, 300, 3)
        self.grids=[1]
        self.margin_rate=0.3
        self.accuracy=0.5
        self.iou_threshold=0.5

        # detector and utilities
        self.detector = self.load_detector()
        self.bbox_util = BBoxUtility(len(self.classes)+1)


    # load models
    def load_detector(self):
        # config
        np.set_printoptions(suppress=True)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        set_session(tf.Session(config=config))

        # load ssd
        NUM_CLASSES = len(self.classes) + 1
        detector = SSD300(self.resolution, num_classes=NUM_CLASSES)
        detector.load_weights('./SSD/weights/weights_SSD300.hdf5')

        return detector

    # divide image by grid
    def divide_by_grid(self, image, grids=None, margin_rate=None):
        # parameter setting with default value
        if grids is None:
            grids = self.grids
        if margin_rate is None:
            margin_rate = self.margin_rate

        img_h, img_w = image.shape[0:2]
        image = image.reshape(img_h, img_w, 3)

        # result
        new_images = []
        new_coords = []

        # divide image by grid
        for grid in grids:
            # calculate len for each sliced image and stride to next image
            len_r = int(img_h / (grid - margin_rate * (grid - 1)))
            len_c = int(img_w / (grid - margin_rate * (grid - 1)))
            stride_r = int(len_r * (1 - margin_rate))
            stride_c = int(len_c * (1 - margin_rate))

            # slice image according to current grid
            for idx_r in range(grid):
                for idx_c in range(grid):
                    # get coord of current sliced image
                    p_r = idx_r * stride_r
                    p_c = idx_c * stride_c

                    new_coords.append({
                        'r':p_r,
                        'c':p_c,
                        'len_r':len_r,
                        'len_c':len_c
                    })

                    # get current sliced image
                    image_sliced = image[p_r:p_r + len_r, p_c:p_c + len_c:]
                    image_sliced_resized = cv2.resize(image_sliced, self.resolution[:2])
                    new_images.append(image_sliced_resized)

        return np.array(new_images), np.array(new_coords)

    # use nms to get the object with highest confidence in each area
    def nms(self, boxes, scores, labels, iou_threshold=None):
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        # If no bounding boxes, return empty list
        if len(boxes) == 0:
            return [], [], []

        # Bounding boxes
        boxes = np.array(boxes)

        # coordinates of bounding boxes
        start_rs = boxes[:, 0]
        start_cs = boxes[:, 1]
        end_rs = boxes[:, 2]
        end_cs = boxes[:, 3]

        # Confidence scores of bounding boxes
        scores = np.array(scores)

        # each box's labels
        labels = np.array(labels)

        # Picked bounding boxes
        picked_boxes = []
        picked_scores = []
        picked_labels = []

        # Compute areas of bounding boxes
        areas = (end_rs - start_rs + 1) * (end_cs - start_cs + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(scores)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(boxes[index])
            picked_scores.append(scores[index])
            picked_labels.append(labels[index])

            # Compute ordinates of intersection-over-union(IOU)
            r1 = np.maximum(start_rs[index], start_rs[order[:-1]])
            r2 = np.minimum(end_rs[index], end_rs[order[:-1]])
            c1 = np.maximum(start_cs[index], start_cs[order[:-1]])
            c2 = np.minimum(end_cs[index], end_cs[order[:-1]])

            # Compute areas of intersection-over-union
            h = np.maximum(0.0, r2 - r1 + 1)
            w = np.maximum(0.0, c2 - c1 + 1)
            intersection = h * w

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < iou_threshold)
            order = order[left]
        return picked_boxes, picked_scores, picked_labels

    # converge the objects from images(sliced by grids) to REAL_OBJECTS in input shape
    def transform_detection_result(self, results, coords, accuracy=None):
        if accuracy is None:
            accuracy = self.accuracy

        boxes = []
        scores = []
        labels = []

        # for each grid cells
        for i, img in enumerate(results):
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

            # for each objects in current grid
            for j in range(top_conf.shape[0]):
                rmin = int(round(top_ymin[j] * coords[i]['len_r']) + coords[i]['r'])
                cmin = int(round(top_xmin[j] * coords[i]['len_c']) + coords[i]['c'])
                rmax = int(round(top_ymax[j] * coords[i]['len_r']) + coords[i]['r'])
                cmax = int(round(top_xmax[j] * coords[i]['len_c']) + coords[i]['c'])

                score = top_conf[j]
                label = int(top_label_indices[j])
                label_name = self.classes[label - 1]

                boxes.append([rmin, cmin, rmax, cmax])
                scores.append(score)
                labels.append(label_name)

        boxes_nms, scores_nms, labels_nms = self.nms(boxes, scores, labels)

        # get real coord of each objects
        objects = []
        for i in range(len(scores_nms)):
            objects.append({
                'r':boxes_nms[i][0],
                'c':boxes_nms[i][1],
                'len_r':boxes_nms[i][2] - boxes_nms[i][0],
                'len_c':boxes_nms[i][3] - boxes_nms[i][1],
                'label':labels_nms[i]
            })
        return objects

    # Main Function
    def get_objects(self, image):

        # divide images
        divided_images, coords = self.divide_by_grid(image)

        # get objects detected from all sliced images by grid
        batch_size = 1
        verbose = 1
        preds = self.detector.predict(divided_images, batch_size, verbose)
        results = self.bbox_util.detection_out(preds) # 0.0-1.0 position in each grid cell

        ###############
        ## IMPORTANT ##
        ###############
        # only SSD module use (x,y) - camera coords. others use (r,c) - matrix coords
        # nms objects form each slice images, and then get the real coords in input image
        return self.transform_detection_result(results, coords)


if __name__ == "__main__":
    image = cv2.imread('../../../images/3.png', cv2.IMREAD_COLOR)
    detector = Detector()
    result = detector.get_objects(image)
    print(result)