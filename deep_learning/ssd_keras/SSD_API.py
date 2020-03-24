import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os

from ssd import SSD300
from ssd_utils import BBoxUtility

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

grids=[1,2]

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

# data(컬러,w,h)
def divide_by_grid(data, img_resolution=(300,300), grids=[1,2,4,7], margin_rate=0.3):
    img_w, img_h = data.shape[1:3]
    #print(data.shape)
    new_data=[]
    new_coord=[]
    new_targets=[]
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
        return [], []

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


def single_shot_multibox_detector(image, accuracy=0.5,iou_threshold=0.5):
    #by grid
    divided_images,coord = divide_by_grid(images, grids=grids)
    
    #image detection
    preds = model.predict(divided_images, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)
    
    boxs=[]
    scores=[]
    labels=[]

    for i, img in enumerate(divided_images):
        # Parse the outputs.
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
    
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        currentAxis = plt.gca()
    
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
    selected_boxes,selected_scores,selected_labels=nms(boxs,scores,labels,iou_threshold)
    '''
    selected_indices = tf.image.non_max_suppression(boxs, scores, len(scores), iou_threshold=iou_threshold)
    selected_boxes = tf.gather(boxs, selected_indices)
    sess =tf.Session()
    value=sess.run(selected_boxes)
    '''

    #print(value.shape())
    return selected_boxes,selected_scores,selected_labels


path = './pics/1.jpg'
img = cv2.imread(path,cv2.IMREAD_COLOR)
images = np.array([img])
selected_boxes,selected_scores,selected_labels=single_shot_multibox_detector(images)
print(np.array(selected_boxes).shape)
print(np.array(selected_scores).shape)
print(np.array(selected_labels).shape)
