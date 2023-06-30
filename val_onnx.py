import cv2
import numpy as np
import keras_ocr
import os
import time
import easyocr
import Levenshtein
from app import save_image
from app import plate_utils
from app import utils
import glob
from datetime import datetime
import csv
import onnxruntime
from PIL import Image


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes[:1]

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# Get all paths to your images files and text files
PATH = 'datasets/'
paths = glob.glob(PATH+'validation/images/*')

opt_session = onnxruntime.SessionOptions()
opt_session.enable_mem_pattern = False
opt_session.enable_cpu_mem_arena = True
opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

model_path = 'result/train4/weights/best.onnx'
EP_list = ['CPUExecutionProvider']
CLASSES = [
	'license_plate'
]

ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

model_inputs = ort_session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
input_shape = model_inputs[0].shape

model_output = ort_session.get_outputs()
output_names = [model_output[i].name for i in range(len(model_output))]


with open('result/val_onnx_{dt}.csv'.format(dt=datetime.now()), 'w') as fp:
    # create the csv writer
    writer = csv.writer(fp)
    # header
    writer.writerow([
            'filename',
            'ground_truth',
            'plate_text',
            'similarity_score'
        ])
    
    rows = []
    
    # Initialize fps variables
    frame_count = 0
    start_time = time.time()
    count = 0

    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        ground_truth = filename.split('_')[1]

        image = cv2.imread(path)
        
        image_height, image_width = image.shape[:2]
        input_height, input_width = input_shape[2:]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (input_width, input_height))

        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

        outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
        predictions = np.squeeze(outputs).T
        
        conf_thresold = 0.7
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]
        
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Get bounding boxes for each object
        boxes = predictions[:, :4]

        #rescale box
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)
        
        indices = nms(boxes, scores, 0.7)
        
        bbox = xywh2xyxy(boxes[indices]).round().astype(np.int32)
        if len(bbox) == 0:
            continue
        
        for coords in bbox:
            plate_image = plate_utils.extract_plate(image, coords)
            to_ocr = plate_utils.recognition_preprocessing_2(plate_image)
            plate_text, confidence = plate_utils.ocr_plate1(to_ocr)
            if len(plate_text) <= 3:
                to_ocr = plate_utils.recognition_preprocessing_1(plate_image)
                plate_text, confidence = plate_utils.ocr_plate1(to_ocr)

        if ground_truth == plate_text:
            count += 1
        
        frame_count += 1

        # write a row to the csv file
        rows.append([
            filename,
            ground_truth,
            plate_text,
            utils.calculate_similarity(ground_truth, plate_text)
        ])


    end_time = time.time()
    writer.writerows(rows)  
    
# Calculate elapsed time
elapsed_time = end_time - start_time

# Calculate FPS
fps = frame_count / elapsed_time


print(f"FRAME COUNT: {frame_count}")
print(f"ELAPSED TIME: {elapsed_time}")
print(f"FPS: {fps:.2f}")

print(
    "SUCCESSFULY RECOGNIZE {count} plate".format(
            count=count
        )
    )
