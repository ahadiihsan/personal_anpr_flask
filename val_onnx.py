import cv2
import numpy as np
import keras_ocr
import os
import time
import easyocr
import Levenshtein
from app import config
from app import plate_utils
from app import utils
import glob
from datetime import datetime
import csv
import onnxruntime
from PIL import Image
import pytesseract

# setting up tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.3.1_1/bin/tesseract'

def ocr_plate3(plate_image):

    # recognizing text
    config = '-l eng --oem 1 --psm 7'
    text = pytesseract.image_to_string(plate_image, config=config)

    return text


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
            'similarity_score',
            'elapsed_time_detection',
            'elapsed_time_extraction',
            'elapsed_time_preprocessing_1',
            'elapsed_time_ocr_1',
            'elapsed_time_preprocessing_2',
            'elapsed_time_ocr_2',
            'elapsed_time_per_frame'
        ])
    
    rows = []
    
    elapsed_time_detection = 0
    elapsed_time_extraction = 0
    elapsed_time_preprocessing_1 = 0
    elapsed_time_ocr_1 = 0
    elapsed_time_preprocessing_2 = 0
    elapsed_time_ocr_2 = 0
    elapsed_time_per_frame = 0
    
    # Initialize fps variables
    frame_count = 0
    start_time = time.time()
    count = 0

    for path in paths:
        start_time_frame = time.time()
        filename = os.path.splitext(os.path.basename(path))[0]
        ground_truth = filename.split('_')[1]

        image = cv2.imread(path)
        
        start_time_detection = time.time()
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
        
        indices = utils.nms(boxes, scores, 0.7)
        
        bbox = utils.xywh2xyxy(boxes[indices]).round().astype(np.int32)
        
        end_time_detection = time.time()
        elapsed_time_detection = end_time_detection - start_time_detection
        print(f"ELAPSED TIME PER DETECTION: {elapsed_time_detection}")
        
        if len(bbox) == 0:
            continue
        
        for coords in bbox:
            start_time_extraction = time.time()
            plate_image = plate_utils.extract_plate(image, coords)
            end_time_extraction = time.time()
            elapsed_time_extraction = end_time_extraction - start_time_extraction
            print(f"ELAPSED TIME EXTRACT PLATE: {elapsed_time_extraction}")
            # to_ocr = plate_image
            
            start_time_preprocessing_1 = time.time()
            to_ocr = plate_utils.recognition_preprocessing_2(plate_image)
            end_time_preprocessing_1 = time.time()
            elapsed_time_preprocessing_1 = end_time_preprocessing_1 - start_time_preprocessing_1
            print(f"ELAPSED TIME PREPROCESSING 1: {elapsed_time_preprocessing_1}")

            start_time_ocr_1 = time.time()
            plate_text, confidence = plate_utils.ocr_plate1(to_ocr)
            end_time_ocr_plate1 = time.time()
            elapsed_time_ocr_1 = end_time_ocr_plate1 - start_time_ocr_1
            print(f"ELAPSED TIME OCR 1: {elapsed_time_ocr_1}")
            
            # start_time_ocr_3 = time.time()
            # plate_text = ocr_plate3(to_ocr)
            # end_time_ocr_plate3 = time.time()
            # elapsed_time_ocr_3 = end_time_ocr_plate3 - start_time_ocr_3
            # print(f"ELAPSED TIME OCR 3: {elapsed_time_ocr_3}")
            
            if len(plate_text) <= 3:
                start_time_preprocessing_2 = time.time()
                to_ocr = plate_utils.recognition_preprocessing_1(plate_image)
                end_time_preprocessing_2 = time.time()
                elapsed_time_preprocessing_2 = end_time_preprocessing_2 - start_time_preprocessing_2
                print(f"ELAPSED TIME PREPROCESSING 2: {elapsed_time_preprocessing_2}")

                start_time_ocr_2 = time.time()
                plate_text, confidence = plate_utils.ocr_plate1(to_ocr)
                end_time_ocr_plate2 = time.time()
                elapsed_time_ocr_2 = end_time_ocr_plate2 - start_time_ocr_2
                print(f"ELAPSED TIME OCR 2: {elapsed_time_ocr_2}")

        end_time_frame = time.time()
        elapsed_time_per_frame = end_time_frame - start_time_frame
        print(f"ELAPSED TIME PER FRAME: {elapsed_time_per_frame}")
        
        if ground_truth == plate_text:
            count += 1
        
        frame_count += 1

        # write a row to the csv file
        rows.append([
            filename,
            ground_truth,
            plate_text,
            utils.calculate_similarity(ground_truth, plate_text),
            elapsed_time_detection,
            elapsed_time_extraction,
            elapsed_time_preprocessing_1,
            elapsed_time_ocr_1,
            elapsed_time_preprocessing_2,
            elapsed_time_ocr_2,
            elapsed_time_per_frame
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
