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


# Get all paths to your images files and text files
PATH = 'datasets/'
paths = glob.glob(PATH+'validation/validation-set/images/*')


with open('result/val_tensorflow_{dt}.csv'.format(dt=datetime.now()), 'w') as fp:
    # create the csv writer
    writer = csv.writer(fp)
    # header
    writer.writerow([
            'no',
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
        no = filename.split('_')[0]

        image = cv2.imread(path)
        
        start_time_detection = time.time()
        bbox, det_confidences = plate_utils.detect_plate(image)
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
            plate_text, confidence = plate_utils.ocr_plate(to_ocr)
            end_time_ocr_plate1 = time.time()
            elapsed_time_ocr_1 = end_time_ocr_plate1 - start_time_ocr_1
            print(f"ELAPSED TIME OCR 1: {elapsed_time_ocr_1}")
            
            # if len(plate_text) <= 3:
            #     start_time_preprocessing_2 = time.time()
            #     to_ocr = plate_utils.recognition_preprocessing_1(plate_image)
            #     end_time_preprocessing_2 = time.time()
            #     elapsed_time_preprocessing_2 = end_time_preprocessing_2 - start_time_preprocessing_2
            #     print(f"ELAPSED TIME PREPROCESSING 2: {elapsed_time_preprocessing_2}")

            #     start_time_ocr_2 = time.time()
            #     plate_text, confidence = plate_utils.ocr_plate(to_ocr)
            #     end_time_ocr_plate2 = time.time()
            #     elapsed_time_ocr_2 = end_time_ocr_plate2 - start_time_ocr_2
            #     print(f"ELAPSED TIME OCR 2: {elapsed_time_ocr_2}")

        end_time_frame = time.time()
        elapsed_time_per_frame = end_time_frame - start_time_frame
        print(f"ELAPSED TIME PER FRAME: {elapsed_time_per_frame}")
        
        if ground_truth == plate_text:
            count += 1
        
        frame_count += 1

        # write a row to the csv file
        rows.append([
            no,
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
