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

# Get all paths to your images files and text files
PATH = 'datasets/'
paths = glob.glob(PATH+'validation/plate/*')

with open('result/val_ocr_{dt}.csv'.format(dt=datetime.now()), 'w') as fp:
    # create the csv writer
    writer = csv.writer(fp)
    # header
    writer.writerow([
            'filename',
            'ground_truth',
            'plate_text',
            'similarity_score'
        ])
    
    # Initialize fps variables
    frame_count = 0
    start_time = time.time()
    count = 0

    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        ground_truth = filename

        img = cv2.imread(path)
        plate_text, confidence = plate_utils.ocr_plate1(
                img
            )

        if ground_truth == plate_text:
            count += 1
        
        frame_count += 1

        # write a row to the csv file
        writer.writerow([
            filename,
            ground_truth,
            plate_text,
            utils.calculate_similarity(ground_truth, plate_text)
        ])

# Calculate elapsed time
end_time = time.time()
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


