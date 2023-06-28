import cv2
from app import plate_utils
from app import utils
import glob
import os
from datetime import datetime
import csv
import time


# Get all paths to your images files and text files
PATH = 'datasets/'

paths = glob.glob(PATH+'validation/images/*')

ground_truths = []
results = []
count = 0
dt = datetime.now()

with open('result/val_{dt}.csv'.format(dt=dt), 'w') as fp:
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

    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        ground_truth = filename.split('_')[1]
        ground_truths.append(ground_truth)

        img = cv2.imread(path)
        detected_image, plate_text = plate_utils.get_plates_from_image(
                img,
                filename
            )

        results.append(plate_text)

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
