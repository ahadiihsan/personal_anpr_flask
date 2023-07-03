import cv2
import numpy as np
import os
import time
import glob
from datetime import datetime
import csv

# Get all paths to your images files and text files
paths = glob.glob(r'/Users/ahadiihsanrasyidin/Projects/final-project/personal_anpr_flask/datasets/validation/images/*')

with open('dataset_shape_{dt}.csv'.format(dt=datetime.now()), 'w') as fp:
    # create the csv writer
    writer = csv.writer(fp)
    # header
    writer.writerow([
            'filename',
            'height',
            'width',
            'dimension'
        ])

    counter = []
    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]

        img = cv2.imread(path)
        h, w, c = img.shape
        dimension = f'{h}x{w}'
        meta = [dimension, 1]
        if dimension not in list(
                    set(item[0] for item in counter)
                ):
            counter.append(meta)
        else:
            for item in counter:
                if item[0] == dimension:
                    # Check if the ocr confidenence is maximum or not
                    item[1] += 1
                    break

        # write a row to the csv file
        writer.writerow([
            filename,
            h,
            w,
            dimension
        ])
    
    writer.writerows(counter)



