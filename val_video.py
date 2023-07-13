from app import plate_utils
import os
from datetime import datetime
import csv


dt = datetime.now()

source = 'datasets/example/1689101739608727.mp4'
print("file exists?", os.path.exists(source))
# source = 0

with open('result/val_video_{dt}.csv'.format(dt=dt), 'w') as fp:
    # create the csv writer
    writer = csv.writer(fp)
    # header
    writer.writerow([
            'track_id',
            'plate_text',
            'confidence'
        ])

    rows = plate_utils.get_plates_from_video(source)

    # write rows to the csv file
    writer.writerows(rows)
