import cv2
from app import plate_utils
from app import save_image
import glob
import os


# Get all paths to your images files and text files
PATH = '../datasets/'

paths = glob.glob(PATH+'park/train/images/*')

results = []
count = 1

for path in paths:
    ground_truth = os.path.splitext(os.path.basename(path))[0]

    img = cv2.imread(path)
    detected_image, plate_text = plate_utils.get_plates_from_image(
            img,
            ""
        )

    if save_image: cv2.imwrite(
            "./image/res_path_park_v2/{idx} {res} {plate}.jpeg".format(
                    idx=count,
                    res=plate_text,
                    plate=ground_truth
                ),
            img
        )

    count += 1
