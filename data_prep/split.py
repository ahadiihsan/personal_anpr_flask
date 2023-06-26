import glob
# import random
import os
import csv
# import shutil

# Get all paths to your images files and text files
PATH = '../../datasets'
# img_paths = glob.glob(PATH+'labeled_images/images/*.jpg')
# txt_paths = glob.glob(PATH+'labeled_images/labels/*.txt')

paths = glob.glob(PATH+'/res_path_v2/*')

with open('res_path_v.csv', 'w') as fp:
    for path in paths:
        # filename = os.path.splitext(os.path.basename(path))[0]
        filename = os.path.splitext(os.path.basename(path))[0]

        # create the csv writer
        writer = csv.writer(fp)

        # write a row to the csv file
        writer.writerow(filename.split())
