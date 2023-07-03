import glob
import os
import shutil

# Get all paths to your images files and text files
PATH = '../datasets/validation/validation-set'
# img_paths = glob.glob(PATH+'labeled_images/images/*.jpg')
# txt_paths = glob.glob(PATH+'labeled_images/labels/*.txt')

paths = glob.glob(PATH+'/images/*')

idx = 1
for path in paths:
    # filename = os.path.splitext(os.path.basename(path))[0]
    old_filename = os.path.splitext(os.path.basename(path))[0]
    filename = old_filename.split('_')[1]

    src_txt_path = (PATH+'/labels/'+old_filename+'.txt')
    src_img_path = (PATH+'/images/'+old_filename+'.jpeg')
    
    dst_txt_path = (PATH+'/fix/labels/{idx}_{filename}.txt'.format(idx=idx, filename=filename))
    dst_img_path = (PATH+'/fix/images/{idx}_{filename}.jpeg'.format(idx=idx, filename=filename))
    idx += 1
    
    shutil.copy(src_img_path, dst_img_path)
    shutil.copy(src_txt_path, dst_txt_path)