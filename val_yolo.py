from ultralytics import YOLO
import glob
import os
import time
import cv2


# Get all paths to your images files and text files
PATH = 'datasets/'

paths = glob.glob(PATH+'validation/images/*')
model = YOLO('models/best2.pt')  # load a custom model

images = []
# for path in paths:
#     img = cv2.imread(path)
#     images.append(path)
# Initialize fps variables
frame_count = len(paths)
start_time = time.time()

# Load a model

model(
    paths,
    augment=True,
    iou=0.7,
    half=True,
    device='mps',
    conf=0.7,
    agnostic_nms=True,
)

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate FPS
fps = frame_count / elapsed_time

print(f"FRAME COUNT: {frame_count}")
print(f"ELAPSED TIME: {elapsed_time}")
print(f"FPS: {fps:.2f}")

