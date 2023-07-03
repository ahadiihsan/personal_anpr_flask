from ultralytics import YOLO
import time

# Load a model
model = YOLO('./result/train6/weights/best.pt')  # load a custom model

# Validate the model
start_time = time.time()
metrics = model.val(data=r'/Users/ahadiihsanrasyidin/Projects/final-project/PrototypeV8/yolov8/datasets-2/data.yaml', device='cpu')
# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"ELAPSED TIME: {elapsed_time}")