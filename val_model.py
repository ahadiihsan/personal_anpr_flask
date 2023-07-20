from ultralytics import YOLO
import time

# Load a model
model = YOLO('./result/train4/weights/best.pt')  # load a custom model

# Validate the model
start_time = time.time()
metrics = model.val(data='datasets/data.yaml', device='cpu')
# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"ELAPSED TIME: {elapsed_time}")