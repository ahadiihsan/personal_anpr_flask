from ultralytics import YOLO

# Load a model
model = YOLO('models/best2.pt')  # load a custom model

# Validate the model
metrics = model.val(data='../datasets/data.yaml')
