from ultralytics import YOLO

# Load your model
model = YOLO("model/yolov8x.pt")  # Make sure the path is correct

# Print class names
print(model.names)
