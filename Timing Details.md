## To get Timing details : 

## use command : yolo task=detect mode=predict model=yolov8n.pt source=data/images/example_image.jpg verbose=True

## OUTOUT
image 1/1 C:\Users\Firmusoft\Documents\AI\PPE-Detection-YOLO\data\images\example_image.jpg: 448x640 5 persons, 237.9ms
Speed: 11.0ms preprocess, 237.9ms inference, 15.0ms postprocess per image at shape (1, 3, 448, 640) 

model summary
--------------

1. 168 layers: The model consists of 168 layers in its neural network architecture. These include convolutional, activation, normalization, and other layers that process the image.
2. 3,151,904 parameters: The model has 3.15 million trainable parameters, which determine its ability to learn features and perform object detection.
3. 0 gradients: This likely indicates that the model is in inference mode, where no gradients are computed because backpropagation is not required.
4. 8.7 GFLOPs: The model performs 8.7 billion floating-point operations per second, a measure of its computational cost. Lower GFLOPs mean better efficiency.

Image Processing Breakdown
---------------------------

For the image example_image.jpg (size reshaped to 448x640 pixels):

1. 5 persons detected: The model identified 5 instances of the class "person" in the image.
2. Total inference time: 237.9 milliseconds (ms) for running the detection on this single image.

Detailed Speed Breakdown
-------------------------

1. Preprocessing (11.0ms): Time taken to resize, normalize, and prepare the input image for inference.
2. Inference (237.9ms): The main detection time where the model processes the image and identifies objects.
3. Postprocessing (15.0ms): Time spent on tasks like applying non-max suppression (NMS) to filter overlapping bounding boxes, and formatting the results.

Performance Metrics
-------------------

1. Resolution: The input image was resized to 448x640 pixels and processed with a batch size of 1 (shape (1, 3, 448, 640) refers to batch size, color channels, and image dimensions).
2. Speed: This total of approximately 264ms per image (preprocessing + inference + postprocessing) is typical for a lightweight model like YOLOv8n on a CPU or lower-end GPU.