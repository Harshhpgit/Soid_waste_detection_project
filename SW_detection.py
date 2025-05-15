from ultralytics import YOLO

# Load a base YOLOv8 model (small, medium, large – choose based on your resources)
model = YOLO('yolov8m.pt')  # 's' for small; you can use 'm' or 'n' too

# Train the model
model.train(
    data="E:\project XYZ\DATASET 2\data.yaml",  # <-- Change this to the full path of your file
    epochs=50,
    imgsz=640,
    batch=16,
    project='waste_detection',
    name='yolov8_roboflow_custom',
    val=True
)
