from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLOv8 detection model
detector = YOLO(r"E:\project XYZ\best.pt")  # Trained YOLOv8 model for detecting "waste" objects

# Load classifier model (from your existing ResNet code)
from tensorflow.keras.models import load_model
classifier = load_model("E:\project XYZ\my_model.keras")  # Save your trained model from earlier

# Class labels (match your classifier)
class_labels = ["organic", "non-organic"]  # Adjust based on your dataset

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 object detection
    results = detector(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Crop the detected object
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Resize and preprocess for classifier
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict class
            pred = classifier.predict(img_array)[0][0]
            label = class_labels[0] if pred < 0.5 else class_labels[1]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Waste Detection & Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
