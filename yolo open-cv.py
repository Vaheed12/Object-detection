import random
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("weights/yolov8n.pt")

# Load class names from coco.txt file
with open(r"C:\Users\Vaheed\AVS Code\8. YOLO\utils\coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Generate random colors for class list
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in class_list
]

# Set up video input
video_path = r"C:\Users\Vaheed\AVS Code\8. YOLO\video_sample1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Retrieve video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video output
output_file = "annotated_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream. Exiting gracefully.")
        break

    # YOLO model prediction
    results = model.predict(source=[frame], conf=0.45)

    # Annotate detected objects
    for box in results[0].boxes:
        bb = box.xyxy.numpy()[0]  # Bounding box [x_min, y_min, x_max, y_max]
        cls_id = int(box.cls.numpy()[0])  # Class ID
        conf = round(box.conf.numpy()[0] * 100, 1)  # Confidence score

        # Draw bounding box
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),  # Top-left corner
            (int(bb[2]), int(bb[3])),  # Bottom-right corner
            detection_colors[cls_id], 3,
        )

        # Add label and confidence score
        label = f"{class_list[cls_id]} {conf}%"
        cv2.putText(
            frame,
            label,
            (int(bb[0]), int(bb[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Write annotated frame to the output video
    out.write(frame)

    # Display the annotated frame (optional)
    cv2.imshow("Object Detection", frame)

    # Exit when 'Q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as {output_file}")