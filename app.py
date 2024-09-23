# File: yolo_flask_stream.py

import cv2
import torch
import numpy as np
from flask import Flask, Response

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize Flask app
app = Flask(__name__)

# Function to process video frames and perform object detection


def detect_objects(frame):
    # Convert frame to format compatible with the YOLO model
    results = model(frame)
    return results

# Function to draw bounding boxes and labels on the frame


def draw_boxes(frame, results):
    for detection in results.xyxy[0]:  # Loop through detections
        x1, y1, x2, y2, conf, cls = detection[:6]  # Bounding box and label
        label = model.names[int(cls)]  # Get label name
        if conf > 0.5:  # Confidence threshold
            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(
                y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Function to generate frames for the video stream


def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam (or change to video file if needed)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        results = detect_objects(frame)

        # Draw bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame, results)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame_with_boxes = buffer.tobytes()

        # Yield the frame in HTTP response format (MJPEG stream)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_boxes + b'\r\n')

    cap.release()

# Route to access the video stream


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main entry point for the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
