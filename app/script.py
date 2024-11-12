import argparse
import pickle
import socket
import struct
import cv2
import numpy as np
import torch
import time
import threading
import warnings
warnings.filterwarnings("ignore")

from app import YoloV5Detector
from flask import Flask, request, jsonify
from deep_sort_realtime.deepsort_tracker import DeepSort


# Flask app to receive configuration updates
app = Flask(__name__)
detector = None
object_tracker = None
client_socket = None
polygon_A = None
polygon_B = None
person_count = 0
tracked_positions = {}
ip = "0.0.0.0"
port = 9999

# API to update configurations
@app.route('/update_config', methods=['POST'])
def update_config():
    global ip, port, polygon_A, polygon_B, client_socket
    data = request.json

    # Update parameters from request
    ip = data.get('ip', ip)
    port = int(data.get('port', port))
    polygon_A = np.array(eval(data['polygonA']))
    polygon_B = np.array(eval(data['polygonB']))

    # Reconnect to the updated socket server
    if client_socket:
        client_socket.close()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    return jsonify({"status": "Configuration updated successfully"}), 200
    
####################################################################################################################

def is_inside_polygon(bbox_center, polygon):
    return cv2.pointPolygonTest(polygon, bbox_center, False) >= 0

####################################################################################################################
  
# Main YOLOv5 Detection Loop
def run_detection():
    global client_socket, polygon_A, polygon_B, person_count, tracked_positions
    data = b""
    payload_size = struct.calcsize("L")
    display_width = 1920
    display_height = 1080

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        frame = cv2.resize(frame, (display_width, display_height))
        
        results = detector.rescale_frame(frame)
        labels, cord, frame, height, width, downscale_factor = results
        frame, _, detections = detector.process_detections((labels, cord), frame, height, width)
        tracks = object_tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox_center = ((ltrb[0] + ltrb[2]) // 2, (ltrb[1] + ltrb[3]) // 2)

            if track_id in tracked_positions:
                prev_position = tracked_positions[track_id]
                if is_inside_polygon(prev_position, polygon_A) and is_inside_polygon(bbox_center, polygon_B):
                    person_count += 1
                elif is_inside_polygon(prev_position, polygon_B) and is_inside_polygon(bbox_center, polygon_A):
                    if person_count > 0:
                        person_count -= 1

            tracked_positions[track_id] = bbox_center

        cv2.polylines(frame, [polygon_A], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [polygon_B], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, f"People Inside: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv5 + DeepSORT", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    client_socket.close()
    cv2.destroyAllWindows()

####################################################################################################################

# Initialize Detector and Tracker
def initialize_detection(model_size="x"):
    global detector, object_tracker
    detector = YoloV5Detector(model_size)
    object_tracker = DeepSort()

####################################################################################################################

# Start Flask and YOLOv5 + DeepSORT
if __name__ == '__main__':
    initialize_detection()
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.start()
    app.run(host='0.0.0.0', port=8765)