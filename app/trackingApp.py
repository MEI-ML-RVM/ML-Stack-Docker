import asyncio
import websockets
import requests
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from YoloV5Detector import YoloV5Detector
from flask import Flask, request, jsonify
from deep_sort_realtime.deepsort_tracker import DeepSort


class TrackerApp:
    def __init__(self):
        self.detector = YoloV5Detector("x")
        self.object_tracker = DeepSort()
        self.flask_url = None
        self.polygon_A = None
        self.polygon_B = None
        self.ws_ip = None      # WebSocket server IP
        self.ws_port = None    # WebSocket server port
        self.person_count = 0
        self.tracked_positions = {}

    def update_config(self, config):
        """
        Update configuration dynamically from Flask app.
        """
        self.flask_url = config['flask_url']
        self.polygon_A = np.array(config['entry_polygon'])
        self.polygon_B = np.array(config['exit_polygon'])
        self.ws_ip = config['ws_ip']
        self.ws_port = config['ws_port']

    async def process_stream(self):
        """
        Connect to WebSocket server and process incoming frames.
        """
        try:
            async with websockets.connect(f"ws://{self.ws_ip}:{self.ws_port}") as websocket:
                print(f"Connected to WebSocket at ws://{self.ws_ip}:{self.ws_port}")
                while True:
                    frame_data = await websocket.recv()
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.process_frame(frame)
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")

    def process_frame(self, frame):
        """
        Process a single frame: detect, track, and send results.
        """
        results = self.detector.rescale_frame(frame)
        labels, cord, frame, height, width, downscale_factor = results
        frame, confidence, detections = self.detector.process_detections((labels, cord), frame, height, width)
        tracks = self.object_tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox_center = ((ltrb[0] + ltrb[2]) // 2, (ltrb[1] + ltrb[3]) // 2)

            if track_id in self.tracked_positions:
                prev_position = self.tracked_positions[track_id]

                if self.is_inside_polygon(prev_position, self.polygon_A) and self.is_inside_polygon(bbox_center, self.polygon_B):
                    self.person_count += 1
                    self.send_event(track_id, 1)

                elif self.is_inside_polygon(prev_position, self.polygon_B) and self.is_inside_polygon(bbox_center, self.polygon_A):
                    if self.person_count > 0:
                        self.person_count -= 1
                        self.send_event(track_id, -1)

            self.tracked_positions[track_id] = bbox_center

        self.send_frame(frame)

    def is_inside_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon.
        """
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def send_event(self, track_id, event):
        """
        Send entry/exit events to the Flask app.
        """
        try:
            data = {'track_id': track_id, 'event': event, 'person_count': self.person_count}
            requests.post(f"{self.flask_url}/results", json=data)
        except Exception as e:
            print(f"Error sending event to Flask: {e}")

    def send_frame(self, frame):
        """
        Send the processed frame to the Flask app.
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            requests.post(f"{self.flask_url}/frame", files={'frame': buffer.tobytes()})
        except Exception as e:
            print(f"Error sending frame to Flask: {e}")


if __name__ == "__main__":
    # Fetch configuration from Flask app
    try:
        flask_url = "http://<flask_server_ip>:5000/configure"  #TODO: HARDCODE THIS THING 
        config = requests.get(flask_url).json()
    except Exception as e:
        print(f"Error fetching configuration: {e}")
        exit(1)

    # Initialize TrackerApp and update configuration
    app = TrackerApp()  # Removed static flask_url from constructor
    app.update_config(config)

    # Start processing the video stream
    asyncio.run(app.process_stream())