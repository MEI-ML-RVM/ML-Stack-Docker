import cv2
import torch
import warnings
warnings.filterwarnings("ignore")

class YoloV5Detector():
    """
    YoloV5Detector class for detecting objects using YOLOv5 model.

    Parameters:
    - fileSize (str): The size specification for the YOLOv5 model file. Default is "n".

    Attributes:
    - model: YOLOv5 model loaded using torch.hub.load.
    - device (str): Device used for inference, either 'cuda' or 'cpu'.
    
    Methods:
    - __init__(self, fileSize: str = "n"): Initializes the YoloV5Detector instance, loading the YOLOv5 model.
    - rescale_frame(self, frame): Rescales the input frame, runs YOLOv5 inference, and returns detected labels and coordinates.
    - plot_boxes(self, results, frame, height, width, confidence=0.3): Plots bounding boxes on the input frame based on detection results.

    """

    def __init__(self, fileSize: str = "n"):
        """
        Initializes the YoloV5Detector instance.

        Parameters:
        - fileSize (str): The size specification for the YOLOv5 model file. Default is "n".
        """

        ptFile = "yolov5" + fileSize

        # Load YoloV5 Model
        self.model = torch.hub.load(
            "ultralytics/yolov5:master",    # GitHub repository and branch specification
            ptFile                         # Name of the File to load
        )
        
        self.model.conf = 0.50   # filter all results under 0.50 confidence
        self.model.classes = [0] # filter all results that are not form class 0 = "person"

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
    
    def rescale_frame(self, frame):
        """
        Rescales the input frame, runs YOLOv5 inference, and returns detected labels and coordinates.

        Parameters:
        - frame: Input frame to be processed.

        Returns:
        Tuple containing labels and coordinates of detected objects.
        """

        self.model.to(self.device)
        downscale_factor = 2

        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord, frame, height, width, downscale_factor

    
    def process_detections(self, results, frame, height, width):
        """
        Processes detection results, returning the frame with bounding boxes
        and a list of detections along with their confidence scores.

        Parameters:
        - results: Tuple containing labels and coordinates of detected objects.
        - frame: Input frame to be annotated.
        - height: Height of the original frame.
        - width: Width of the original frame.

        Returns:
        Tuple containing annotated frame, the maximum confidence score, and a list of detections.
        """

        labels, cord = results
        detections = []
        confidence = 0.0

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

            confidence = float(row[4].item())

            class_num = int(labels[i])
            class_name = self.model.names[class_num]
            detections.append(([x1, y1, int(x2-x1), int(y2-y1)], confidence, class_name))
        
        return frame, confidence, detections