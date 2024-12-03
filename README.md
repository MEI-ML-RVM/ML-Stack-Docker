# MLStack-DOCKER
Docker Stack for the Machine Learning part of the project (YOLOv5 and DeepSort)

## Build Container: 

1. Download yolov5x.pt weight file

2. Run the following commands:
```shell
docker build -t ml_stack .
docker run --gpus all -p 8765:8765 ml_stack
```
