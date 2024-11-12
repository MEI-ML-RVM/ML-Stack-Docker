# MLStack-DOCKER
Docker Stack for the Machine Learning part of the project (YOLOv5 and DeepSort)

## Build Container: 

```shell
docker build -t yolov5-deepsort-api .
docker run --gpus all -p 8765:8765 yolov5-deepsort-api
```
