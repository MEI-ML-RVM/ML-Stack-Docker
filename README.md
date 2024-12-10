# MLStack-DOCKER
Docker Stack for the Machine Learning part of the project (YOLOv5 and DeepSort)

## Build Container: 

1. Download yolov5x.pt weight file

2. Run the following commands:
If dev environment is WSL:
```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

```shell
docker build -t ml_stack .
docker run --gpus all -p 8765:8765 ml_stack
```
