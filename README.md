# 🎯 CARTESIAN

This is a respository that facilitates the deployment of non-invasive (i.e., camera based) physiological signals.

The _vision_ of this project is to run the models on an low-powered device with sensors attached at a clinically safe distance 
providing accurate and precise data.

## Stack
We provide an opinionated stack to run. Namely:

- Dockerised linux, namely Ubuntu 22.04 supplied with NVIDIA CUDA libraries
- ROS2 for containerisation of nodes
  - Customised Messages that are semantically meaningful
- OpenCV with `cv_bridge` for ROS-OpenCV integration
- PyTorch (>2.1) for inference
- VSCode
  - With the `dev containers` extension to facilitate development

## Models 
The currently working models are:

1. Face detection
2. Landmark extraction

Working on:
1. Eye-gaze regression
2. Blink detection
3. Pose estimation

## Future work
We aim to incoorporate multi-modal sensing using depth, infra-red, and motion tracking