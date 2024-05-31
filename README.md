# 🎯 MOVE-ICU

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

## Example

### Setup
To run the landmark extraction demo, open a terminal into the loaded container in VSCode using the `dev containers` extension. Then:
`source /opt/ros/humble/setup.bash`
`source /home/ros/venv/bin/activate`

Build the workspace:
`colcon build --symlink-install`

### Run
Once that finishes, source the installation files:
`source install/setup.bash`

Then run the demo (make sure you have a webcam attached to /dev/video0):
`ros2 launch landmark_d webcam_landmark_d.launch.py`

### View
Then view the output using `rqt`; in another terminal:
`source /opt/ros/humble/setup.bash`
`rqt`

Use the plugins file to visualise Images, there should be a `/landmarks_image` topic that can visualise that.