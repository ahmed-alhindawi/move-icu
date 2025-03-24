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
If running the code for the first time on your device, open a command line console and enter the following command to create a configuration file:

`sudo nano /etc/sysctl.d/10-cyclone-max.conf`

Paste the following into the file:

```
# IP fragmentation settings
net.ipv4.ipfrag_time=3  # in seconds, default is 30 s
net.ipv4.ipfrag_high_thresh=134217728  # 128 MiB, default is 256 KiB

# Increase the maximum receive buffer size for network packets
net.core.rmem_max=2147483647  # 2 GiB, default is 208 KiB
```

Save the file and reboot.

### Run
Build the workspace:
`colcon build --symlink-install`

Once that finishes, source the installation files:
`source install/setup.bash`

Then run the demo (make sure you have a webcam attached to /dev/video0):
`ros2 launch landmark_d webcam_landmark_d.launch.py`

(For MediaPipe skeleton tracking run `ros2 launch mpipe mpipe_webcam_launch.py` instead.)

### View
Then view the output using `rqt`; in another terminal:
`source /opt/ros/humble/setup.bash`
`rqt`

Use the plugins file to visualise Images, there should be a `/landmarks_image` topic that can visualise that.

## Troubleshooting
### `/usr/lib64/libEGL_nvidia.so.***.**.**: No such file or directory`

If the container fails to build and the log file containes the above error, this is due to a recent update on the host. The podman container searches for the GPU runtime libraries through a predefined configuration file (nvidia.yaml) that locates the libraries on the host. When the drivers are updated, those libraries don't exist anymore and therefore the container can't find them to load the cuda runtime. 

To fix, run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` and restart VSCode.

