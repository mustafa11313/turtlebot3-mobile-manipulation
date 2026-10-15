# TurtleBot3 Fetch & Deliver - ROS 2 Humble + Gazebo Classic development image.
#
#   docker compose -f docker/docker-compose.yml build
#   docker compose -f docker/docker-compose.yml run --rm sim
FROM osrf/ros:humble-desktop-full

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions python3-rosdep python3-pip python3-matplotlib \
      ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
      ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gripper-controllers \
      ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
      ros-humble-moveit ros-humble-py-trees ros-humble-cv-bridge ros-humble-xacro \
      ros-humble-turtlebot3-msgs \
    && rm -rf /var/lib/apt/lists/*

ENV WS=/ws
WORKDIR ${WS}
COPY . ${WS}/src/turtlebot3-fetch-deliver

RUN apt-get update \
    && (rosdep init 2>/dev/null || true) && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/humble/setup.bash && colcon build --symlink-install

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo 'source /usr/share/gazebo/setup.sh' >> /root/.bashrc \
    && echo "source ${WS}/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
