cmake_minimum_required(VERSION 3.0.2)
project(mk_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  python_qt_binding
  rviz
)


catkin_package()

install(PROGRAMS
  scripts/pyqt_rviz_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)