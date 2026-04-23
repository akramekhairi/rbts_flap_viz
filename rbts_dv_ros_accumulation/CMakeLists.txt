cmake_minimum_required(VERSION 3.14)
project(rbts_dv_ros_accumulation VERSION 1.3.0)

find_package(
  catkin REQUIRED
  COMPONENTS roscpp
             std_msgs
             sensor_msgs
             geometry_msgs
             image_transport
             cv_bridge
             visualization_msgs
             dv_ros_msgs
             dv_ros_messaging
             dynamic_reconfigure
             message_generation)

generate_dynamic_reconfigure_options(cfg/Accumulator.cfg cfg/EdgeMap.cfg)

add_message_files(
  FILES
  HoleEvent.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

find_package(dv-processing 2.0.3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  visualization_msgs
  dv_ros_msgs
  dv_ros_messaging)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(accumulator src/accumulator.cpp)

add_dependencies(accumulator ${catkin_EXPORTED_TARGETS}
                 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(accumulator ${catkin_LIBRARIES} dv::processing)

add_executable(edge_map src/edge_map.cpp)

add_dependencies(edge_map ${catkin_EXPORTED_TARGETS}
                 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(edge_map ${catkin_LIBRARIES} dv::processing)

add_executable(motion_compensator src/motion_compensator.cpp)

add_dependencies(motion_compensator ${catkin_EXPORTED_TARGETS}
                 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(motion_compensator ${catkin_LIBRARIES} dv::processing)

add_executable(hole_detector src/hole_detector.cpp)

add_dependencies(hole_detector ${catkin_EXPORTED_TARGETS}
                 ${${PROJECT_NAME}_EXPORTED_TARGETS}
                 ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries(hole_detector ${catkin_LIBRARIES} ${OpenCV_LIBS})

catkin_install_python(PROGRAMS
  scripts/hole_detector_gui.py
  scripts/unified_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
