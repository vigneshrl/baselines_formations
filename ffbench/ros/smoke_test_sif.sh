#!/usr/bin/env bash
# Smoke-test a built DEFORM container: ROS workspace, planner package, bridge
# package, CasADi, f1tenth_gym import, TurtleBot3 sim packages (native mode).
#     bash ffbench/ros/smoke_test_sif.sh [ffbench_generated/deform_ros.sif]
SIF=${1:-ffbench_generated/deform_ros.sif}
apptainer exec --cleanenv --no-home --writable-tmpfs --env HOME=/tmp/ros_home --env ROS_HOME=/tmp/ros_home "$SIF" bash -c '
mkdir -p /tmp/ros_home
source /opt/ros/noetic/setup.bash && source /root/DEFORM/devel/setup.bash
echo "plan_manager:   $(rospack find plan_manager 2>&1)"
echo "bridge pkg:     $(rospack find f1tenth_deform_bridge 2>&1)"
echo "bridge script:  $(ls /root/DEFORM/devel/lib/f1tenth_deform_bridge/ 2>&1 | tr "\n" " ")"
echo "one_run.xml:    $(ls $(rospack find plan_manager)/launch/one_run.xml 2>&1)"
echo "traj_opti caps: $(grep -h "MAX_LINEAR_VELOCITY\|MAX_ANGULAR_VELOCITY" /root/DEFORM/src/control/traj_opti/include/traj_opti/traj_opti.h | tr -s " " | tr "\n" ";")"
echo "turtlebot3:     $(rospack find turtlebot3_description 2>&1) | $(rospack find turtlebot3_gazebo 2>&1)"
echo "map_server:     $(rospack find map_server 2>&1)"
python3 -c "import casadi, ot, numpy, gymnasium; print(\"python: casadi\", casadi.__version__, \"pot ok numpy\", numpy.__version__, \"gymnasium\", gymnasium.__version__)"
PYTHONPATH=/opt/f1tenth_gym python3 -c "import f1tenth_gym, pathlib; print(\"f1tenth_gym:\", pathlib.Path(f1tenth_gym.__file__).parent, \"maps:\", len(list((pathlib.Path(f1tenth_gym.__file__).parent.parent/\"maps\").iterdir())))"
python3 -c "import rospy; print(\"rospy ok\")"
'
