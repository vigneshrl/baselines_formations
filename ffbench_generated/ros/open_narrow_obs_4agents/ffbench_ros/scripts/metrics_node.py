#!/usr/bin/env python3
"""ffbench metrics node: scores a Gazebo/ROS run with the shared FullRunMetrics.

Subscribes to ``/tb_<i>/odom`` for every agent, feeds the same collector every
other backend uses, and writes one CSV row (``zone_ep_<stamp>.csv``) to
``results_dir`` when every agent has cleared the zone, reached the goal, or the
timeout elapses.  Then it shuts ROS down so the driver can move to the next
trial.

    python3 metrics_node.py _config:=<bundle>/config/metrics.yaml _num_agents:=4 \
        _episode_timeout_s:=300 _results_dir:=/tmp/deform_results
"""
import csv
import math
import os
import sys
import time

import numpy as np
import rospy
import yaml
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import FullRunMetrics  # noqa: E402  (copied next to this file)


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MetricsNode:
    def __init__(self):
        rospy.init_node("ffbench_metrics", anonymous=True)
        cfg_path = rospy.get_param("~config")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.n = int(rospy.get_param("~num_agents", 4))
        self.timeout = float(rospy.get_param("~episode_timeout_s", 300.0))
        self.results_dir = rospy.get_param("~results_dir", "/tmp/deform_results")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.end_mode = rospy.get_param("~episode_end_mode", "narrow_clear")
        self.map_name = cfg.get("map", "map")
        cl = np.asarray(cfg["centerline"], dtype=float)
        self.metrics = FullRunMetrics(
            cl[:, 0], cl[:, 1], dt=1.0 / self.rate_hz,
            narrow_center_xy=tuple(cfg["narrow_xy"]), gap_width_m=float(cfg["gap_width_m"]),
            goal_xy=tuple(cfg.get("goal", cl[-1])),
            zone_half_width=int(cfg.get("zone_half_width_wp", 8)),
            collision_thresh=float(cfg.get("collision_thresh_m", 0.5)),
        )
        self.pose = [None] * self.n
        for i in range(self.n):
            rospy.Subscriber(f"/tb_{i}/odom", Odometry, self._odom_cb, callback_args=i, queue_size=1)
        self.t_start = None

    def _odom_cb(self, msg, i):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.pose[i] = (p.x, p.y, _yaw(msg.pose.pose.orientation), math.hypot(v.x, v.y))

    def _obs(self):
        arr = np.asarray(self.pose, dtype=float)
        return {"poses_x": arr[:, 0], "poses_y": arr[:, 1], "poses_theta": arr[:, 2],
                "linear_vels_x": arr[:, 3]}

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        reason = "timeout"
        while not rospy.is_shutdown():
            if any(p is None for p in self.pose):
                rate.sleep()
                continue
            if self.t_start is None:
                self.t_start = time.time()
            self.metrics.step(self._obs())
            if self.metrics.all_zone_cleared and self.end_mode == "narrow_clear":
                reason = "zone_cleared"
                break
            if self.metrics.all_done:
                reason = "goal"
                break
            if time.time() - self.t_start > self.timeout:
                break
            rate.sleep()
        self._flush(reason)
        rospy.signal_shutdown("ffbench episode finished")

    def _flush(self, reason):
        os.makedirs(self.results_dir, exist_ok=True)
        row = {"n_agents": self.n, "map_name": self.map_name, "terminated": reason,
               "sim": "native", **self.metrics.summary()}
        path = os.path.join(self.results_dir, f"zone_ep_{int(time.time())}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        rospy.loginfo("[ffbench_metrics] wrote %s (%s)", path, reason)


if __name__ == "__main__":
    MetricsNode().run()
