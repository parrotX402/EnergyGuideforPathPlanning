#!/usr/bin/env python3
import os, time
import numpy as np
import rospy
from nav_msgs.msg import Odometry

class TrajRecorder:
    def __init__(self, out_path):
        self.out_path = out_path
        self.xy = []
        rospy.Subscriber("/odom", Odometry, self.cb, queue_size=50)

    def cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.xy.append([p.x, p.y])

    def save(self):
        arr = np.array(self.xy, dtype=np.float32)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        np.save(self.out_path, arr)
        rospy.loginfo(f"[REC] saved traj: {self.out_path}, N={len(arr)}")

if __name__ == "__main__":
    rospy.init_node("traj_recorder")
    out = rospy.get_param("~out", "/tmp/traj.npy")
    rec = TrajRecorder(out)
    rospy.loginfo("[REC] recording /odom ... Ctrl+C to stop and save")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rec.save()