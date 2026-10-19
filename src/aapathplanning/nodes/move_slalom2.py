#!/usr/bin/env python3
import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion


def main():
    rospy.init_node("move_slalom2", anonymous=False)

    # ---- params (set from launch) ----
    model_name = rospy.get_param("~model_name", "Slalom_2")
    x_min = float(rospy.get_param("~x_min", -1.5))
    x_max = float(rospy.get_param("~x_max", -0.5))
    A = float(rospy.get_param("~A", 0.5))
    y0 = float(rospy.get_param("~y0", 0.2))
    period = float(rospy.get_param("~period", 12.0))   # seconds, one full back-and-forth
    rate_hz = float(rospy.get_param("~rate", 30.0))
    z = float(rospy.get_param("~z", 0.25))

    # ---- validate ----
    if period <= 0:
        rospy.logerr("period must be > 0 (got %.3f)", period)
        return
    if rate_hz <= 0:
        rospy.logerr("rate must be > 0 (got %.3f)", rate_hz)
        return

    # keep x_min < x_max
    if x_max < x_min:
        x_min, x_max = x_max, x_min
        rospy.logwarn("Swapped x_min/x_max -> x_min=%.3f x_max=%.3f", x_min, x_max)

    x_c = 0.5 * (x_min + x_max)
    half = 0.5 * (x_max - x_min)
    if abs(half) < 1e-6:
        rospy.logerr("x_min and x_max too close (x_min=%.3f x_max=%.3f). "
                     "Use a non-zero range, e.g. -1.5 to -0.5", x_min, x_max)
        return

    rospy.loginfo(
        "move_slalom2 params: model=%s x=[%.2f, %.2f] center=(%.2f, %.2f) A=%.2f period=%.2f rate=%.1f z=%.2f",
        model_name, x_min, x_max, x_c, y0, A, period, rate_hz, z
    )

    # wait sim time start (use_sim_time=true)
    while not rospy.is_shutdown() and rospy.get_time() == 0.0:
        rospy.sleep(0.05)

    # yaw = 0
    quat = Quaternion(0.0, 0.0, 0.0, 1.0)

    # connect service
    srv_name = "/gazebo/set_model_state"
    rospy.loginfo("Waiting for %s ...", srv_name)
    rospy.wait_for_service(srv_name)
    set_state = rospy.ServiceProxy(srv_name, SetModelState, persistent=True)
    rospy.loginfo("Connected to %s. Start moving.", srv_name)

    omega = 2.0 * math.pi / period
    phase = 0.0

    last_t = rospy.get_time()
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        now = rospy.get_time()
        dt = now - last_t
        last_t = now

        # sim time jumped (reset/pause/unpause) -> don't integrate
        if dt < 0 or dt > 0.5:
            dt = 0.0

        # integrate phase (no modulo; continuous back-and-forth)
        phase += omega * dt

        # smooth back-and-forth in x (endpoint velocity = 0)
        x = x_c + half * math.sin(phase)

        # "sine-shaped" curve in x-y plane
        # normalize x to u in [-1,1], then map to y
        u = (x - x_c) / half
        y = y0 + A * math.sin((math.pi / 2.0) * u)

        state = ModelState()
        state.model_name = model_name
        state.reference_frame = "world"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation = quat
        state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
        state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0

        try:
            resp = set_state(state)
            if not resp.success:
                rospy.logwarn_throttle(2.0, "set_model_state failed: %s", resp.status_message)
        except Exception as e:
            rospy.logerr_throttle(2.0, "set_model_state exception: %s", str(e))

        # IMPORTANT: handle ROS time moving backwards (Gazebo reset)
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            # resync; keep node alive
            last_t = rospy.get_time()
            phase = 0.0  # reset motion phase (prevents weird jumps after reset)
            rospy.logwarn("ROS time moved backwards. Resyncing timers and resetting phase.")
            continue


if __name__ == "__main__":
    main()