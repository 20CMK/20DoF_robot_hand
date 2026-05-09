#!/usr/bin/env python3
import sys
import select
import tty
import termios

import rospy
from std_msgs.msg import Bool, Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def get_key(timeout=0.1):
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return None

def make_zero_traj(joint_names, duration_sec=5.0):
    traj = JointTrajectory()
    traj.joint_names = joint_names
    pt = JointTrajectoryPoint()
    pt.positions = [0.0] * len(joint_names)
    pt.time_from_start = rospy.Duration.from_sec(duration_sec)
    traj.points.append(pt)
    return traj

def main():
    rospy.init_node("switch_node", anonymous=True)

    switch_topic = rospy.get_param("~switch_topic", "/switch")
    imu_mode_topic = rospy.get_param("~imu_mode_topic", "/imu_reset_mode")
    traj_topic   = rospy.get_param("~traj_topic", "/arm_controller/command")
    joint_names  = rospy.get_param("~joint_names",
                                   ["joint1","joint2","joint3","joint4","joint5","joint6"])

    rate_hz = rospy.get_param("~rate", 20)

    switch_pub = rospy.Publisher(switch_topic, Bool, queue_size=10)
    traj_pub   = rospy.Publisher(traj_topic, JointTrajectory, queue_size=10)
    imu_mode_pub = rospy.Publisher(imu_mode_topic, Int32, queue_size=10)

    rate = rospy.Rate(rate_hz)

    # 터미널 설정
    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while not rospy.is_shutdown():
            key = get_key(timeout=1.0/rate_hz)
            if not key:
                rate.sleep()
                continue

            key = key.lower()

            if key == 's':
                rospy.loginfo("Stop ik move")
                switch_pub.publish(Bool(data=False))

            elif key == 'g':
                rospy.loginfo("Go ik move")
                switch_pub.publish(Bool(data=True))

            elif key == 'z':
                rospy.loginfo("Zero pose — publishing 0 0 0 0 0 0 (7s)")
                rospy.loginfo("Stop ik move")
                switch_pub.publish(Bool(data=False))
                traj = make_zero_traj(joint_names, duration_sec=7.0)
                traj_pub.publish(traj)
                
            elif key == 'r':                                                        # ← 새로운 키 입력 처리
                rospy.loginfo("IMU reset request")
                imu_mode_pub.publish(Int32(data=2))
                
            elif key == 'o':                                                        # ← 새로운 키 입력 처리
                rospy.loginfo("IMU offset request")
                imu_mode_pub.publish(Int32(data=1))
                
            # elif key == 'q':
            #     rospy.loginfo("quit requested.")
            #     break

            rate.sleep()

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
