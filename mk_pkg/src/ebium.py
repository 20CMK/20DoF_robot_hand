#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Quaternion, TransformStamped
import serial
import tf2_ros

comport_num = "/dev/ebium"
comport_baudrate = 115200

ser = serial.Serial(port=comport_num, baudrate=comport_baudrate, timeout=0.1)


def publish_data():
    pub = rospy.Publisher("wrist_imu_data", Quaternion, queue_size=10)
    rospy.init_node("wrist_imu_publisher", anonymous=True)
    rate = rospy.Rate(500)
    
    # br = tf2_ros.TransformBroadcaster()
    # parent = rospy.get_param("~parent_frame", "world")
    # child  = rospy.get_param("~child_frame",  "wrist_imu_link")


    while not rospy.is_shutdown():
        try:
            line = ser.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="ignore").strip()
            if text.startswith("*"):
                text = text[1:]
            parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 4:
                qz, qy, qx, qw = [float(parts[i]) for i in range(4)]
                # msg = Quaternion(w=qw, x=qz, y=-qy, z=qx)
                msg = Quaternion(w=-qw, x=-qx, y=qy, z=qz)
                pub.publish(msg)
                
            # t = TransformStamped()
            # t.header.stamp = rospy.Time.now()
            # t.header.frame_id = parent
            # t.child_frame_id  = child
            # t.transform.translation.x = 0.0
            # t.transform.translation.y = 0.0
            # t.transform.translation.z = 0.0
            # t.transform.rotation = msg
            # br.sendTransform(t)
                
        except Exception as e:
            rospy.logwarn(f"read/publish error: {e}")
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_data()
    except rospy.ROSInterruptException:
        pass
