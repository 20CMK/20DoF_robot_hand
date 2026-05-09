#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('usb_camera_publisher')

    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
    bridge = CvBridge()

    cap = cv2.VideoCapture(0)  # /dev/video0

    if not cap.isOpened():
        rospy.logerr("❌ Failed to open camera.")
        return

    rate = rospy.Rate(30)  # 30 FPS

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("⚠️ Failed to grab frame.")
            continue

        img_msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        pub.publish(img_msg)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
