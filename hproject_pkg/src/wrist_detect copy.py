#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
import time



point_msg = PointStamped()


class RedObjectDetector:
    def __init__(self):
        rospy.init_node('wrist_detector', anonymous=True)

        self.bridge = CvBridge()
        self.intrinsics = None
        
        # Pose publisher
        self.point_pub = rospy.Publisher('/wrist_point', PointStamped, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.color_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
        self.depth_image = None
        self.color_image = None

        rospy.spin()

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            # rospy.loginfo("Camera intrinsics received.")

    def color_callback(self, msg):
        # 압축 이미지 -> OpenCV 이미지
        self.color_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_images()



    def depth_callback(self, msg):
        # Depth 이미지 -> numpy 배열
        depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_image = np.array(depth_raw, dtype=np.float32)

    def process_images(self):
        # if self.color_image is None or self.depth_image is None or self.intrinsics is None:
        #     return

        # HSV 변환
        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

        # # 빨간색 범위
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        # ── 주황색 범위(권장 기본값) ─────────────────
        # lower_orange = np.array([8, 120, 80], dtype=np.uint8)   # H≈8°
        # upper_orange = np.array([24, 255, 255], dtype=np.uint8) # H≈24°
        
        # lower_orange = np.array([35, 80, 80], dtype=np.uint8)
        # upper_orange = np.array([55, 255, 255], dtype=np.uint8)
        # mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # (선택) 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 가장 큰 윤곽선
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # rospy.loginfo("빨간색 물체 없음.")
            return

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            # rospy.logwarn("영역이 0입니다.")
            return

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # Depth 값 (meter 단위)
        z_raw = self.depth_image[v, u]
        if np.isnan(z_raw) or z_raw <= 0:
            rospy.logwarn("잘못된 Depth 값.")
            return

        # 단위 변환
        Z = z_raw  # meter
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        

        # rospy.loginfo("빨간색 물체 중심 좌표 (cm): X=%.1f, Y=%.1f, Z=%.1f", X, Y, Z)

        # 시각화
        result_img = self.color_image.copy()
        
        
        
        real_x = ((-Y/1000) * 0.8 + 0.15) + 0.15 
        if real_x < 0.2 :
            real_x = 0.2
        if real_x > 0.8 :
            real_x = 0.8
            
        real_y = (X/1000)* 2 + 0.05
        
        
        real_z = 1.2 - ((Z/1000) + 0.2)
        if real_z < 0.2 :
            real_z = 0.2
        if real_z > 0.7 :
            real_z = 0.7
            
        real_x = round(real_x, 4)
        real_y = round(real_y, 4)
        real_z = round(real_z, 4)
        
        
        
        text = "X={:.3f}m Y={:.3f}m Z={:.3f}m".format(real_x, real_y, real_z)
        cv2.putText(
            result_img,
            text,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        
        # PoseStamped 메시지 생성
        point_msg.point.x = real_x
        point_msg.point.y = real_y
        point_msg.point.z = real_z

        
        
        self.point_pub.publish(point_msg)
        rospy.loginfo("target_pose 발행")

            
        
        cv2.circle(result_img, (u, v), 5, (0, 255, 0), -1)
        cv2.drawContours(result_img, [c], -1, (0, 255, 255), 2)
        # cv2.imshow("yellow Object Detection", result_img)
        # cv2.waitKey(1)

if __name__ == '__main__':
    try:
        RedObjectDetector()
    except rospy.ROSInterruptException:
        pass

