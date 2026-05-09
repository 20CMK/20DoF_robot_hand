#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool


class RedObjectDetector:
    def __init__(self):
        rospy.init_node('wrist_detector', anonymous=True)

        # === 파라미터 ===
        self.frame_id = rospy.get_param("~frame_id", "camera_color_optical_frame")
        self.cam_info_topic = rospy.get_param("~cam_info", "/camera/color/camera_info")
        self.color_topic = rospy.get_param("~color", "/camera/color/image_raw/compressed")
        self.depth_topic = rospy.get_param("~depth", "/camera/aligned_depth_to_color/image_raw")

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        # 퍼블리셔
        self.point_pub = rospy.Publisher('/wrist_point', PointStamped, queue_size=10)
        self.debug_img_pub = rospy.Publisher('/wrist_detector/debug_image', Image, queue_size=1)
        self.debug_img_comp_pub = rospy.Publisher('/wrist_detector/debug_image/compressed', CompressedImage, queue_size=1)

        # 구독자
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber(self.color_topic, CompressedImage, self.color_callback, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)

        # 내부 상태
        self.depth_image = None
        self.color_image = None

        rospy.loginfo("wrist_detector started.")
        rospy.spin()

    # --------- Callbacks ----------
    def camera_info_callback(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]

    def color_callback(self, msg: CompressedImage):
        # 압축 컬러 이미지 -> BGR
        try:
            self.color_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"Failed to decode color image: {e}")
            return
        # 컬러 수신 시 처리 시도(깊이/내부파라미터 준비되었는지 가드)
        self.process_images()

    def depth_callback(self, msg: Image):
        # Depth 이미지 -> numpy 배열
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logwarn(f"Failed to decode depth image: {e}")
            return

        # float32 또는 uint16 모두 수용
        self.depth_image = np.array(depth_raw)

    # --------- Processing ----------
    def process_images(self):
        if self.color_image is None or self.depth_image is None:
            return
        if self.fx is None or self.fy is None:
            return

        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

        # 빨간색 2구간 범위
        lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 120, 70], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 가장 큰 컨투어 선택
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # 인덱스 범위 가드
        if not (0 <= v < self.depth_image.shape[0] and 0 <= u < self.depth_image.shape[1]):
            return

        z_raw = float(self.depth_image[v, u])

        # 잘못된 깊이 필터
        if np.isnan(z_raw) or z_raw <= 0:
            rospy.logwarn_throttle(2.0, "Invalid depth value.")
            return

        # 깊이 단위 정규화:
        #  - 보통 RealSense 16UC1은 mm, 32FC1은 m
        #  - 휴리스틱: 값이 10보다 크면 mm로 간주
        if z_raw > 10.0:
            Z_mm = z_raw
        else:
            Z_mm = z_raw * 1000.0

        # 핀홀 역투영 (mm 단위로 계산)
        X_mm = (u - self.cx) * Z_mm / self.fx
        Y_mm = (v - self.cy) * Z_mm / self.fy

        # 디버그 렌더링
        result_img = self.color_image.copy()

        # ---- 사용자 정의 실제 좌표 매핑(요청 코드 유지) ----
        # real_x = ((-Y_mm / 1000.0) * 0.8 + 0.15) + 0.15
        # if real_x < 0.2:
        #     real_x = 0.2
        # if real_x > 1.0:
        #     real_x = 1.0
        
        real_x = ((-Y_mm / 1000.0) * 1.2 + 0.15) + 0.4
        if real_x < 0.2:
            real_x = 0.2
        if real_x > 0.85:
            real_x = 0.85

        real_y = (X_mm / 1000.0) * 2.0 + 0.05

        real_z = 1.2 - ((Z_mm / 1000.0) + 0.2)
        if real_z < 0.2:
            real_z = 0.2
        if real_z > 0.7:
            real_z = 0.7

        # real_x = round(real_x, 4)
        # real_y = round(-real_y, 4)
        # real_z = round(real_z, 4)
        real_x = round(real_x, 3)
        real_y = round(-real_y, 3)
        real_z = round(real_z, 3)


        # 텍스트/마커
        text = f"X={real_x:.3f}m Y={real_y:.3f}m Z={real_z:.3f}m"
        cv2.putText(result_img, text, (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(result_img, (u, v), 5, (0, 255, 0), -1)
        cv2.drawContours(result_img, [c], -1, (0, 255, 255), 2)

        # 포인트 발행
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = self.frame_id
        point_msg.point.x = real_x
        point_msg.point.y = real_y
        point_msg.point.z = real_z
        self.point_pub.publish(point_msg)

        # rqt_image_view 용 이미지 발행 (비압축)
        img_msg = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
        img_msg.header.stamp = point_msg.header.stamp
        img_msg.header.frame_id = self.frame_id
        self.debug_img_pub.publish(img_msg)

        # rqt_image_view 용 이미지 발행 (압축)
        ok, buf = cv2.imencode('.jpg', result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            comp_msg = CompressedImage()
            comp_msg.header = img_msg.header
            comp_msg.format = "jpeg"
            comp_msg.data = np.asarray(buf).tobytes()
            self.debug_img_comp_pub.publish(comp_msg)

        rospy.loginfo_throttle(2.0, "Published /wrist_point and /wrist_detector/debug_image")


if __name__ == '__main__':
    try:
        RedObjectDetector()
    except rospy.ROSInterruptException:
        pass
