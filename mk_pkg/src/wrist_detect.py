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

        # 마커 실제 크기 (m)
        self.marker_length = rospy.get_param("~marker_length", 0.1)  # 10cm

        self.bridge = CvBridge()

        # 카메라 내부 파라미터
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco 관련
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.R_inv = None  # 카메라 → 마커 좌표 변환용
        self.T_inv = None
        self.rvec_fixed = None
        self.tvec_fixed = None

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
        """카메라 내부 파라미터 및 왜곡계수 셋업"""
        if self.fx is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]

            self.camera_matrix = np.array([
                [self.fx, 0.0,      self.cx],
                [0.0,     self.fy,  self.cy],
                [0.0,     0.0,      1.0]
            ], dtype=np.float32)

            # 왜곡계수 D는 길이가 5 이상일 수 있음 → 앞 5개만 사용
            if len(msg.D) >= 5:
                self.dist_coeffs = np.array(msg.D[:5], dtype=np.float32)
            else:
                # 없으면 0으로
                self.dist_coeffs = np.zeros((5,), dtype=np.float32)

            rospy.loginfo("Camera intrinsics set.")

    def color_callback(self, msg: CompressedImage):
        # 압축 컬러 이미지 -> BGR
        try:
            self.color_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"Failed to decode color image: {e}")
            return

        # 먼저 ArUco 마커 업데이트 (좌표계 고정)
        self.update_aruco()

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

    # --------- ArUco 처리 ----------
    def update_aruco(self):
        """
        color_image에서 ArUco 마커(id=1)를 탐지하고,
        카메라→마커 좌표 변환(R_inv, T_inv)을 한 번 고정.
        """
        if self.color_image is None:
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None or 1 not in ids:
            # 마커 안 보이면 스킵
            return

        # id == 1 인 마커만 사용
        idx = np.where(ids == 1)[0][0]
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], self.marker_length,
            self.camera_matrix, self.dist_coeffs
        )

        self.rvec_fixed = rvec[0]
        self.tvec_fixed = tvec[0]

        # 디버그용: 마커 좌표축 그리기
        cv2.drawFrameAxes(
            self.color_image,
            self.camera_matrix,
            self.dist_coeffs,
            self.rvec_fixed,
            self.tvec_fixed,
            0.05  # 5cm 축 길이
        )

        # R_inv, T_inv 고정 (카메라 → 마커 좌표)
        if self.R_inv is None or self.T_inv is None:
            R, _ = cv2.Rodrigues(self.rvec_fixed)
            T = self.tvec_fixed.reshape((3, 1))  # 마커 원점(월드)에서 본 카메라 위치
            self.R_inv = R.T
            self.T_inv = -self.R_inv @ T
            rospy.loginfo("✅ ArUco marker (id=1) coordinate frame fixed.")

    # --------- Processing ----------
    def process_images(self):
        if self.color_image is None or self.depth_image is None:
            return
        if self.fx is None or self.fy is None:
            return
        if self.R_inv is None or self.T_inv is None:
            # 마커 좌표계 아직 못 잡았으면 좌표 계산 안 함
            rospy.logwarn_throttle(2.0, "Aruco marker not detected yet. Skipping.")
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

        # 깊이 단위 정규화 (m 단위로 변환)
        if z_raw > 10.0:
            # mm 로 간주
            Z_m = z_raw / 1000.0
        else:
            # 이미 m 라고 가정
            Z_m = z_raw

        # 핀홀 역투영 (카메라 좌표계, m 단위)
        X_m = (u - self.cx) * Z_m / self.fx
        Y_m = (v - self.cy) * Z_m / self.fy
        # 카메라 좌표계 3D 벡터
        P_cam = np.array([[X_m], [Y_m], [Z_m]], dtype=np.float32)

        # ----- 카메라 → 마커 좌표계 변환 -----
        # P_marker = R_inv * P_cam + T_inv  (단위: m)
        P_marker = self.R_inv @ P_cam + self.T_inv
        mx, my, mz = P_marker[0, 0], P_marker[1, 0], P_marker[2, 0]

        # ----- 마커 원점이 (0.2, 0, 0.2) 인 월드(로봇) 좌표로 변환 -----
        # 기존 hand-control 코드에서:
        #   world_x ≈ marker_y, world_y ≈ -marker_x 를 쓰고 있었으므로
        #   같은 축 정렬을 유지하면서 원점만 (0.2, 0, 0.2) 로 이동.
        real_x = 0.2 + my      # 마커 Y축 방향 → 로봇 X, 원점 0.2
        real_y = -mx           # 마커 X축 반대방향 → 로봇 Y
        real_z =  mz      # 마커 Z축 방향 → 로봇 Z, 원점 0.2

        # ----- 범위 제한 -----
        # x축 최대값 1.0, 최소값 0.2
        real_x = max(0.15, min(1.0, real_x))

        # x픽셀 기준 y값은 -6 ~ 6 (cm) → [-0.06, 0.06] m 로 클램프
        real_y = max(-0.6, min(0.6, real_y))

        # z축은 기존과 비슷하게 0.2 ~ 0.7 m
        real_z = max(0.2, min(0.6, real_z))

        # 출력용 반올림
        real_x_r = round(real_x, 3)
        real_y_r = round(real_y, 3)
        real_z_r = round(real_z, 3)

        # 디버그 렌더링
        result_img = self.color_image.copy()

        text = f"X={real_x_r:.3f}m Y={real_y_r:.3f}m Z={real_z_r:.3f}m"
        cv2.putText(result_img, text, (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(result_img, (u, v), 5, (0, 255, 0), -1)
        cv2.drawContours(result_img, [c], -1, (0, 255, 255), 2)

        # (선택) 마커 좌표축이 보이도록 drawFrameAxes는 update_aruco 쪽에서 이미 호출

        # 포인트 발행
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = self.frame_id
        point_msg.point.x = real_x_r
        point_msg.point.y = real_y_r
        point_msg.point.z = real_z_r
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
