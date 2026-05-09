#!/usr/bin/env python3
import sys, time, numpy as np, cv2, rospy
from python_qt_binding.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDockWidget, QToolBar, QTabWidget, QFrame, QSizePolicy,
    QDoubleSpinBox, QGroupBox, QFormLayout
)
from python_qt_binding.QtGui import QColor, QImage, QPixmap
from python_qt_binding.QtCore import Qt, QTimer, pyqtSignal, QObject
import rviz

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import signal

# pyqtgraph (플롯) 사용 — 미설치 시 라벨 안내
try:
    import pyqtgraph as pg
    _HAVE_PG = True
except Exception:
    _HAVE_PG = False

# RViz 바인딩 환경 차이 흡수
rv = getattr(rviz, "bindings", rviz)

# ====================== RViz 뷰 ======================
class MinimalRobotView(QWidget):
    def __init__(self, fixed_frame="base_link"):
        super().__init__()
        vbox = QVBoxLayout(self)

        self.frame = rv.VisualizationFrame()
        self.frame.initialize()
        self.frame.setSplashPath("")
        vbox.addWidget(self.frame)

        self.manager = self.frame.getManager()
        root = self.manager.getRootDisplayGroup()

        # 기존 디스플레이 비활성화
        try:
            count = root.numDisplays()
        except Exception:
            count = 0

        def get_disp(idx):
            for name in ("getDisplayAt", "displayAt", "at"):
                if hasattr(root, name):
                    try:
                        return getattr(root, name)(idx)
                    except Exception:
                        pass
            return None

        for i in range(count):
            d = get_disp(i)
            if d:
                try:
                    d.setEnabled(False)
                except Exception:
                    pass

        # Global Options
        try:
            global_opts = root.subProp("Global Options")
            global_opts.subProp("Fixed Frame").setValue(fixed_frame)
            global_opts.subProp("Background Color").setValue(QColor(50, 50, 50))
        except Exception:
            pass

        # Grid + RobotModel
        grid = self.manager.createDisplay("rviz/Grid", "Grid", True)
        if grid:
            try:
                grid.subProp("Color").setValue(QColor(150, 150, 150))
                grid.subProp("Alpha").setValue(0.7)
            except Exception:
                pass

        robot = self.manager.createDisplay("rviz/RobotModel", "Robot", True)
        if robot:
            try:
                robot.subProp("Visual Enabled").setValue(True)
                robot.subProp("Collision Enabled").setValue(False)
                robot.subProp("Alpha").setValue(1.0)
            except Exception:
                pass
            
        fusion_pose = self.manager.createDisplay("rviz/Pose", "FusionPose", True)
        if fusion_pose:
            try:
                fusion_pose.subProp("Topic").setValue("/fusion_pose")   # 구독 토픽
                fusion_pose.subProp("Shape").setValue("Axes")            # 좌표축으로 표시
                fusion_pose.subProp("Axes Length").setValue(0.1)         # 축 길이 (m)
                fusion_pose.subProp("Axes Radius").setValue(0.01)        # 축 두께 (m)
                # 필요시 화살표로 보고 싶으면: fusion_pose.subProp("Shape").setValue("Arrow")
            except Exception:
                pass

        # 뷰 설정
        try:
            view_mgr = self.manager.getViewManager()
            view_mgr.setCurrentViewControllerType("rviz/Orbit")
            view = view_mgr.getCurrent()
            view.subProp("Target Frame").setValue(fixed_frame)
            view.subProp("Distance").setValue(2.5)
            view.subProp("Pitch").setValue(0.6)
            view.subProp("Yaw").setValue(-0.7)
            fp = view.subProp("Focal Point")
            fp.subProp("X").setValue(0.3)
            fp.subProp("Y").setValue(-0.2)
            fp.subProp("Z").setValue(0.3)
        except Exception:
            pass

        # RViz 내부 UI 숨기기
        for dock in self.frame.findChildren(QDockWidget): dock.hide()
        for tb in self.frame.findChildren(QToolBar): tb.hide()
        if hasattr(self.frame, "menuBar") and self.frame.menuBar(): self.frame.menuBar().hide()
        if hasattr(self.frame, "statusBar") and self.frame.statusBar(): self.frame.statusBar().hide()


# ====================== 영상 뷰어 (cv_bridge + QLabel) ======================
class _ImageBus(QObject):
    updated = pyqtSignal(QImage)

class RqtImageViewOnly(QWidget):
    """
    경량 이미지 뷰어
    - set_topic(topic): Image / CompressedImage 자동 처리
    - 콜백 → 시그널 → QLabel 갱신 (GUI 스레드 안전)
    """
    def __init__(self, topic):
        super().__init__()
        self._topic = None
        self._sub = None
        self._bridge = CvBridge()
        self._bus = _ImageBus()

        v = QVBoxLayout(self)
        self._label = QLabel("No Image")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background:#111; color:#bbb;")
        self._label.setMinimumHeight(120)
        self._label.setScaledContents(True)
        v.addWidget(self._label)

        self._bus.updated.connect(self._on_qimage)
        self.set_topic(topic)

    def _unsubscribe(self):
        if self._sub is not None:
            try:
                self._sub.unregister()
            except Exception:
                pass
            self._sub = None

    def set_topic(self, topic):
        if not topic or topic == self._topic:
            return
        self._unsubscribe()
        self._topic = topic

        is_compressed = topic.endswith("/compressed") or "compressed" in topic
        if is_compressed:
            self._sub = rospy.Subscriber(topic, RosCompressedImage, self._cb_compressed, queue_size=1)
        else:
            self._sub = rospy.Subscriber(topic, RosImage, self._cb_image, queue_size=1)

        self._label.setText(f"Subscribing: {topic}")

    def _cb_image(self, msg: RosImage):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._emit_qimage(cv_img)
        except Exception:
            pass

    def _cb_compressed(self, msg: RosCompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                self._emit_qimage(cv_img)
        except Exception:
            pass

    def _emit_qimage(self, cv_img_bgr):
        try:
            rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._bus.updated.emit(qimg.copy())
        except Exception:
            pass

    def _on_qimage(self, qimg: QImage):
        self._label.setPixmap(QPixmap.fromImage(qimg))


# ====================== 조인트 플롯 (pyqtgraph) ======================
class JointPlotWidget(QWidget):
    """
    /joint_states에서 원하는 인덱스의 position을 실시간 플로팅 (기본: 1, 2번)
    - pyqtgraph 단일 구현 (간단/가벼움)
    """
    def __init__(self, indices=(1, 2), buffer_len=1000, hz=10.0, window_span=5.0):
        super().__init__()
        v = QVBoxLayout(self)

        if not _HAVE_PG:
            v.addWidget(QLabel("[pyqtgraph 미설치]  pip install pyqtgraph"))
            self.setLayout(v)
            return

        self.indices = indices
        self.buffer_len = buffer_len
        self.period_ms = int(1000.0 / hz)
        self.window_span = window_span

        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'position (rad)')
        self.plot.setLabel('bottom', 'time (s)')
        self.plot.setLimits(xMin=0)
        self.plot.setYRange(-3.0, 3.0, padding=0)
        self.plot.addLegend(offset=(-10, -10))
        v.addWidget(self.plot)

        # 데이터 버퍼
        self.t0 = time.time()
        self.ts = []
        self.data = [[] for _ in self.indices]

        # 곡선
        colors = ['r', 'g', 'b', 'm', 'c']
        names  = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        self.curves = [
            self.plot.plot(
                [], [],
                pen=pg.mkPen(color=colors[k % len(colors)], width=2),
                name=names[k % len(names)]
            )
            for k in range(len(self.indices))
        ]

        # ROS 구독
        self.sub = rospy.Subscriber("/joint_states", JointState, self._cb_joint, queue_size=20)

        # 타이머 갱신
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(self.period_ms)

    def _cb_joint(self, msg: JointState):
        t = time.time() - self.t0
        self.ts.append(t)

        for k, idx in enumerate(self.indices):
            v = float('nan')
            try:
                if msg.position and len(msg.position) > idx:
                    v = msg.position[idx]
            except Exception:
                pass
            self.data[k].append(v)

        # 버퍼 제한
        if len(self.ts) > self.buffer_len:
            self.ts = self.ts[-self.buffer_len//2:]
            for k in range(len(self.data)):
                self.data[k] = self.data[k][-self.buffer_len//2:]

    def _refresh(self):
        if not _HAVE_PG:
            return
        if len(self.ts) > 0:
            xmax = self.ts[-1]
            xmin = xmax - self.window_span
            self.plot.setXRange(xmin, xmax, padding=0)

        for k, curve in enumerate(self.curves):
            ys = self.data[k] if k < len(self.data) else []
            curve.setData(self.ts, ys)


# ====================== Pose 플롯 (PoseStamped) ======================
class PosePlotWidget(QWidget):
    """
    /fusion_pose (geometry_msgs/PoseStamped) 구독해서 position.x/y/z를 실시간 플롯
    fields: ("x","y","z") 또는 ("ox","oy","oz","ow") 등 지정 가능
    """
    def __init__(self, topic="/fusion_pose", fields=("x","y"), buffer_len=1000, hz=10.0, window_span=5.0):
        super().__init__()
        v = QVBoxLayout(self)

        if not _HAVE_PG:
            v.addWidget(QLabel("[pyqtgraph 미설치]  pip install pyqtgraph"))
            self.setLayout(v)
            return

        self.topic = topic
        self.fields = fields
        self.buffer_len = buffer_len
        self.period_ms = int(1000.0 / hz)
        self.window_span = window_span

        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'value')
        self.plot.setLabel('bottom', 'time (s)')
        self.plot.setLimits(xMin=0)
        self.plot.setYRange(-0.8, 0.8, padding=0)   # 필요시 조정
        self.plot.addLegend(offset=(-10, -10))
        v.addWidget(self.plot)

        # 데이터 버퍼
        self.t0 = time.time()
        self.ts = []
        self.data = [[] for _ in self.fields]

        # 곡선 (색상 + 이름)
        colors   = ['r', 'b', 'm', 'c', 'g']
        name_map = {
            "x": "pos.x", "y": "pos.y", "z": "pos.z",
            "ox": "ori.x", "oy": "ori.y", "oz": "ori.z", "ow": "ori.w",
        }
        self.curves = [
            self.plot.plot(
                [], [],
                pen=pg.mkPen(color=colors[k % len(colors)], width=2),
                name=name_map.get(self.fields[k], self.fields[k])
            )
            for k in range(len(self.fields))
        ]

        # ROS 구독
        self.sub = rospy.Subscriber(self.topic, PoseStamped, self._cb_pose, queue_size=20)

        # 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(self.period_ms)

    def _cb_pose(self, msg: PoseStamped):
        t = time.time() - self.t0
        self.ts.append(t)

        p = msg.pose.position
        o = msg.pose.orientation
        value_map = {
            "x": p.x, "y": p.y, "z": p.z,
            "ox": o.x, "oy": o.y, "oz": o.z, "ow": o.w,
        }
        for k, f in enumerate(self.fields):
            self.data[k].append(value_map.get(f, float('nan')))

        # 버퍼 제한
        if len(self.ts) > self.buffer_len:
            self.ts = self.ts[-self.buffer_len//2:]
            for k in range(len(self.data)):
                self.data[k] = self.data[k][-self.buffer_len//2:]

    def _refresh(self):
        if not _HAVE_PG:
            return
        if len(self.ts) > 0:
            xmax = self.ts[-1]
            xmin = xmax - self.window_span
            self.plot.setXRange(xmin, xmax, padding=0)

        for k, curve in enumerate(self.curves):
            ys = self.data[k] if k < len(self.data) else []
            curve.setData(self.ts, ys)








# ====================== 두 번째 탭 (텍스트) ======================
class SecondTab(QWidget):
    def __init__(self):
        super().__init__()
        vbox = QVBoxLayout(self)
        label = QLabel("두 번째 GUI 탭")
        label.setStyleSheet("font-size: 200px; color: black;")
        vbox.addWidget(label)


# ====================== 메인 윈도우 ======================
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt + RViz + Image + Plot (정리 버전)")

        # --- ROS params & publishers ---
        self.switch_topic   = rospy.get_param("~switch_topic", "/switch")
        self.imu_mode_topic = rospy.get_param("~imu_mode_topic", "/imu_reset_mode")
        self.traj_topic     = rospy.get_param("~traj_topic", "/arm_controller/command")
        self.joint_names    = rospy.get_param("~joint_names",
                                              ["joint1","joint2","joint3","joint4","joint5","joint6"])

        self.switch_pub   = rospy.Publisher(self.switch_topic, Bool, queue_size=10)
        self.traj_pub     = rospy.Publisher(self.traj_topic, JointTrajectory, queue_size=10)
        self.imu_mode_pub = rospy.Publisher(self.imu_mode_topic, Int32, queue_size=10)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # ----- Tab 1: 왼쪽 RViz / 오른쪽: 이미지 2개 + 버튼 -----
        tab1_container = QWidget()
        hroot = QHBoxLayout(tab1_container)
        left_group = QWidget()
        vleft = QVBoxLayout(left_group)

        # 왼쪽: RViz + (플롯 2개 가로 배치)
        arm_rviz = MinimalRobotView(fixed_frame="base_link")
        vleft.addWidget(arm_rviz, 2)

        plot  = JointPlotWidget(indices=(1, 2), buffer_len=1000, hz=10.0)
        plot2 = PosePlotWidget(topic="/fusion_pose", fields=("x","y"), buffer_len=1000, hz=10.0, window_span=5.0)

        plot_group = QWidget()
        hplot = QHBoxLayout(plot_group)
        hplot.addWidget(plot, 3)
        hplot.addWidget(plot2, 3)
        
        
        #################################
        # ★★★ 추가: 각도(6) + 시간(1) + 확인 버튼 UI
        inputs_box = QGroupBox("User Input")
        form = QFormLayout(inputs_box)

        # 6개 각도 스핀박스 (기본값 0°, -360~360, 소수 1자리)
        self.angle_spins = []
        for i in range(6):
            sb = QDoubleSpinBox()
            sb.setRange(-180.0, 180.0)
            sb.setDecimals(1)
            sb.setSingleStep(1.0)
            sb.setValue(0.0)  # 기본값 0°
            sb.setSuffix(" °")
            sb.setAlignment(Qt.AlignRight)
            sb.setFixedHeight(30)
            self.angle_spins.append(sb)
            form.addRow(f"joint{i+1} (deg)", sb)

        # 시간 스핀박스 (기본 5.0초, 0.5~30)
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.5, 30.0)
        self.time_spin.setDecimals(2)
        self.time_spin.setSingleStep(0.1)
        self.time_spin.setValue(5.0)
        self.time_spin.setSuffix(" s")
        self.time_spin.setAlignment(Qt.AlignRight)
        self.time_spin.setFixedHeight(30)
        form.addRow("time_from_start (s)", self.time_spin)
        
        # ★★★ 가속도 스핀박스
        self.acc_spin = QDoubleSpinBox()
        self.acc_spin.setRange(0.0, 100.0)  
        self.acc_spin.setDecimals(2)
        self.acc_spin.setSingleStep(0.1)
        self.acc_spin.setValue(0.0)         # 기본값
        # self.acc_spin.setSuffix(" s2")
        self.acc_spin.setAlignment(Qt.AlignRight)
        self.acc_spin.setFixedHeight(30)
        form.addRow("Acceleration", self.acc_spin)

        # 확인 버튼
        self.btn_confirm = QPushButton("Move")
        self.btn_confirm.setStyleSheet("""
            QPushButton {
                font-size: 14px;       /* ← 글자 크기 */
                padding: 14px;          /* 버튼 내부 여백 */
            }
        """)
        form.addRow(self.btn_confirm)

        hplot.addWidget(inputs_box, 1)
        
        ##################################
        
        
        
        
        vleft.addWidget(plot_group, 1)

        # 오른쪽: 영상2 + 버튼 그룹
        right_group = QWidget()
        vright = QVBoxLayout(right_group)

        view1 = RqtImageViewOnly(topic="/camera/color/image_raw/compressed")
        view2 = RqtImageViewOnly(topic="/wrist_detector/debug_image/compressed")
        vright.addWidget(view1, 2)
        vright.addWidget(view2, 2)

        # 버튼 행 1
        button_group_1 = QWidget()
        hbtn_1 = QHBoxLayout(button_group_1)

        self.btn_zero      = QPushButton("Zero Pose")
        self.btn_ready     = QPushButton("Ready Pose")
        self.btn_imu_reset = QPushButton("IMU\nReset")

        hbtn_1.addWidget(self.btn_zero)
        hbtn_1.addWidget(self.btn_ready)
        hbtn_1.addWidget(self.btn_imu_reset)

        # 버튼 행 2
        button_group_2 = QWidget()
        hbtn_2 = QHBoxLayout(button_group_2)

        self.btn_on  = QPushButton("ON")
        self.btn_off = QPushButton("OFF")

        hbtn_2.addWidget(self.btn_on)
        hbtn_2.addWidget(self.btn_off)

        hbtn_1.setSpacing(8); hbtn_2.setSpacing(8)
        for layout in (hbtn_1, hbtn_2):
            for i in range(layout.count()):
                btn = layout.itemAt(i).widget()
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        vbtn_group = QWidget()
        vbtn = QVBoxLayout(vbtn_group)
        vbtn.addWidget(button_group_1, 2)
        vbtn.addWidget(button_group_2, 2)

        self.color_box = QFrame()
        self.color_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.set_state_color("red")
        vbtn.addWidget(self.color_box, 1)

        vright.addWidget(vbtn_group, 1)



        # 좌/우 붙이기
        hroot.addWidget(left_group, 2)
        hroot.addWidget(right_group, 1)

        tabs.addTab(tab1_container, "Arm GUI")

        # ----- Tab 2: 텍스트 탭 -----
        tab2 = SecondTab()
        tabs.addTab(tab2, "AMR GUI")

        tabs.setCurrentIndex(0)

        # --- 버튼 클릭 -> 퍼블리시 연결 ---
        self.btn_on.clicked.connect(lambda: self.publish_switch(True))
        self.btn_off.clicked.connect(lambda: self.publish_switch(False))
        self.btn_imu_reset.clicked.connect(self.publish_imu_reset)

        # Zero/Ready Pose: JointTrajectory 발행
        self.btn_zero.clicked.connect(lambda: self.publish_traj([0.0]*len(self.joint_names)))
        self.btn_confirm.clicked.connect(self.publish_arm_command_from_inputs)


        ready_deg = [90.0, 45.0, 90.0, 0.0, 45.0, 90.0]
        ready_rad = [d*np.pi/180.0 for d in ready_deg]
        self.btn_ready.clicked.connect(lambda: self.publish_traj(ready_rad))
        
        self.btn_on.clicked.connect(lambda: self.set_state_color("green"))
        self.btn_off.clicked.connect(lambda: self.set_state_color("red"))
        self.btn_imu_reset.clicked.connect(lambda: self.set_state_color("red"))
        self.btn_zero.clicked.connect(lambda: self.set_state_color("red"))
        self.btn_ready.clicked.connect(lambda: self.set_state_color("red"))
        self.btn_confirm.clicked.connect(lambda: self.set_state_color("red"))
        
    # --------- 발행 함수들 ---------
    def publish_switch(self, state: bool):
        """ON=True / OFF=False"""
        try:
            self.switch_pub.publish(Bool(data=state))
        except Exception as e:
            rospy.logerr(f"switch_pub error: {e}")

    def publish_imu_reset(self):
        """IMU Reset 모드: 1 발행"""
        try:
            self.switch_pub.publish(Bool(data=False))
            self.imu_mode_pub.publish(Int32(data=1))
        except Exception as e:
            rospy.logerr(f"imu_mode_pub error: {e}")

    def publish_traj(self, positions):
        """JointTrajectory 발행: positions 길이는 joint_names에 맞춰 패딩/자름"""
        try:
            self.switch_pub.publish(Bool(data=False))
            n = len(self.joint_names)
            pos = list(positions[:n]) + [0.0]*max(0, n - len(positions))

            traj = JointTrajectory()
            traj.joint_names = list(self.joint_names)

            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.time_from_start = rospy.Duration(5.0)

            traj.points.append(pt)
            traj.header.stamp = rospy.Time.now()
            self.traj_pub.publish(traj)
        except Exception as e:
            rospy.logerr(f"traj_pub error: {e}")

    def set_state_color(self, color: str):
        self.color_box.setObjectName("stateBox")
        self.color_box.setStyleSheet(f"""
            #stateBox {{
                background-color: {color};
                border: 1px solid #333;
            }}
        """)

            
            
    def publish_arm_command_from_inputs(self):
        try:
            self.switch_pub.publish(Bool(data=False))
            
            # angles: deg -> rad
            degs = [sb.value() for sb in self.angle_spins]
            rads = [d*np.pi/180.0 for d in degs]

            # ★★★ acc: deg/s^2 -> rad/s^2
            acc_deg = float(self.acc_spin.value())
            acc_rad = acc_deg * np.pi / 180.0

            duration = max(0.0, float(self.time_spin.value()))

            n = len(self.joint_names)
            pos = list(rads[:n]) + [0.0]*max(0, n - len(rads))
            acc = [acc_rad] * n  # 모든 조인트 동일 가속도

            traj = JointTrajectory()
            traj.joint_names = list(self.joint_names)

            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.accelerations = acc             # ★★★ 가속도 세팅
            pt.time_from_start = rospy.Duration(duration)

            traj.points.append(pt)
            traj.header.stamp = rospy.Time.now()

            self.traj_pub.publish(traj)

        except Exception as e:
            rospy.logerr(f"/arm_command publish error: {e}")
            
# if __name__ == "__main__":
#     rospy.init_node("pyqt_rviz_image_plot_clean", anonymous=True)
#     app = QApplication(sys.argv)
#     win = Main()
#     win.showMaximized()
#     win.show()
#     sys.exit(app.exec_())


if __name__ == "__main__":
    rospy.init_node("pyqt_rviz_image_plot_clean", anonymous=True)
    app = QApplication(sys.argv)
    win = Main()
    win.showMaximized()
    win.show()

    # a) Ctrl+C(Sigint) 들어오면 ROS 먼저 내리기
    def _handle_sigint(*_):
        rospy.signal_shutdown("SIGINT")

    signal.signal(signal.SIGINT, _handle_sigint)

    # b) ROS가 내려가면 Qt도 함께 종료
    rospy.on_shutdown(app.quit)

    # c) 주기적으로 ROS 상태 체크해서 내려가면 quit
    t = QTimer()
    t.timeout.connect(lambda: (app.quit() if rospy.is_shutdown() else None))
    t.start(50)  # ms

    sys.exit(app.exec_())
