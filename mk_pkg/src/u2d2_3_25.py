#!/usr/bin/env python3
import rospy
from dynamixel_sdk import *   # Uses Dynamixel SDK
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

import math

# ======== 포트 / 프로토콜 설정 ========
DEVICENAME = "/dev/u2d2"      # U2D2 포트
BAUDRATE   = 57600            # 필요시 1_000_000 등으로 변경
PROTOCOL_VERSION = 2.0        # MX-64(2.0) 은 Protocol 2.0

# ======== MX-64(2.0) Control Table 주소 ========
ADDR_TORQUE_ENABLE       = 64
ADDR_OPERATING_MODE      = 11
ADDR_PROFILE_ACCEL       = 108
ADDR_PROFILE_VELOCITY    = 112
ADDR_GOAL_POSITION       = 116
ADDR_PRESENT_CURRENT     = 126
ADDR_PRESENT_POSITION    = 132

LEN_GOAL_POSITION        = 4
LEN_PRESENT_POSITION     = 4
LEN_PRESENT_CURRENT      = 2

TORQUE_ENABLE            = 1
TORQUE_DISABLE           = 0
OPERATING_MODE_POSITION  = 3   # 또는 4(확장 위치) 사용 가능

# ======== 모터 설정 ========
DXL_IDS = [1, 2, 3, 4, 5, 6]

# 1~3은 감속기 6:1, 4~6은 1:1
GEAR_RATIO = [6.0, 6.0, 6.0, 1.0, 1.0, 1.0]

# ======== 속도 제한 (rad/s) ========
MAX_VEL_RAD = 1.0  # 요청: 속도는 1.0 rad/s 안으로

# MX-64(2.0) : Goal Velocity / Profile Velocity 1 tick ≈ 0.229 rev/min
#   => rad/s = 0.229 * 2π / 60 ≈ 0.02396
VEL_UNIT_RAD = 0.229 * 2.0 * math.pi / 60.0  # ≈ 0.02396 rad/s


class MX64Controller:
    def __init__(self):
        rospy.init_node("mx64_syncwrite_node", anonymous=True)

        # ---- Dynamixel Port/Packet Handler ----
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            rospy.logerr("Failed to open the port: %s", DEVICENAME)
            raise RuntimeError("Port open failed")
        rospy.loginfo("Succeeded to open the port: %s", DEVICENAME)

        if not self.portHandler.setBaudRate(BAUDRATE):
            rospy.logerr("Failed to change the baudrate")
            raise RuntimeError("Baudrate change failed")
        rospy.loginfo("Succeeded to change the baudrate: %d", BAUDRATE)

        # ---- GroupSyncWrite: Goal Position ----
        self.groupSyncWritePos = GroupSyncWrite(
            self.portHandler, self.packetHandler,
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )

        # ---- Publisher: /motor_state (JointState) ----
        self.state_pub = rospy.Publisher("/motor_state", JointState, queue_size=10)

        # ---- Subscribe: /arm_controller/command ----
        rospy.Subscriber("/arm_controller/command", JointTrajectory, self.traj_callback)

        # ---- 모터 초기화 ----
        self.init_motors()

        rospy.loginfo("MX64Controller ready. Subscribing to /arm_controller/command")
        rospy.Timer(rospy.Duration(0.02), self.publish_state)  # 50 Hz로 상태 발행
        rospy.spin()

    # ------------------------ 공통 유틸 ------------------------
    def dxl_write1(self, dxl_id, addr, data):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, addr, data
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("DXL ID %d write1Byte error: %s",
                          dxl_id, self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.logwarn("DXL ID %d write1Byte status error: %s",
                          dxl_id, self.packetHandler.getRxPacketError(dxl_error))

    def dxl_write2(self, dxl_id, addr, data):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, dxl_id, addr, data
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("DXL ID %d write2Byte error: %s",
                          dxl_id, self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.logwarn("DXL ID %d write2Byte status error: %s",
                          dxl_id, self.packetHandler.getRxPacketError(dxl_error))

    def dxl_write4(self, dxl_id, addr, data):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, dxl_id, addr, data
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("DXL ID %d write4Byte error: %s",
                          dxl_id, self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.logwarn("DXL ID %d write4Byte status error: %s",
                          dxl_id, self.packetHandler.getRxPacketError(dxl_error))

    def dxl_read2(self, dxl_id, addr):
        data, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.portHandler, dxl_id, addr
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn_throttle(1.0, "DXL ID %d read2Byte error: %s",
                                   dxl_id, self.packetHandler.getTxRxResult(dxl_comm_result))
            return 0
        if dxl_error != 0:
            rospy.logwarn_throttle(1.0, "DXL ID %d read2Byte status error: %s",
                                   dxl_id, self.packetHandler.getRxPacketError(dxl_error))
        return data

    def dxl_read4(self, dxl_id, addr):
        data, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, dxl_id, addr
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn_throttle(1.0, "DXL ID %d read4Byte error: %s",
                                   dxl_id, self.packetHandler.getTxRxResult(dxl_comm_result))
            return 0
        if dxl_error != 0:
            rospy.logwarn_throttle(1.0, "DXL ID %d read4Byte status error: %s",
                                   dxl_id, self.packetHandler.getRxPacketError(dxl_error))
        return data

    # rad → 틱
    def rad_to_tick(self, rad, gear):
        # 1 rev = 4096 tick → rad = 2π → 1 rad = 4096 / (2π)
        tick_per_rad = 4096.0 / (2.0 * math.pi)
        return int(rad * tick_per_rad * gear)

    # tick → rad
    def tick_to_rad(self, tick, gear):
        tick_per_rad = 4096.0 / (2.0 * math.pi)
        return float(tick) / (tick_per_rad * gear)

    # rad/s → Profile Velocity tick (단순 선형 변환)
    def radps_to_profile_vel(self, radps):
        # |v| > MAX_VEL_RAD 이면 clamp
        v = max(min(radps, MAX_VEL_RAD), -MAX_VEL_RAD)
        # rad/s / (rad/s per tick)
        ticks = int(abs(v) / VEL_UNIT_RAD)
        # Profile Velocity는 양수만 사용 (속도 크기)
        if ticks < 1:
            ticks = 1
        return ticks

    # ------------------------ 모터 초기화 ------------------------
    def init_motors(self):
        for dxl_id in DXL_IDS:
            # Torque OFF
            self.dxl_write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

            # Operating Mode: Position(3) 또는 Extended Position(4)
            self.dxl_write1(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)

            # 기본 Profile Acc / Vel 설정 (나중에 토픽 속도로 덮어씀)
            self.dxl_write4(dxl_id, ADDR_PROFILE_ACCEL, 50)
            self.dxl_write4(dxl_id, ADDR_PROFILE_VELOCITY, 50)

            # Torque ON
            self.dxl_write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

            rospy.loginfo("MX-64 ID %d initialized.", dxl_id)

    # ------------------------ Trajectory 콜백 ------------------------
    def traj_callback(self, msg):
        if not msg.points:
            return

        # 가장 마지막 point 사용 (혹은 첫 번째 point 사용해도 OK)
        point = msg.points[-1]

        if len(point.positions) < len(DXL_IDS):
            rospy.logwarn("JointTrajectory positions size < motor count")
            return

        # velocities가 없을 수도 있으므로 가드
        has_vel = len(point.velocities) >= len(DXL_IDS)

        # ===== 1) Profile Velocity 설정 =====
        for i, dxl_id in enumerate(DXL_IDS):
            if has_vel:
                radps = point.velocities[i]
            else:
                radps = MAX_VEL_RAD  # 속도 정보 없으면 최대값 사용

            prof_vel_tick = self.radps_to_profile_vel(radps)
            self.dxl_write4(dxl_id, ADDR_PROFILE_VELOCITY, prof_vel_tick)

        # ===== 2) GroupSyncWrite로 Goal Position 한 번에 전송 =====
        self.groupSyncWritePos.clearParam()

        for i, dxl_id in enumerate(DXL_IDS):
            rad = point.positions[i]
            gear = GEAR_RATIO[i]
            pos_tick = self.rad_to_tick(rad, gear)

            # 4 Byte little-endian
            param_goal_pos = [
                pos_tick & 0xFF,
                (pos_tick >> 8) & 0xFF,
                (pos_tick >> 16) & 0xFF,
                (pos_tick >> 24) & 0xFF
            ]

            add_ok = self.groupSyncWritePos.addParam(dxl_id, param_goal_pos)
            if not add_ok:
                rospy.logwarn("Failed to add param for DXL ID %d", dxl_id)

        dxl_comm_result = self.groupSyncWritePos.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("GroupSyncWrite txPacket failed: %s",
                          self.packetHandler.getTxRxResult(dxl_comm_result))

        rospy.loginfo_throttle(0.2, "Sent GroupSyncWrite GoalPosition with vel <= %.2f rad/s",
                               MAX_VEL_RAD)

    # ------------------------ 상태 발행 ------------------------
    def publish_state(self, event):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = [f"joint{i+1}" for i in range(len(DXL_IDS))]
        js.position = []
        js.velocity = []   # 여기서는 빈 리스트 (필요하면 Present Velocity 추가 가능)
        js.effort = []     # 전류(A)로 채움

        for i, dxl_id in enumerate(DXL_IDS):
            # 현재 위치
            present_pos_tick = self.dxl_read4(dxl_id, ADDR_PRESENT_POSITION)
            rad = self.tick_to_rad(present_pos_tick, GEAR_RATIO[i])
            js.position.append(rad)

            # 현재 전류 → A 로 환산 (약 3.36mA per tick)
            present_cur_tick = self.dxl_read2(dxl_id, ADDR_PRESENT_CURRENT)
            # tick는 부호 있는 값임 (양/음 토크)
            if present_cur_tick > 32767:
                present_cur_tick -= 65536  # signed 변환
            current_amp = present_cur_tick * 0.00336  # [A] 근사
            js.effort.append(current_amp)

        self.state_pub.publish(js)


if __name__ == "__main__":
    try:
        MX64Controller()
    except rospy.ROSInterruptException:
        pass
