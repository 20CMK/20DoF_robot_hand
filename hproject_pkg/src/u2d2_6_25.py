#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from sensor_msgs.msg import JointState

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncWrite,
    COMM_SUCCESS
)

# ========= 사용자 설정 =========
PORT_NAME     = rospy.get_param("/dxl_port", "/dev/ttyUSB0")   # U2D2 포트
BAUDRATE      = rospy.get_param("/dxl_baud", 1000000)
PROTOCOL_VER  = 2.0

ID_START = 6
ID_END   = 25
DXL_IDS  = list(range(ID_START, ID_END + 1))

# Control Table (Protocol 2.0, X-series/XL330 기준)
ADDR_TORQUE_ENABLE      = 64
ADDR_OPERATING_MODE     = 11  # X-series만 해당. 구형 MX 계열은 없을 수 있음
ADDR_CURRENT_LIMIT      = 38  # XL330: 2 bytes (단위 주의)
ADDR_PROFILE_ACC        = 108
ADDR_PROFILE_VEL        = 112
ADDR_POS_P_GAIN         = 84
ADDR_POS_I_GAIN         = 82
ADDR_POS_D_GAIN         = 80
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

LEN_TORQUE_ENABLE       = 1
LEN_OPERATING_MODE      = 1
LEN_CURRENT_LIMIT       = 2
LEN_PROFILE_ACC         = 4
LEN_PROFILE_VEL         = 4
LEN_GOAL_POSITION       = 4

TORQUE_ON  = 1
TORQUE_OFF = 0

# 기본 파라미터 (필요에 맞게 조절하세요)
DEFAULT_MODE            = 5      # 현재기반 위치제어 (XL330 권장)
DEFAULT_CURRENT_LIMIT   = 1000   # XL330 기준 예시(단위는 모델별 다름)
DEFAULT_PROFILE_ACC     = 40     # XL330에서 쓰던 값
DEFAULT_PROFILE_VEL     = 100
DEFAULT_P               = 100
DEFAULT_I               = 10
DEFAULT_D               = 50

# 라디안 -> 엔코더 변환 (X-series/XL330: 0~4095 -> 0~360deg)
def rad_to_pos(rad):
    deg = rad * 180.0 / math.pi
    # -180 ~ +180 -> 0 ~ 4095
    enc = int(round((deg + 180.0) * (4095.0 / 360.0)))
    # 클리핑
    if enc < 0: enc = 0
    if enc > 4095: enc = 4095
    return enc

# 기존 OpenCR 코드에서 쓰던 부호 반전 규칙을 그대로 적용
def maybe_flip_sign(dxl_id, rad):
    if dxl_id in (6, 7, 8, 9, 22):
        return -rad
    return rad

class DXLController(object):
    def __init__(self):
        # ROS
        rospy.init_node("u2d2_dxl_joint_controller")
        
        self.latest_positions = [0.0] * 26
        self.latest_velocities = [0.0] * 26

        # DynamixelSDK
        self.port = PortHandler(PORT_NAME)
        self.packet = PacketHandler(PROTOCOL_VER)
        if not self.port.openPort():
            rospy.logerr("Failed to open the port: %s", PORT_NAME)
            raise SystemExit
        if not self.port.setBaudRate(BAUDRATE):
            rospy.logerr("Failed to set baudrate: %d", BAUDRATE)
            raise SystemExit

        # 초기 설정
        self.init_motors()

        # GroupSyncWrite 준비 (Goal Position)
        self.sync_goal_pos = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        
        self.sub = rospy.Subscriber("joint_states", JointState, self.joint_cb, queue_size=10)


    def safe_write1(self, dxl_id, addr, data):
        dxl_comm_result, dxl_error = self.packet.write1ByteTxRx(self.port, dxl_id, addr, data)
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("ID %d write1 addr %d comm fail: %s", dxl_id, addr, self.packet.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.logwarn("ID %d write1 addr %d error: %s", dxl_id, addr, self.packet.getRxPacketError(dxl_error))

    def safe_write2(self, dxl_id, addr, data):
        dxl_comm_result, dxl_error = self.packet.write2ByteTxRx(self.port, dxl_id, addr, data)
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("ID %d write2 addr %d comm fail: %s", dxl_id, addr, self.packet.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.logwarn("ID %d write2 addr %d error: %s", dxl_id, addr, self.packet.getRxPacketError(dxl_error))

    def safe_write4(self, dxl_id, addr, data):
        dxl_comm_result, dxl_error = self.packet.write4ByteTxRx(self.port, dxl_id, addr, data)
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("ID %d write4 addr %d comm fail: %s", dxl_id, addr, self.packet.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.logwarn("ID %d write4 addr %d error: %s", dxl_id, addr, self.packet.getRxPacketError(dxl_error))

    def try_set_operating_mode(self, dxl_id, mode):
        # 일부 모델(MX(2.0) 미적용 등)은 Operating_Mode(11)가 없을 수 있음 → 실패해도 넘어감
        try:
            self.safe_write1(dxl_id, ADDR_OPERATING_MODE, mode)
            return True
        except Exception:
            rospy.logwarn("ID %d: Operating_Mode not supported; skipping.", dxl_id)
            return False

    def init_motors(self):
        rospy.loginfo("Initializing DXL IDs %s on %s @ %d", DXL_IDS, PORT_NAME, BAUDRATE)

        # 토크 OFF
        for dxl_id in DXL_IDS:
            self.safe_write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)

        # 모드/리밋/프로파일/게인 설정
        for dxl_id in DXL_IDS:
            self.try_set_operating_mode(dxl_id, DEFAULT_MODE)  # 실패해도 진행

            # (가능 모델에서만) 전류리밋/프로파일/게인 세팅
            self.safe_write2(dxl_id, ADDR_CURRENT_LIMIT, DEFAULT_CURRENT_LIMIT)
            self.safe_write4(dxl_id, ADDR_PROFILE_ACC, DEFAULT_PROFILE_ACC)
            self.safe_write4(dxl_id, ADDR_PROFILE_VEL, DEFAULT_PROFILE_VEL)
            self.safe_write2(dxl_id, ADDR_POS_P_GAIN, DEFAULT_P)
            self.safe_write2(dxl_id, ADDR_POS_I_GAIN, DEFAULT_I)
            self.safe_write2(dxl_id, ADDR_POS_D_GAIN, DEFAULT_D)

        # 토크 ON
        for dxl_id in DXL_IDS:
            self.safe_write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

        rospy.loginfo("DXL init done.")

    def joint_cb(self, msg):
        # 0) 준비 가드 (제일 먼저!)
        if not hasattr(self, "sync_goal_pos") or self.sync_goal_pos is None:
            rospy.logwarn_throttle(5.0, "sync_goal_pos not ready yet; skipping this message.")
            return

        # 1) 배열 길이 방어
        npos = min(len(msg.position), 26)
        for i in range(npos):
            self.latest_positions[i] = msg.position[i]
        nvel = min(len(msg.velocity), 26) if len(msg.velocity) else 0
        for i in range(nvel):
            self.latest_velocities[i] = msg.velocity[i]

        # 2) (옵션) 프로파일 속도 갱신
        #    velocities 길이가 충분할 때만 안전하게 갱신
        if len(msg.velocity) >= (ID_END + 1):
            for dxl_id in DXL_IDS:
                vel = abs(self.latest_velocities[dxl_id])  # rad/s
                vel_scaled = int(max(5, min(200, vel * 100)))  # 필요 시 스케일 조정
                self.safe_write4(dxl_id, ADDR_PROFILE_VEL, vel_scaled)

        # 3) 목표각 동기 전송
        self.sync_goal_pos.clearParam()

        for dxl_id in DXL_IDS:
            # latest_positions 인덱스는 0~25, ID는 6~25 → 바로 인덱싱 OK
            rad = self.latest_positions[dxl_id] if dxl_id < len(self.latest_positions) else 0.0
            rad = maybe_flip_sign(dxl_id, rad)
            pos = rad_to_pos(rad)

            param = bytearray([
                (pos     ) & 0xFF,
                (pos >> 8) & 0xFF,
                (pos >>16) & 0xFF,
                (pos >>24) & 0xFF
            ])

            ok = self.sync_goal_pos.addParam(dxl_id, param)
            if not ok:
                rospy.logwarn("GroupSyncWrite addParam failed for ID %d", dxl_id)

        dxl_comm_result = self.sync_goal_pos.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            rospy.logwarn("GroupSyncWrite txPacket failed: %s",
                        self.packet.getTxRxResult(dxl_comm_result))


    def spin(self):
        rospy.loginfo("u2d2_dxl_joint_controller running. Subscribing /joint_states")
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == "__main__":
    try:
        node = DXLController()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except SystemExit:
        pass
