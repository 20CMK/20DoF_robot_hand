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

# ========= 공통 설정 =========
PROTOCOL_VER  = 2.0

ID_START = 3
ID_END   = 25
DXL_IDS  = list(range(ID_START, ID_END + 1))

# Control Table (Protocol 2.0, X-series/XL330 기준)
ADDR_TORQUE_ENABLE      = 64
ADDR_OPERATING_MODE     = 11   # 일부 MX(2.0)에서 지원, 미지원이면 스킵
ADDR_CURRENT_LIMIT      = 38   # X-series 전용; MX64에선 생략
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

# ID 그룹
GROUP_A_IDS = {3, 4, 5}            # MX64 (Protocol 2.0)
GROUP_B_IDS = set(range(6, 26))    # XL330 등 X-series

# GROUP A (MX64) 기본값
GA_MODE   = 3      # Position Control
GA_ACC    = 100
GA_VEL    = 30
GA_P      = 400
GA_I      = 200
GA_D      = 200

# GROUP B (XL330) 기본값
GB_MODE   = 5      # Current-based Position
GB_CURR   = 1000   # 모델/전압에 맞게 조정
GB_ACC    = 40
GB_VEL    = 100    # XL330은 이 값을 계속 유지(콜백에서 갱신 안 함)
GB_P      = 100
GB_I      = 10
GB_D      = 50

# 라디안 -> 0~4095 (−180~+180도 매핑)
def rad_to_pos(rad: float) -> int:
    deg = rad * 180.0 / math.pi
    enc = int(round((deg + 180.0) * (4095.0 / 360.0)))
    if enc < 0: enc = 0
    if enc > 4095: enc = 4095
    return enc

# 부호 반전 규칙 (OpenCR 코드와 동일)
def maybe_flip_sign(dxl_id: int, rad: float) -> float:
    if dxl_id in (3, 4, 5, 6, 7, 8, 9, 22):
        return -rad
    return rad

class DXLController(object):
    def __init__(self):
        rospy.init_node("u2d2_dxl_joint_controller")

        # 파라미터는 init 이후 읽기 (launch에서 넘기기 좋음)
        self.port_name = rospy.get_param("~dxl_port", "/dev/u2d2")
        self.baudrate  = rospy.get_param("~dxl_baud", 1000000)
        # self.baudrate  = rospy.get_param("~dxl_baud", 57600)

        self.latest_positions  = [0.0] * 26
        self.latest_velocities = [0.0] * 26

        # DynamixelSDK 포트/패킷
        self.port   = PortHandler(self.port_name)
        self.packet = PacketHandler(PROTOCOL_VER)
        if not self.port.openPort():
            rospy.logerr("Failed to open the port: %s", self.port_name)
            raise SystemExit
        if not self.port.setBaudRate(self.baudrate):
            rospy.logerr("Failed to set baudrate: %d", self.baudrate)
            raise SystemExit

        # 초기 설정
        self.init_motors()

        # GroupSyncWrite (Goal Position)
        self.sync_goal_pos = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        # 준비 완료 후 구독 시작
        self.sub = rospy.Subscriber("joint_states", JointState, self.joint_cb, queue_size=10, tcp_nodelay=True)
        rospy.loginfo("Ready. Port=%s, Baud=%d, IDs=%s", self.port_name, self.baudrate, DXL_IDS)

    # 안전 쓰기 래퍼
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
        # 일부 모델은 Operating_Mode(11)가 없을 수 있음 → 실패해도 진행
        try:
            self.safe_write1(dxl_id, ADDR_OPERATING_MODE, mode)
            return True
        except Exception:
            rospy.logwarn("ID %d: Operating_Mode not supported; skipping.", dxl_id)
            return False

    def init_motors(self):
        rospy.loginfo("Initializing DXL IDs %s on %s @ %d", DXL_IDS, self.port_name, self.baudrate)

        # 토크 OFF
        for dxl_id in DXL_IDS:
            self.safe_write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)

        # ID별 파라미터 적용
        for dxl_id in DXL_IDS:
            if dxl_id in GROUP_A_IDS:
                # MX64 그룹 (Protocol 2.0) — 위치제어
                self.try_set_operating_mode(dxl_id, GA_MODE)
                self.safe_write4(dxl_id, ADDR_PROFILE_ACC, GA_ACC)
                self.safe_write4(dxl_id, ADDR_PROFILE_VEL, GA_VEL)
                self.safe_write2(dxl_id, ADDR_POS_P_GAIN, GA_P)
                self.safe_write2(dxl_id, ADDR_POS_I_GAIN, GA_I)
                self.safe_write2(dxl_id, ADDR_POS_D_GAIN, GA_D)
            else:
                # X-Series 그룹 (XL330) — 전류기반 위치제어
                self.try_set_operating_mode(dxl_id, GB_MODE)
                self.safe_write2(dxl_id, ADDR_CURRENT_LIMIT, GB_CURR)
                self.safe_write4(dxl_id, ADDR_PROFILE_ACC, GB_ACC)
                self.safe_write4(dxl_id, ADDR_PROFILE_VEL, GB_VEL)
                self.safe_write2(dxl_id, ADDR_POS_P_GAIN, GB_P)
                self.safe_write2(dxl_id, ADDR_POS_I_GAIN, GB_I)
                self.safe_write2(dxl_id, ADDR_POS_D_GAIN, GB_D)

        # 토크 ON
        for dxl_id in DXL_IDS:
            self.safe_write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

        rospy.loginfo("DXL init done.")

    def joint_cb(self, msg: JointState):
        # 준비 가드
        if not hasattr(self, "sync_goal_pos") or self.sync_goal_pos is None:
            rospy.logwarn_throttle(5.0, "sync_goal_pos not ready yet; skipping this message.")
            return

        # 배열 길이 방어
        npos = min(len(msg.position), 26)
        for i in range(npos):
            self.latest_positions[i] = msg.position[i]
        nvel = min(len(msg.velocity), 26) if len(msg.velocity) else 0
        for i in range(nvel):
            self.latest_velocities[i] = msg.velocity[i]

        # (옵션) MX64(3~5)만 velocity로 Profile_Vel 갱신
        if len(msg.velocity) >= 6:
            for dxl_id in (3, 4, 5):
                vel = abs(self.latest_velocities[dxl_id])  # rad/s
                # 스케일은 상황에 맞게 조절 (예: rad/s * 100)
                # vel_scaled = int(max(10, min(50, vel * 100)))
                vel = int(vel * 100)
                if vel > 300 : 
                    vel = 300
                elif vel < 10 :
                    vel = 10
                self.safe_write4(dxl_id, ADDR_PROFILE_VEL, vel)

        # 목표각 동기 전송
        self.sync_goal_pos.clearParam()

        for dxl_id in DXL_IDS:
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
