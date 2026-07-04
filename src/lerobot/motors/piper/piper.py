import math
import time
from dataclasses import dataclass
from typing import Dict

from piper_sdk import C_PiperInterface_V2


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]

class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        # 录制数据集时改成0
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool, timeout: float = 10) -> bool:
        '''
            使能机械臂并检测使能状态; 若使能超时则返回 False (由调用方重试)。
            timeout: 使能等待上限(秒)。从冷/失能状态使能较慢, 给足时间避免误判未使能。
        '''
        enable_flag = False
        loop_flag = False
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                # EnablePiper() returns the PRE-command enable status, so it only
                # returns True once all 6 motors already report enabled. Bound this
                # wait by the outer timeout: if the arm never enables (enable frame
                # dropped in the connect burst, arm unpowered/e-stopped, driver
                # fault, or status frames not arriving) an UNBOUNDED loop here hangs
                # forever printing "piper initing" and blocks both the outer timeout
                # and the caller's retry. On timeout, fall through -> outer loop
                # returns False -> caller retries / raises loudly.
                while( not self.piper.EnablePiper()):
                    if time.time() - start_time > timeout:
                        print("enable timed out (motors never reported enabled)")
                        break
                    print('piper initing')
                    time.sleep(0.1)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        if enable and resp:
            # Readiness gate: do NOT hand control to teleop/record until the arm is
            # actually READY and STAYS ready. Declaring success the instant enable
            # first reads True (then a blind sleep) causes the intermittent "follower
            # sometimes not activated -> leader can't control it". Three cold-start
            # races feed that flake, all covered by requiring the ready condition to
            # hold *continuously* for a short window:
            #   1. driver_enable_status can read True for a single sample before the
            #      driver is really holding torque -> require it STABLE, not one-shot.
            #   2. an EnablePiper() frame issued during the 4-arm connect burst can be
            #      dropped (ENOBUFS / SEND_MESSAGE_FAILED), so the enable doesn't stick
            #      -> a dropout resets the window and the loop re-confirms.
            #   3. on the first process after CAN (re)activation joint feedback isn't
            #      flowing yet (GetArmJointMsgs().Hz == 0), so a leader would stream
            #      stale zeros / a follower would record stale state -> require Hz > 0.
            if not self._wait_until_ready(timeout=5.0, stable_needed=0.6):
                # Never became stably ready in time. Report failure so the caller's
                # 3x retry re-runs connect() (re-issues EnablePiper) instead of
                # handing a not-actually-holding arm to teleop.
                print("arm not stably ready (enable+feedback) within deadline; failing connect to trigger retry")
                return False
        print(f"Returning response: {resp}")
        return resp

    def _all_motors_enabled(self) -> bool:
        """True iff all 6 joint drivers currently report enabled."""
        info = self.piper.GetArmLowSpdInfoMsgs()
        return all(
            getattr(info, f"motor_{i}").foc_status.driver_enable_status
            for i in range(1, 7)
        )

    def is_ready(self) -> bool:
        """Arm is holding torque (all drivers enabled) AND joint feedback is live.

        Cheap, side-effect-free snapshot -- safe to poll as a readiness barrier
        before starting teleop.
        """
        try:
            return self._all_motors_enabled() and self.piper.GetArmJointMsgs().Hz > 0
        except Exception:
            return False

    def _wait_until_ready(self, timeout: float = 5.0, stable_needed: float = 0.6) -> bool:
        """Block until is_ready() holds continuously for `stable_needed` seconds.

        Any dropout (enable flicker, feedback stall) resets the stability window,
        so a single lucky sample can't declare the arm ready. Returns False if it
        never stabilizes within `timeout`.
        """
        deadline = time.time() + timeout
        stable_since = None
        while time.time() < deadline:
            if self.is_ready():
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_needed:
                    return True
            else:
                stable_since = None
            time.sleep(0.05)
        return False

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def move_to_home(self, timeout: float = 10.0, tol_rad: float = 0.05, speed: int = 50) -> bool:
        """Drive all joints to home (0) under position control, REPEATING the
        command until the arm actually reaches home (verified from joint feedback)
        or `timeout`. Returns True iff home was reached.

        Why polled instead of a single write(): right after enable the arm is not
        yet in CAN position-control mode, so a lone MotionCtrl_2+JointCtrl often
        does not take -- the mode switch is still settling, or the frame is dropped
        in the 4-arm connect burst. The follower then never moves to home AND
        ignores the teleop JointCtrl that follow ("follower doesn't move to home =>
        teleop dead"). Repeating the command forces the mode switch to register and
        proves the follower is accepting motion before teleop starts. Mirrors the
        proven go-home phase of gentle_disable().
        """
        RAW_TO_RAD = 0.001 * math.pi / 180.0

        def joints_rad() -> list[float]:
            js = self.piper.GetArmJointMsgs().joint_state
            return [round(getattr(js, f"joint_{i}") * RAW_TO_RAD, 3) for i in range(1, 7)]

        def status_str() -> str:
            st = self.piper.GetArmStatus().arm_status
            return f"ctrl_mode=0x{st.ctrl_mode:02X} arm_status=0x{st.arm_status:02X}"

        start_joints = joints_rad()
        start = time.time()
        cmds_sent = 0
        while time.time() - start < timeout:
            self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
            self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            cmds_sent += 1
            time.sleep(0.05)
            js = self.piper.GetArmJointMsgs().joint_state
            if max(abs(getattr(js, f"joint_{i}") * RAW_TO_RAD) for i in range(1, 7)) < tol_rad:
                # Success by position -- but if the arm STARTED within tolerance
                # this proved nothing about command acceptance, so surface the
                # arm's actual control mode: ctrl_mode must be 0x01 (CAN command
                # control) for teleop JointCtrl to be obeyed. "home ok after 1
                # cmd + ctrl_mode=0x00" = arm at home but ignoring commands ->
                # teleop will be dead despite connect() succeeding.
                print(f"move_to_home ok after {cmds_sent} cmd(s): {status_str()} start={start_joints}")
                return True
        # Timed-out diagnostic: unchanged joints = arm ignored position commands
        # (mode never registered); moved-but-short = just too slow for timeout.
        print(
            f"move_to_home timed out after {timeout}s: {status_str()} "
            f"joints(rad) start={start_joints} end={joints_rad()}"
        )
        return False

    def apply_calibration(self) -> bool:
        """移动到初始位置 (follower). Polled + verified: returns True iff home reached."""
        return self.move_to_home()


    def apply_calibration_master(self):
        """
            master移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 -92000 ~ 92000 / 57324.840764
                joint_2 (float): 关节2角度 -2400 ~ 120000 / 57324.840764
                joint_3 (float): 关节3角度 3000 ~ -110000 / 57324.840764
                joint_4 (float): 关节4角度 -90000 ~ 90000 / 57324.840764
                joint_5 (float): 关节5角度 80000 ~ -80000 / 57324.840764
                joint_6 (float): 关节6角度 -90000 ~ 90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """
        joint_0 = round(target_joint[0]*self.joint_factor)
        joint_1 = round(target_joint[1]*self.joint_factor)
        joint_2 = round(target_joint[2]*self.joint_factor)
        joint_3 = round(target_joint[3]*self.joint_factor)
        joint_4 = round(target_joint[4]*self.joint_factor)
        joint_5 = round(target_joint[5]*self.joint_factor)
        gripper_range = round(target_joint[6]*1000*1000)
        
        self.piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位 0.001°

    def read(self) -> Dict:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        return {
            "joint_1": joint_state.joint_1,
            "joint_2": joint_state.joint_2,
            "joint_3": joint_state.joint_3,
            "joint_4": joint_state.joint_4,
            "joint_5": joint_state.joint_5,
            "joint_6": joint_state.joint_6,
            "gripper": gripper_state.grippers_angle
        }


    def safe_disconnect(self):
        """
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)

    def gentle_disable(self, kp0: float = 10.0, kd: float = 0.8, duration: float = 2.0,
                       go_home: bool = True, home_speed: int = 15, settle: float = 0.6):
        """
            软失能: 避免机械臂直接断电自由落体硬砸下去。
            流程 (对应 utils/gentle_disable_arm.py):
              1. (可选) 用位置控制缓慢回到 home 姿态;
              2. 用 MIT 力控保持当前关节角, 把位置增益 kp 在 duration 秒内线性降到 0,
                 保留阻尼 kd, 机械臂被阻尼缓慢放下;
              3. kp=0 只留阻尼沉降一小段, 最后真正失能并恢复位置/速度控制模式。

            kp0  初始保持增益 (SDK 参考 10)
            kd   阻尼增益 (SDK 参考 0.8, 最大 5), 越大放下越慢越软, 过大会抖动
            duration  kp 从 kp0 降到 0 的时长(秒)
        """
        RAW_TO_RAD = 0.001 * math.pi / 180.0  # 关节反馈单位 0.001 度 -> 弧度
        NUM_JOINTS = 6
        RATE_HZ = 100.0
        HOME_TOL_RAD = 0.05  # ~3 度, 每个关节都在此范围内视为已到 home
        dt = 1.0 / RATE_HZ

        def read_joints_rad():
            js = self.piper.GetArmJointMsgs().joint_state
            return [getattr(js, f"joint_{i}") * RAW_TO_RAD for i in range(1, NUM_JOINTS + 1)]

        # 确保已使能, 才能在放下前接管 MIT 控制 (录制时通常已使能)
        self.piper.EnablePiper()

        # 阶段 0: 缓慢回 home, 轮询直到到位或超时
        if go_home:
            start = time.time()
            while time.time() - start < 10.0:
                if max(abs(a) for a in read_joints_rad()) < HOME_TOL_RAD:
                    break
                self.piper.MotionCtrl_2(0x01, 0x01, home_speed, 0x00)
                self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                time.sleep(0.05)

        hold = read_joints_rad()
        steps = max(1, int(duration * RATE_HZ))

        # 阶段 1: kp 线性降到 0, 阻尼 kd 抵抗下落
        for s in range(steps + 1):
            kp = kp0 * (1.0 - s / steps)
            self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)  # 进入/保持 MIT 模式
            for j in range(NUM_JOINTS):
                self.piper.JointMitCtrl(j + 1, hold[j], 0.0, kp, kd, 0.0)
            time.sleep(dt)

        # 阶段 2: kp=0, 仅保留阻尼, 缓慢沉降
        for _ in range(int(settle * RATE_HZ)):
            self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
            for j in range(NUM_JOINTS):
                self.piper.JointMitCtrl(j + 1, hold[j], 0.0, 0.0, kd, 0.0)
            time.sleep(dt)

        # 阶段 3: 真正失能 + 恢复位置/速度控制模式
        while self.piper.DisablePiper():
            time.sleep(0.01)
        self.piper.MotionCtrl_1(0x02, 0, 0)
        time.sleep(0.3)

    def safe_disconnect_master(self):
        """ 
            Move to safe disconnect position
        """
        self.write_master(target_joint=self.safe_disable_position)