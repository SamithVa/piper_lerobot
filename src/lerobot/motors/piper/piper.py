import time
from dataclasses import dataclass
from typing import Dict

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

# rad <-> raw 0.001-deg firmware scale. 57324.840764 == 1000 * 180 / pi.
# read() MUST keep emitting these raw units: piper_follower records them directly
# as observation.state and piper_leader divides joints by this factor (and the
# gripper by 1e6). pyAgxArm reports rad / m, so we convert back at the boundary.
JOINT_RAD_TO_RAW = 57324.840764
GRIPPER_M_TO_RAW = 1_000_000.0  # meters -> raw (~micrometers); leader divides by 1e6


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]


class PiperMotorsBus:
    """对 AgileX pyAgxArm SDK 的二次封装 (由旧 piper_sdk 封装迁移而来)。

    对外接口保持不变: read() 返回原始固件单位 (关节 0.001°, 夹爪 ~µm),
    write() 接收弧度 (关节) 与米 (夹爪 0~0.08)。内部驱动改为 pyAgxArm 的
    arm driver (self.robot) 与 gripper effector (self.gripper)。
    """

    def __init__(self, config: PiperMotorsBusConfig):
        cfg = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=PiperFW.V188,
            interface="socketcan",
            channel=config.can_name,
        )
        self.robot = AgxArmFactory.create_arm(cfg)
        # Effector must be created BEFORE connect() and can only be created once.
        self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.robot.connect()  # start the shared read thread (arm + gripper feedback)
        self.motors = config.motors
        self._speed_pct = None
        # 录制数据集时改成0
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [6 joints + 1 gripper]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pose_factor = 1000  # 单位 0.001mm
        self.joint_factor = JOINT_RAD_TO_RAW  # rad -> 0.001°

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def _ensure_speed(self, pct: int) -> None:
        """Set joint speed percent only when it changes (avoid flooding config
        frames at teleop rate). move_j runs at whatever percent was last set."""
        if getattr(self, "_speed_pct", None) != pct:
            self.robot.set_speed_percent(pct)
            self._speed_pct = pct

    def connect(self, enable: bool, timeout: float = 10) -> bool:
        '''
            使能机械臂并检测使能状态; 若使能超时则返回 False (由调用方重试)。
            timeout: 使能等待上限(秒)。从冷/失能状态使能较慢, 给足时间避免误判未使能。
        '''
        enable_flag = False
        loop_flag = False
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            # get_joints_enable_status_list() -> [bool]*6, mirrors reading the 6
            # per-motor foc_status.driver_enable_status flags in the old SDK.
            enable_list = self.robot.get_joints_enable_status_list()
            if enable:
                enable_flag = all(enable_list)
                # robot.enable(255) returns True once all joints already report
                # enabled, so it only flips True after the enable actually stuck.
                # Bound this wait by the outer timeout: if the arm never enables
                # (enable frame dropped in the connect burst, arm unpowered/
                # e-stopped, driver fault, or status frames not arriving) an
                # UNBOUNDED loop here hangs forever printing "piper initing" and
                # blocks both the outer timeout and the caller's retry. On
                # timeout, fall through -> outer loop returns False -> caller
                # retries / raises loudly.
                while not self.robot.enable(255):
                    if time.time() - start_time > timeout:
                        print("enable timed out (motors never reported enabled)")
                        break
                    print('piper initing')
                    time.sleep(0.1)
                self.gripper.move_gripper_m(0.0, 1.0)  # close gripper (was GripperCtrl enable)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.robot.disable(255)
                self.gripper.disable_gripper()
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if (enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        if enable and resp:
            # Readiness gate: do NOT hand control to teleop/record until the arm
            # is actually READY and STAYS ready. Declaring success the instant
            # enable first reads True (then a blind sleep) causes the intermittent
            # "follower sometimes not activated -> leader can't control it". Three
            # cold-start races feed that flake, all covered by requiring the ready
            # condition to hold *continuously* for a short window:
            #   1. enable status can read True for a single sample before the
            #      driver is really holding torque -> require it STABLE, not one-shot.
            #   2. an enable frame issued during the 4-arm connect burst can be
            #      dropped (ENOBUFS), so the enable doesn't stick -> a dropout
            #      resets the window and the loop re-confirms.
            #   3. on the first process after CAN (re)activation joint feedback
            #      isn't flowing yet (get_joint_angles().hz == 0), so a leader
            #      would stream stale zeros / a follower would record stale state
            #      -> require hz > 0.
            if not self._wait_until_ready(timeout=5.0, stable_needed=0.6):
                print("arm not stably ready (enable+feedback) within deadline; failing connect to trigger retry")
                return False
        print(f"Returning response: {resp}")
        return resp

    def _all_motors_enabled(self) -> bool:
        """True iff all 6 joint drivers currently report enabled."""
        return bool(self.robot.get_joint_enable_status(255))

    def is_ready(self) -> bool:
        """Arm is holding torque (all drivers enabled) AND joint feedback is live.

        Cheap, side-effect-free snapshot -- safe to poll as a readiness barrier
        before starting teleop.
        """
        try:
            ja = self.robot.get_joint_angles()
            return self._all_motors_enabled() and ja is not None and ja.hz > 0
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

        Why polled instead of a single write(): right after enable a dropped
        motion frame (4-arm connect burst) can mean the follower never moves to
        home AND ignores the teleop move_j that follow. Repeating the command
        proves the follower is accepting motion before teleop starts.
        """
        def joints_rad() -> list[float]:
            ja = self.robot.get_joint_angles()
            return [round(a, 3) for a in ja.msg] if ja is not None else [0.0] * 6

        def status_str() -> str:
            st = self.robot.get_arm_status()
            if st is None:
                return "ctrl_mode=?? arm_status=??"
            return f"ctrl_mode=0x{st.msg.ctrl_mode:02X} arm_status=0x{st.msg.arm_status:02X}"

        start_joints = joints_rad()
        start = time.time()
        cmds_sent = 0
        self._ensure_speed(speed)
        while time.time() - start < timeout:
            self.robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.gripper.move_gripper_m(0.0, 1.0)
            cmds_sent += 1
            time.sleep(0.05)
            ja = self.robot.get_joint_angles()
            cur = ja.msg if ja is not None else [1.0] * 6
            if max(abs(a) for a in cur) < tol_rad:
                # ctrl_mode must be 0x01 (CAN command control) for teleop move_j
                # to be obeyed; surface it so "home ok but ctrl_mode=0x00" (arm at
                # home yet ignoring commands) is visible.
                print(f"move_to_home ok after {cmds_sent} cmd(s): {status_str()} start={start_joints}")
                return True
        print(
            f"move_to_home timed out after {timeout}s: {status_str()} "
            f"joints(rad) start={start_joints} end={joints_rad()}"
        )
        return False

    def apply_calibration(self) -> bool:
        """移动到初始位置 (follower). Polled + verified: returns True iff home reached."""
        return self.move_to_home()

    def apply_calibration_master(self):
        """master移动到初始位置"""
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint: list):
        """
            Joint control
            - target_joint: 前 6 个为关节角度 (弧度), 第 7 个为夹爪行程 (米, 0~0.08)
        """
        self._ensure_speed(50)
        self.robot.move_j([float(target_joint[i]) for i in range(6)])
        gripper_m = abs(float(target_joint[6]))
        self.gripper.move_gripper_m(gripper_m, 1.0)

    def read(self) -> Dict:
        """
            返回原始固件单位:
            - 关节: 0.001度 (rad * 57324.840764)
            - 夹爪: 原始单位 (~µm, m * 1e6)
        """
        ja = self.robot.get_joint_angles()
        joints_rad = ja.msg if ja is not None else [0.0] * 6

        gs = self.gripper.get_gripper_status()
        gripper_m = gs.msg.value if gs is not None else 0.0

        return {
            "joint_1": joints_rad[0] * self.joint_factor,
            "joint_2": joints_rad[1] * self.joint_factor,
            "joint_3": joints_rad[2] * self.joint_factor,
            "joint_4": joints_rad[3] * self.joint_factor,
            "joint_5": joints_rad[4] * self.joint_factor,
            "joint_6": joints_rad[5] * self.joint_factor,
            "gripper": gripper_m * GRIPPER_M_TO_RAW,
        }

    def safe_disconnect(self):
        """Move to safe disconnect position"""
        self.write(target_joint=self.safe_disable_position)

    def gentle_disable(self, kp0: float = 10.0, kd: float = 0.8, duration: float = 2.0,
                       go_home: bool = True, home_speed: int = 15, settle: float = 0.6):
        """
            软失能: 避免机械臂直接断电自由落体硬砸下去。
            流程:
              1. (可选) 用位置控制缓慢回到 home 姿态;
              2. 用 MIT 力控保持当前关节角, 把位置增益 kp 在 duration 秒内线性降到 0,
                 保留阻尼 kd, 机械臂被阻尼缓慢放下;
              3. kp=0 只留阻尼沉降一小段, 最后真正失能。

            kp0  初始保持增益 (SDK 参考 10)
            kd   阻尼增益 (SDK 参考 0.8, 最大 5), 越大放下越慢越软, 过大会抖动
            duration  kp 从 kp0 降到 0 的时长(秒)
        """
        NUM_JOINTS = 6
        RATE_HZ = 100.0
        HOME_TOL_RAD = 0.05  # ~3 度, 每个关节都在此范围内视为已到 home
        dt = 1.0 / RATE_HZ

        def read_joints_rad():
            ja = self.robot.get_joint_angles()
            return list(ja.msg) if ja is not None else [0.0] * NUM_JOINTS

        # 确保已使能, 才能在放下前接管 MIT 控制 (录制时通常已使能)
        self.robot.enable(255)

        # 阶段 0: 缓慢回 home, 轮询直到到位或超时
        if go_home:
            self._ensure_speed(home_speed)
            start = time.time()
            while time.time() - start < 10.0:
                if max(abs(a) for a in read_joints_rad()) < HOME_TOL_RAD:
                    break
                self.robot.move_j([0.0] * NUM_JOINTS)
                self.gripper.move_gripper_m(0.0, 1.0)
                time.sleep(0.05)

        hold = read_joints_rad()
        steps = max(1, int(duration * RATE_HZ))

        # 阶段 1: kp 线性降到 0, 阻尼 kd 抵抗下落 (move_mit 内部会切到 MIT 模式)
        for s in range(steps + 1):
            kp = kp0 * (1.0 - s / steps)
            for j in range(NUM_JOINTS):
                self.robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=kp, kd=kd, t_ff=0.0)
            time.sleep(dt)

        # 阶段 2: kp=0, 仅保留阻尼, 缓慢沉降
        for _ in range(int(settle * RATE_HZ)):
            for j in range(NUM_JOINTS):
                self.robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=0.0, kd=kd, t_ff=0.0)
            time.sleep(dt)

        # 阶段 3: 真正失能 (下一次 move_j 会自动切回位置/速度控制模式)
        while not self.robot.disable(255):
            time.sleep(0.01)
        time.sleep(0.3)
