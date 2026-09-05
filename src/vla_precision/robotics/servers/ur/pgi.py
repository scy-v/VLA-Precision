"""PGI serial gripper drivers used by the UR robot service.

The command semantics intentionally match the data-collection implementation:
device units are clamped to a task-configured range, the left reset opens once,
and the right reset performs open-close-open.
"""

from __future__ import annotations

import time
import types


def validate_position_range(min_position: int, max_position: int) -> tuple[int, int]:
    minimum = int(min_position)
    maximum = int(max_position)
    if not 0 <= minimum < maximum <= 1000:
        raise ValueError(
            "gripper position range must satisfy "
            f"0 <= min_position < max_position <= 1000, got [{minimum}, {maximum}]"
        )
    return minimum, maximum


def _set_device_position(device, val: int, blocking: bool = True) -> None:
    val = int(val)
    if not 0 <= val <= 1000:
        raise RuntimeError("PGI position is outside device range 0..1000")
    device.write_uart(modbus_high_addr=0x01, modbus_low_addr=0x03, val=val)
    if blocking:
        while device.read_state() not in (1, 2):
            time.sleep(0.01)


class PGIGripperServer:
    """One PGI/PGE device with the exact queued server command interface."""

    kind = "pgi"

    def __init__(
        self,
        *,
        init_gripper: bool,
        gripper_port: str,
        gripper_force: int,
        gripper_speed: int,
        min_position: int,
        max_position: int,
        side: str = "left",
    ) -> None:
        self.side = side
        self.gripper_port = gripper_port
        self.min_position, self.max_position = validate_position_range(
            min_position, max_position
        )
        self.parallel_force = int(gripper_force)
        self.parallel_velocity = int(gripper_speed)
        self.abs_angle = 99_999_999
        self.rel_angle = 0
        self.position = 500
        self.stop_flag = 0

        self.gripper = self._create_gripper(init_gripper)
        self.gripper.set_pos = types.MethodType(_set_device_position, self.gripper)
        if init_gripper:
            self.gripper.init_feedback()
        self.gripper.set_force(self.parallel_force)
        self.gripper.set_vel(self.parallel_velocity)

    def _create_gripper(self, init_gripper: bool = True):
        """Construct the established PGI driver, retaining the test/extension seam."""
        from pyDHgripper import PGE

        if init_gripper:
            return PGE(port=self.gripper_port)

        import crcmod
        import serial

        gripper = PGE.__new__(PGE)
        gripper.ser = serial.Serial(port=self.gripper_port, baudrate=115200)
        gripper.crc16 = crcmod.mkCrcFun(
            0x18005,
            rev=True,
            initCrc=0xFFFF,
            xorOut=0x0000,
        )
        return gripper

    def clamp_position(self, position) -> int:
        return max(self.min_position, min(self.max_position, int(position)))

    def set_position(self, position, blocking: bool = False) -> None:
        self.position = self.clamp_position(position)
        self.gripper.set_pos(val=self.position, blocking=blocking)

    def open(self, blocking: bool = False) -> None:
        self.set_position(self.max_position, blocking=blocking)

    def open_forced(self, blocking: bool = False) -> None:
        self.open(blocking=blocking)

    def close(self, blocking: bool = False) -> None:
        self.set_position(self.min_position, blocking=blocking)

    def get_state(self):
        return self.gripper.read_state()

    def get_gripose(self) -> None:
        self.position = self.gripper.read_pos() / 1000

    def _update_gripper(self, _message=None) -> None:
        """Retain the original open-close-refresh callback behavior."""
        self.open()
        self.close()
        self.position = self.gripper.read_pos() / 1000

    def reset_gripper(self) -> None:
        self.open()
        if self.side == "right":
            self.close()
            self.open()


class LeftPGIGripperServer(PGIGripperServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(side="left", **kwargs)

    def move(self, command: str, blocking: bool = False) -> None:
        if command == "open":
            self.open(blocking=blocking)
        elif command == "close":
            self.close(blocking=blocking)
        else:
            raise ValueError("Invalid move type")


class RightPGIGripperServer(PGIGripperServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(side="right", **kwargs)

    def move(self) -> None:
        """Preserve the established right-side no-op hook."""
