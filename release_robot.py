#!/usr/bin/env python3
"""Disable all SO-101 motor torque. Support the arm before running this command.

Stop the policy runner first. This releases the joints, including the gripper;
it does not move the robot to a rest pose or switch off its power supply.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/cu.usbmodem5B421342761')
    args = parser.parse_args()
    if not Path(args.port).exists():
        raise SystemExit(f'Serial port not found: {args.port}. Connect the robot USB and power first.')

    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    bus = SerialMotorsBus(
        port=args.port,
        motors={
            name: Motor(index + 1, 'sts3215', MotorNormMode.DEGREES if index < 5 else MotorNormMode.RANGE_0_100)
            for index, name in enumerate(names)
        },
    )
    try:
        bus.connect()
        bus.disable_torque(num_retry=3)
        enabled = bus.sync_read('Torque_Enable', normalize=False, num_retry=3)
        if any(enabled.values()):
            raise RuntimeError(f'Torque release could not be confirmed for every motor: {enabled}')
        print('Torque disabled on all six motors. The arm and gripper can now be moved manually.')
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == '__main__':
    main()
