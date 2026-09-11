"""Pure-numpy 6-DOF quadrotor rigid-body dynamics, RK4-integrated.

Not connectome-derived (there's no biology here) -- this is the plant the
fly-brain-driven controller flies. State is position, velocity, orientation
quaternion (w, x, y, z), and body-frame angular velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ox, oy, oz = omega_body
    return 0.5 * np.array(
        [
            -x * ox - y * oy - z * oz,
            w * ox + y * oz - z * oy,
            w * oy - x * oz + z * ox,
            w * oz + x * oy - y * ox,
        ]
    )


@dataclass
class DroneParams:
    mass: float = 0.03  # kg, small quadrotor-scale
    inertia: np.ndarray = field(default_factory=lambda: np.array([2e-5, 2e-5, 4e-5]))
    gravity: float = 9.81
    linear_drag: float = 0.2
    angular_drag: float = 0.01
    max_thrust: float = 2.0  # N, > mass*gravity so hover/climb is achievable
    max_forward_thrust: float = 0.4  # N, body +x -- flies vector thrust directly (unlike a planar quadrotor)
    max_torque: np.ndarray = field(default_factory=lambda: np.array([2e-4, 2e-4, 1e-4]))
    rate_gain: float = 8.0  # body-rate P-controller gain, torque = I * rate_gain * (cmd - omega)


@dataclass
class DroneState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity, self.quat, self.omega])

    @staticmethod
    def from_vector(v: np.ndarray) -> "DroneState":
        return DroneState(
            position=v[0:3].copy(),
            velocity=v[3:6].copy(),
            quat=quat_normalize(v[6:10].copy()),
            omega=v[10:13].copy(),
        )


class DroneDynamics:
    """command = [roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, thrust_cmd,
    forward_thrust_cmd], rates roughly in [-1, 1] and both thrust commands in
    [0, 1]; scaled by DroneParams. Forward thrust is a separate body-frame
    +x actuator (flies vector their net wing-thrust force directly, unlike a
    planar quadrotor that must tilt to translate) so the connectome-derived
    readout only has to supply steering *corrections* via the rate channels,
    matching the standard tethered closed-loop-flight-simulator paradigm
    (Reiser & Dickinson 2008) where forward thrust is held roughly constant
    while vision drives steering.
    """

    def __init__(self, params: DroneParams = None, state: DroneState = None):
        self.params = params or DroneParams()
        self.state = state or DroneState()

    def reset(self, state: DroneState = None) -> None:
        self.state = state or DroneState()

    def _derivative(self, v: np.ndarray, command: np.ndarray) -> np.ndarray:
        p = self.params
        s = DroneState.from_vector(v)
        rot = quat_to_rotmat(s.quat)

        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, forward_cmd = command
        rate_cmd = np.array([roll_cmd, pitch_cmd, yaw_cmd])
        thrust = np.clip(thrust_cmd, 0.0, 1.0) * p.max_thrust
        forward_thrust = np.clip(forward_cmd, 0.0, 1.0) * p.max_forward_thrust

        torque = np.clip(
            p.inertia * p.rate_gain * (rate_cmd - s.omega),
            -p.max_torque,
            p.max_torque,
        )

        thrust_world = rot @ np.array([forward_thrust, 0.0, thrust])
        gravity_world = np.array([0.0, 0.0, -p.mass * p.gravity])
        drag_world = -p.linear_drag * s.velocity
        force = thrust_world + gravity_world + drag_world
        accel = force / p.mass

        omega_cross_I_omega = np.cross(s.omega, p.inertia * s.omega)
        angular_drag_torque = -p.angular_drag * s.omega
        omega_dot = (torque - omega_cross_I_omega + angular_drag_torque) / p.inertia

        qdot = quat_derivative(s.quat, s.omega)

        return np.concatenate([s.velocity, accel, qdot, omega_dot])

    def step(self, command: np.ndarray, dt: float) -> DroneState:
        v0 = self.state.as_vector()
        k1 = self._derivative(v0, command)
        k2 = self._derivative(v0 + 0.5 * dt * k1, command)
        k3 = self._derivative(v0 + 0.5 * dt * k2, command)
        k4 = self._derivative(v0 + dt * k3, command)
        v1 = v0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state = DroneState.from_vector(v1)
        return self.state
