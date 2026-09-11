import numpy as np

from fly_drone_pilot.drone.dynamics import DroneDynamics, DroneParams, DroneState


def test_hover_thrust_keeps_altitude_stable():
    params = DroneParams()
    hover_thrust = params.mass * params.gravity / params.max_thrust
    drone = DroneDynamics(params)
    drone.reset(DroneState(position=np.array([0.0, 0.0, 1.5])))

    command = np.array([0.0, 0.0, 0.0, hover_thrust, 0.0])
    for _ in range(500):
        drone.step(command, dt=0.005)

    assert abs(drone.state.position[2] - 1.5) < 0.05
    assert np.linalg.norm(drone.state.velocity) < 0.05


def test_zero_thrust_falls_under_gravity():
    drone = DroneDynamics(DroneParams())
    drone.reset(DroneState(position=np.array([0.0, 0.0, 5.0])))
    command = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        drone.step(command, dt=0.005)
    assert drone.state.position[2] < 5.0
    assert drone.state.velocity[2] < 0


def test_yaw_rate_command_rotates_without_nan():
    drone = DroneDynamics(DroneParams())
    drone.reset(DroneState(position=np.array([0.0, 0.0, 1.5])))
    hover_thrust = drone.params.mass * drone.params.gravity / drone.params.max_thrust
    command = np.array([0.0, 0.0, 1.0, hover_thrust, 0.0])
    for _ in range(200):
        state = drone.step(command, dt=0.005)
    assert np.isfinite(state.quat).all()
    assert abs(np.linalg.norm(state.quat) - 1.0) < 1e-6
    assert state.omega[2] > 0
