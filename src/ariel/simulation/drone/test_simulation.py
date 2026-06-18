

from drone_simulator import DroneSimulator

drone = DroneSimulator(propellers=None, dt=0.01, gravity=9.81)

number_of_steps = 100

for i in range(number_of_steps):
    drone.step([0.1, 0.1, 0.1, 0.1])  # Step the simulation by 0.01 seconds
    state = drone.get_state()
    print(f"Step {i+1}, P: {state['position']}, V: {state['velocity']}, O: {state['attitude']}, W: {state['angular_velocity']}")
