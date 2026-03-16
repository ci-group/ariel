import threading
import time
import numpy as np

class DataLogger:
    def __init__(self, env, log_interval=0.01):
        self.env = env
        # self.robot = self.env.robot
        # self.sensor = self.env.sensor
        self.log_interval = log_interval
        self.joint_pos_data = []
        self.ee_pos_data = []
        self.ee_quat_data = []
        self.action_data = []
        self.distance_data = []
        self._running = False
        self._thread = None

    def _log_data(self):
        while self._running:
            joint_pos = self.env.robot._current_joint_angles_deg.copy()
            ee_pos = self.env.sensor.data_handler.latest_relative_pos  # Assuming latest_relative_pos is [x, y, z, qx, qy, qz, qw]
            ee_quat = self.env.sensor.data_handler.latest_relative_quat
            action = self.env.robot._last_action / 100.0
            distance = self.env._current_distance

            # self.joint_pos_data.append(joint_pos)
            self.ee_pos_data.append(ee_pos)
            self.ee_quat_data.append(ee_quat)
            self.action_data.append(action)
            # print(f"current action: {action}, distance: {distance}")
            self.joint_pos_data.append(joint_pos)
            self.distance_data.append(distance)
            time.sleep(self.log_interval)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._log_data)
            self._thread.daemon = True  # Allow the main program to exit even if thread is running
            self._thread.start()
            print("DataLogger started.")

    def stop(self):
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join()
            print("DataLogger stopped.")

    def get_logged_data(self):
        return {
            "joint_positions": np.array(self.joint_pos_data),
            "ee_positions": np.array(self.ee_pos_data),
            "ee_quaternions": np.array(self.ee_quat_data),
            "actions": np.array(self.action_data),  # Uncomment if you want to log actions
            "distance": np.array(self.distance_data),
        }

    def clear_data(self):
        self.joint_pos_data = []
        self.ee_pos_data = []
        self.ee_quat_data = []
        self.action_data = []
        self.distance_data = []

if __name__ == '__main__':
    # This is a placeholder for testing purposes.
    # In a real scenario, 'robot' would be an instance of your robot class.
    class MockRobot:
        def __init__(self):
            self._joint_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            self._ee_pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

        def get_joint_positions(self):
            self._joint_pos += np.random.rand(6) * 0.01
            return self._joint_pos

        def get_ee_pose(self):
            self._ee_pose[:3] += np.random.rand(3) * 0.005
            return self._ee_pose

    mock_robot = MockRobot()
    logger = DataLogger(mock_robot, log_interval=0.1)

    logger.start()
    time.sleep(2) # Log data for 2 seconds
    logger.stop()

    data = logger.get_logged_data()
    print("Collected Joint Positions Shape:", data["joint_positions"].shape)
    print("Collected EE Positions Shape:", data["ee_positions"].shape)
    print("Collected EE Quaternions Shape:", data["ee_quaternions"].shape)

    logger.clear_data()
    print("Data cleared.")