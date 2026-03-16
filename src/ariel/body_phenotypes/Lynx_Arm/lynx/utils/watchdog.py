import multiprocessing
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class WatchDog: # Renamed class
    def __init__(self):
        self.data_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self._run_watchdog, args=(self.data_queue, self.stop_event))
        self.process.start()

    def _run_watchdog(self, data_queue, stop_event):
        matplotlib.use('TkAgg') # Explicitly set backend
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('EE and Target Position')
        
        # Set initial limits for the plot
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 1.0])

        # Initialize scatter plots for EE and Target
        ee_plot, = ax.plot([], [], [], 'o', color='blue', markersize=10, label='EE Position')
        last_ee_plot, = ax.plot([], [], [], 'o', color='green', markersize=10, label='Last EE Position')
        target_plot, = ax.plot([], [], [], 'o', color='red', markersize=10, label='Target Position')
        ax.legend()

        plt.ion() # Turn on interactive mode
        plt.show(block=False) # Ensure non-blocking show

        print("WatchDog Matplotlib viewer started. Waiting for data...")

        while not stop_event.is_set():
            if not data_queue.empty():
                data_received = data_queue.get()
                ee_pos = data_received["ee_pos"]
                last_ee_pos = data_received["last_ee_pos"]
                target_pos = data_received["target_pos"]
                # print(f"WatchDog: Received EE Pos: {ee_pos}, Target Pos: {target_pos}") # Debug print

                # Update scatter plot data
                ee_plot.set_data([ee_pos[0]], [ee_pos[1]])
                ee_plot.set_3d_properties([ee_pos[2]])

                last_ee_plot.set_data([last_ee_pos[0]], [last_ee_pos[1]])
                last_ee_plot.set_3d_properties([last_ee_pos[2]])

                target_plot.set_data([target_pos[0]], [target_pos[1]])
                target_plot.set_3d_properties([target_pos[2]])
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                # print(f"WatchDog: Updated EE Geom Pos: {ee_pos}, Target Geom Pos: {target_pos}") # Debug print

            plt.pause(0.001) # Allow GUI events to be processed
            time.sleep(0.01) # Small delay for data polling

        plt.close(fig)
        print("WatchDog process stopped.")

    def send_data(self, ee_pos, last_ee_pos, target_pos):
        if not self.stop_event.is_set():
            self.data_queue.put({"ee_pos": ee_pos, "last_ee_pos": last_ee_pos, "target_pos": target_pos})

    def close(self):
        self.stop_event.set()
        self.process.join()
        print("WatchDog closed.")

def render_ee_and_target_pos(watchdog_instance, ee_pos, last_ee_pos, target_pos):
    """
    Sends end-effector and target position data to the watchdog for rendering.
    """
    watchdog_instance.send_data(ee_pos, last_ee_pos, target_pos)

if __name__ == "__main__":
    # Example usage:
    watchdog = WatchDog() # Changed class name here too
    try:
        for i in range(100):
            # Simulate some random movement for demonstration
            ee_pos_example = np.array([0.1 * np.sin(i * 0.1), 0.1 * np.cos(i * 0.1), 0.5 + 0.05 * np.sin(i * 0.2)])
            last_ee_pos_example = np.array([0.1 * np.sin((i - 1) * 0.1), 0.1 * np.cos((i - 1) * 0.1), 0.5 + 0.05 * np.sin((i - 1) * 0.2)])
            target_pos_example = np.array([0.2 * np.cos(i * 0.05), 0.2 * np.sin(i * 0.05), 0.6 - 0.05 * np.cos(i * 0.15)])
            render_ee_and_target_pos(watchdog, ee_pos_example, last_ee_pos_example, target_pos_example)
            time.sleep(0.1) # Increased sleep time for better visualization in example
    finally:
        watchdog.close()