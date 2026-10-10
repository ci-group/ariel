import numpy as np

def calculate_stats(gate_times_sec, n_gates):
    """Calculate lap times, segment times, and their statistics."""
    segment_durations = np.diff(gate_times_sec)
    segment_indices = np.arange(len(segment_durations)) % n_gates

    avg_times = []
    std_times = []

    for i in range(n_gates):
        segment_times = segment_durations[segment_indices == i]
        avg_times.append(np.mean(segment_times))
        std_times.append(np.std(segment_times))

    n_laps = len(gate_times_sec) // n_gates
    lap_times = []
    for lap in range(n_laps):
        start_idx = lap * n_gates
        end_idx = start_idx + n_gates - 1
        lap_time = gate_times_sec[end_idx] - gate_times_sec[start_idx]
        lap_times.append(lap_time)

    lap_times = np.array(lap_times)

    return {
        "avg_segment_times": avg_times,
        "std_segment_times": std_times,
        "lap_times": lap_times,
        "avg_lap_time": np.mean(lap_times),
        "std_lap_time": np.std(lap_times),
        "total_laps": n_laps,
    }