import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
try:
    from scipy.signal import welch, get_window, spectrogram, detrend as sp_detrend
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def plot_results(logged_data, log_interval):
    time_steps = np.arange(logged_data["joint_positions"].shape[0]) * log_interval

    plt.figure(figsize=(15, 15)) # Increased figure height for 4 subplots

    # Plot Joint Positions
    plt.subplot(5, 1, 1)
    for i in range(logged_data["joint_positions"].shape[1]):
        plt.plot(time_steps, logged_data["joint_positions"][:, i], label=f'Joint {i+1}')
    plt.title('Joint Positions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.legend()
    plt.grid(True)

    # Plot EE Positions
    plt.subplot(5, 1, 2)
    plt.plot(time_steps, logged_data["ee_positions"][:, 0], label='EE Pos X')
    plt.plot(time_steps, logged_data["ee_positions"][:, 1], label='EE Pos Y')
    plt.plot(time_steps, logged_data["ee_positions"][:, 2], label='EE Pos Z')
    plt.title('End-Effector Positions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot EE Quaternions
    plt.subplot(5, 1, 3)
    plt.plot(time_steps, logged_data["ee_quaternions"][:, 0], label='EE Quat X')
    plt.plot(time_steps, logged_data["ee_quaternions"][:, 1], label='EE Quat Y')
    plt.plot(time_steps, logged_data["ee_quaternions"][:, 2], label='EE Quat Z')
    plt.plot(time_steps, logged_data["ee_quaternions"][:, 3], label='EE Quat W')
    plt.title('End-Effector Quaternions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Value')
    plt.legend()
    plt.grid(True)

    # Plot Actions
    plt.subplot(5, 1, 4)
    for i in range(logged_data["actions"].shape[1]):
        plt.plot(time_steps, logged_data["actions"][:, i], label=f'Action {i+1}')
    plt.title('Actions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.plot(time_steps, logged_data["distance"][:], label=f'distance')
    plt.title('distance Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('distance Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_data_dict(data_dict, log_interval):
    """
    Visualizes data from a dictionary, creating subplots for each key.

    Args:
        data_dict (dict): A dictionary where keys are data names (e.g., "joint_positions")
                          and values are NumPy arrays of the corresponding data.
        log_interval (float): The time interval between logged data points.
    """
    if not data_dict:
        print("No data to plot.")
        return

    num_plots = len(data_dict)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3 * num_plots))
    if num_plots == 1:
        axes = [axes] # Ensure axes is iterable even for a single subplot

    time_steps = np.arange(list(data_dict.values())[0].shape[0]) * log_interval

    for i, (key, data) in enumerate(data_dict.items()):
        ax = axes[i]
        if data.ndim == 1:
            ax.plot(time_steps, data, label=key)
            ax.set_ylabel('Value')
        elif data.ndim == 2:
            for j in range(data.shape[1]):
                ax.plot(time_steps, data[:, j], label=f'{key} {j+1}')
            ax.set_ylabel('Value')
        else:
            print(f"Skipping plotting for {key}: unsupported dimension {data.ndim}")
            continue

        ax.set_title(f'{key} Over Time')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_3d_trajectory(ee_positions, target_positions=None, 
                       title='End-Effector 3D Trajectory',
                       save_name='ee_3d_trajectory.png',
                       ):
    """
    Visualizes the 3D trajectory of the end-effector.

    Args:
        ee_positions (np.ndarray): A NumPy array of shape (N, 3) representing
                                   the (x, y, z) coordinates of the end-effector over time.
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], marker='o', markersize=2, linestyle='-')
    # also plot target positions if provided
    if target_positions is not None:
        ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], marker='x', s=10, color='r', label='Target Position')
        ax.legend()
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.grid(True)
    plt.show()

    plt.savefig(save_name)


def _safe_detrend(x):
    if _HAS_SCIPY:
        return sp_detrend(x, type="linear")
    # 无 scipy 时，至少去均值
    return x - np.nanmean(x)

def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())

def _compute_fft(signal_1d, fs, apply_window=True, zero_pad=True):
    """
    输入: signal_1d shape=(N,), 采样频率 fs(Hz)
    返回: freqs(Hz), amp(幅度谱, 单边), complex_fft(单边复数谱)
    """
    x = np.asarray(signal_1d).astype(np.float64)
    x = _safe_detrend(x)

    N = len(x)
    if N < 4:
        return np.array([0.0]), np.array([0.0]), np.array([0.0+0.0j])

    if apply_window:
        if _HAS_SCIPY:
            w = get_window("hann", N, fftbins=True)
        else:
            # 简单 Hann 窗
            n = np.arange(N)
            w = 0.5 * (1 - np.cos(2*np.pi*n/(N-1)))
        x = x * w

    if zero_pad:
        Nfft = _next_pow2(N)
    else:
        Nfft = N

    fft_full = np.fft.rfft(x, n=Nfft)  # 单边谱
    freqs = np.fft.rfftfreq(Nfft, d=1.0/fs)

    # 把幅度谱缩放到“近似幅值”，考虑窗函数能量
    # 对于 rfft 的单边谱：中间频率乘以 2，DC 和 Nyquist 不乘
    # 同时按窗的均方根能量做归一（防止窗导致能量畸变）
    if apply_window:
        # 窗能量校正
        U = (w**2).sum() / N
    else:
        U = 1.0

    # 幅度（近似与原信号幅值一致的量纲）
    amp = np.abs(fft_full) / (N * np.sqrt(U))
    if len(amp) > 2:
        amp[1:-1] *= 2.0

    return freqs, amp, fft_full

def _compute_psd_welch(signal_1d, fs, nperseg=None):
    """
    Welch 法 PSD；返回 freqs(Hz), psd(功率/Hz)
    """
    if not _HAS_SCIPY:
        return None, None
    x = np.asarray(signal_1d).astype(np.float64)
    x = _safe_detrend(x)
    if nperseg is None:
        # 2 秒窗或最近的 256，取小的那个，更稳健
        nperseg = min(max(64, int(2*fs)), 1024)
    f, pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)), noverlap=int(0.5*nperseg), window="hann")
    return f, pxx

def plot_imu_fft(
    imu_dict,
    fs,
    axis_labels=("ax","ay","az","gx","gy","gz"),
    fmax=None,
    do_psd=True,
    do_spectrogram=False,
    spec_axes=("ax","gx"),   # 想看谱图的通道
):
    """
    imu_dict: dict 形如 {"joint1": N x 6, "joint2": N x 6, ...}
    fs: 采样频率(Hz) = 1.0 / log_interval
    axis_labels: 每列的名称
    fmax: 频率上限（Hz），例如 100；None 则显示到 Nyquist
    do_psd: 是否绘制 Welch PSD
    do_spectrogram: 是否绘制谱图（对少量通道）
    spec_axes: 需要画谱图的列名（会在每个 joint 里找对应列）
    """
    nyq = fs / 2.0
    if fmax is None or fmax > nyq:
        fmax = nyq

    for joint_name, arr in imu_dict.items():
        arr = np.asarray(arr)
        if arr.ndim != 2:
            print(f"[WARN] {joint_name} 数据维度不是 (N, C)，跳过。shape={arr.shape}")
            continue
        N, C = arr.shape

        # 轴名称长度自适应
        if len(axis_labels) != C:
            labels = [f"ch{i}" for i in range(C)]
        else:
            labels = list(axis_labels)

        # ---------- 幅度谱 ----------
        fig, axes = plt.subplots(C, 1, figsize=(12, 2.6*C), sharex=True)
        if C == 1:
            axes = [axes]

        for ci in range(C):
            x = arr[:, ci]
            f, amp, _ = _compute_fft(x, fs, apply_window=True, zero_pad=True)
            # 频率限制
            mask = (f >= 0.0) & (f <= fmax)
            axes[ci].plot(f[mask], amp[mask])
            axes[ci].set_ylabel(f"{labels[ci]}\nAmplitude")
            axes[ci].grid(True)

        axes[-1].set_xlabel("Frequency (Hz)")
        fig.suptitle(f"[{joint_name}] IMU Amplitude Spectrum (FFT)")
        plt.tight_layout()
        plt.show()

        # ---------- PSD ----------
        if do_psd and _HAS_SCIPY:
            fig, axes = plt.subplots(C, 1, figsize=(12, 2.6*C), sharex=True)
            if C == 1:
                axes = [axes]
            for ci in range(C):
                x = arr[:, ci]
                f, pxx = _compute_psd_welch(x, fs)
                if f is None:
                    continue
                mask = (f >= 0.0) & (f <= fmax)
                axes[ci].semilogy(f[mask], pxx[mask] + 1e-16)  # 避免 log(0)
                axes[ci].set_ylabel(f"{labels[ci]}\nPSD")
                axes[ci].grid(True, which="both", ls="--", alpha=0.5)
            axes[-1].set_xlabel("Frequency (Hz)")
            fig.suptitle(f"[{joint_name}] IMU Power Spectral Density (Welch)")
            plt.tight_layout()
            plt.show()

        # ---------- 谱图（可选） ----------
        if do_spectrogram and _HAS_SCIPY:
            # 为每个 joint 只画少数几个通道，避免图太多
            for ch_name in spec_axes:
                if ch_name in labels:
                    ci = labels.index(ch_name)
                else:
                    continue
                x = _safe_detrend(arr[:, ci])
                nperseg = min(max(64, int(1.0*fs)), len(x))   # 约 1 秒窗
                noverlap = int(0.5*nperseg)
                f, t, Sxx = spectrogram(
                    x, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", mode="psd"
                )
                mask = f <= fmax
                plt.figure(figsize=(12, 4))
                plt.pcolormesh(t, f[mask], 10*np.log10(Sxx[mask, :] + 1e-16), shading='gouraud')
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                plt.title(f"[{joint_name}] Spectrogram - {ch_name}")
                cbar = plt.colorbar()
                cbar.set_label("PSD (dB/Hz)")
                plt.tight_layout()
                plt.show()

def visualize_imu_frequency(logged_imu_data_dict, log_interval, fmax=None, do_psd=True, do_spectrogram=False):
    fs = 1.0 / float(log_interval)
    print(f"[IMU-FFT] sampling_freq = {fs:.3f} Hz, nyquist = {fs/2:.3f} Hz")
    plot_imu_fft(
        imu_dict=logged_imu_data_dict,
        fs=fs,
        axis_labels=("ax","ay","az","gx","gy","gz"),
        fmax=fmax,
        do_psd=do_psd,
        do_spectrogram=do_spectrogram,
        spec_axes=("ax","gx"),
    )

