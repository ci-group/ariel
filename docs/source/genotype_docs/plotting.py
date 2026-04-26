import numpy as np
import matplotlib.pyplot as plt

def plot_matrices(matrices):
    A, B, C = [np.asarray(m) for m in matrices]

    # --- Figure 1: A and C (same shape) ---
    fig1, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    for ax, M, title in zip(axes, [A, C], ["Type Probability Matrix ", "Rotation Probability Matrix"]):
        im = ax.imshow(M, vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.axis("off")
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Figure 2: B has 6 channels -> plot as 2x3 grid ---


    fig2, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    axes = axes.ravel()
    faces = {0 : "Front",
             1 : "Back",
             2 : "Right",
             3 : "Left",
             4 : "Top", 
             5 : "Bottom"}

    for ch in range(6):
        ax = axes[ch]
        im = ax.imshow(B[:, :, ch], vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Connection Probability Matrix \n Face {faces[ch]}")
        ax.axis("off")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()