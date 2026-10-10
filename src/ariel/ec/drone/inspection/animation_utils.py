"""
Animation utilities for creating videos and GIFs from evolution visualizations.

This module handles the creation of animations from sequences of matplotlib figures,
including temporary file management and format conversion.

# TODO: [LOW] Add progress bars for long animation rendering processes
# TODO: [LOW] Add support for additional output formats (WebM, APNG)
# TODO: [LOW] Implement memory-efficient rendering for very large datasets
"""

import os
import tempfile
from typing import List, Callable, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

from ariel.ec.drone.inspection.create_subplot import create_subplot


def animate_grids(
    grids: List[List[List[Callable]]],
    interval: int = 100,
    dpi: int = 300,
    gif_format: bool = False,
    frame_skip: int = 1,
    output_path: Optional[str] = None,
    cleanup: bool = True
) -> animation.ArtistAnimation:
    """
    Create an animation from a sequence of grid layouts.
    
    Args:
        grids: List of grid layouts (each grid is a 2D array of plotting functions)
        interval: Time between frames in milliseconds
        dpi: Resolution for saved frames
        gif_format: If True, save as GIF; if False, save as MP4
        frame_skip: Skip every N frames for faster animation
        output_path: Custom output path (if None, uses default naming)
        cleanup: If True, remove temporary files after animation creation
        
    Returns:
        matplotlib ArtistAnimation object
    """
    # Apply frame skipping
    grids = grids[::frame_skip]
    
    if len(grids) == 0:
        raise ValueError("No grids provided after frame skipping")
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate and save frame images
        img_paths = _generate_frame_images(grids, temp_dir, dpi)
        
        # Create animation from images
        ani = _create_animation_from_images(img_paths, interval)
        
        # Save animation
        if output_path is None:
            output_path = "./animated_evolution.gif" if gif_format else "./animated_evolution.mp4"
        
        _save_animation(ani, output_path, gif_format)
        
        return ani
        
    finally:
        if cleanup:
            _cleanup_temp_files(temp_dir, img_paths if 'img_paths' in locals() else [])


def _generate_frame_images(
    grids: List[List[List[Callable]]],
    temp_dir: str,
    dpi: int
) -> List[str]:
    """Generate image files for each frame."""
    img_paths = []
    
    for i, grid in enumerate(grids):
        print(f"Generating frame {i+1}/{len(grids)}")
        
        # Create subplot from grid
        fig, axs, img_plots, texts = create_subplot(grid)
        
        # Save frame image
        img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(img_path, dpi=dpi, bbox_inches='tight')
        img_paths.append(img_path)
        
        # Close figure to free memory
        plt.close(fig)
    
    return img_paths


def _create_animation_from_images(
    img_paths: List[str],
    interval: int
) -> animation.ArtistAnimation:
    """Create matplotlib animation from image files."""
    print("Loading images for animation...")
    
    # Load all images
    images = [Image.open(img_path) for img_path in img_paths]
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.axis('off')
    
    # Create animation frames
    animation_frames = []
    for img in images:
        im = ax.imshow(img, animated=True)
        animation_frames.append([im])
    
    # Create animation
    ani = animation.ArtistAnimation(
        fig, 
        animation_frames, 
        interval=interval, 
        blit=True,
        repeat=True
    )
    
    return ani


def _save_animation(
    ani: animation.ArtistAnimation,
    output_path: str,
    gif_format: bool
) -> None:
    """Save animation to file."""
    print(f"Saving animation to {output_path}")
    
    if gif_format:
        # Save as GIF
        ani.save(output_path, writer='pillow', fps=10)
    else:
        # Save as MP4
        try:
            ani.save(output_path, writer='ffmpeg', fps=10, bitrate=1800)
        except Exception as e:
            print(f"FFmpeg not available, trying alternative: {e}")
            # Fallback to other writers
            try:
                ani.save(output_path, writer='imagemagick', fps=10)
            except Exception as e2:
                print(f"ImageMagick also not available: {e2}")
                print("Saving as GIF instead...")
                gif_path = output_path.replace('.mp4', '.gif')
                ani.save(gif_path, writer='pillow', fps=10)


def _cleanup_temp_files(temp_dir: str, img_paths: List[str]) -> None:
    """Clean up temporary files and directory."""
    print("Cleaning up temporary files...")
    
    # Remove image files
    for img_path in img_paths:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except OSError:
            pass
    
    # Remove temporary directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


def create_animation_from_functions(
    plot_functions: List[Callable],
    interval: int = 200,
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
    gif_format: bool = False
) -> animation.FuncAnimation:
    """
    Create an animation where each frame is generated by a plotting function.
    
    Args:
        plot_functions: List of functions that create plots (each should take ax as argument)
        interval: Time between frames in milliseconds
        figsize: Figure size for animation
        output_path: Path to save animation (if None, just returns animation object)
        gif_format: If True, save as GIF; if False, save as MP4
        
    Returns:
        matplotlib FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    def animate_frame(frame_num):
        ax.clear()
        plot_functions[frame_num](ax)
        ax.set_title(f"Frame {frame_num + 1}/{len(plot_functions)}")
    
    ani = animation.FuncAnimation(
        fig,
        animate_frame,
        frames=len(plot_functions),
        interval=interval,
        repeat=True
    )
    
    if output_path:
        _save_animation(ani, output_path, gif_format)
    
    return ani


def create_slideshow_animation(
    figures: List[plt.Figure],
    interval: int = 1000,
    output_path: Optional[str] = None,
    gif_format: bool = True
) -> animation.ArtistAnimation:
    """
    Create a slideshow animation from a list of matplotlib figures.
    
    Args:
        figures: List of matplotlib Figure objects
        interval: Time between slides in milliseconds
        output_path: Path to save animation
        gif_format: If True, save as GIF; if False, save as MP4
        
    Returns:
        matplotlib ArtistAnimation object
    """
    if len(figures) == 0:
        raise ValueError("No figures provided")
    
    # Create a new figure for the animation
    ani_fig, ani_ax = plt.subplots(figsize=figures[0].get_size_inches())
    ani_ax.axis('off')
    
    # Convert figures to images and create animation frames
    temp_dir = tempfile.mkdtemp()
    animation_frames = []
    
    try:
        for i, fig in enumerate(figures):
            # Save figure to temporary file
            temp_path = os.path.join(temp_dir, f"slide_{i}.png")
            fig.savefig(temp_path, bbox_inches='tight')
            
            # Load as image and add to animation
            img = Image.open(temp_path)
            im = ani_ax.imshow(img)
            animation_frames.append([im])
            
            # Clean up
            os.remove(temp_path)
        
        # Create animation
        ani = animation.ArtistAnimation(
            ani_fig,
            animation_frames,
            interval=interval,
            blit=True,
            repeat=True
        )
        
        if output_path:
            _save_animation(ani, output_path, gif_format)
        
        return ani
        
    finally:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def save_frames_as_images(
    grids: List[List[List[Callable]]],
    output_dir: str,
    dpi: int = 300,
    prefix: str = "frame"
) -> List[str]:
    """
    Save each frame as a separate image file.
    
    Args:
        grids: List of grid layouts
        output_dir: Directory to save images
        dpi: Resolution for saved images
        prefix: Prefix for image filenames
        
    Returns:
        List of saved image file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, grid in enumerate(grids):
        # Create subplot from grid
        fig, axs, img_plots, texts = create_subplot(grid)
        
        # Save image
        filename = f"{prefix}_{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        saved_paths.append(filepath)
        
        # Close figure
        plt.close(fig)
        
        print(f"Saved frame {i+1}/{len(grids)}: {filename}")
    
    return saved_paths


def create_comparison_animation(
    grids_list: List[List[List[List[Callable]]]],
    labels: List[str],
    interval: int = 500,
    output_path: Optional[str] = None,
    gif_format: bool = False
) -> animation.ArtistAnimation:
    """
    Create an animation comparing multiple evolution runs side by side.
    
    Args:
        grids_list: List of grid sequences, one for each evolution run
        labels: Labels for each evolution run
        interval: Time between frames in milliseconds
        output_path: Path to save animation
        gif_format: If True, save as GIF; if False, save as MP4
        
    Returns:
        matplotlib ArtistAnimation object
    """
    if len(grids_list) != len(labels):
        raise ValueError("Number of grid sequences must match number of labels")
    
    # Ensure all sequences have the same length
    min_length = min(len(grids) for grids in grids_list)
    grids_list = [grids[:min_length] for grids in grids_list]
    
    # Create combined grids for each frame
    combined_grids = []
    for frame_idx in range(min_length):
        # Combine grids horizontally for this frame
        frame_grids = [grids_list[run_idx][frame_idx] for run_idx in range(len(grids_list))]
        
        # Create a combined grid with labels
        combined_grid = []
        max_rows = max(len(grid) for grid in frame_grids)
        
        for row_idx in range(max_rows):
            combined_row = []
            for run_idx, grid in enumerate(frame_grids):
                if row_idx == 0:
                    # Add label for first row
                    label_func = lambda ax, label=labels[run_idx]: ax.text(
                        0.5, 0.5, label, ha='center', va='center', 
                        fontsize=14, fontweight='bold'
                    )
                    combined_row.append(label_func)
                
                if row_idx < len(grid):
                    combined_row.extend(grid[row_idx])
                else:
                    # Add empty plots if grid is shorter
                    for _ in range(len(frame_grids[0][0]) if frame_grids[0] else 1):
                        combined_row.append(lambda ax: ax.axis('off'))
            
            combined_grid.append(combined_row)
        
        combined_grids.append(combined_grid)
    
    # Create animation from combined grids
    return animate_grids(
        combined_grids,
        interval=interval,
        output_path=output_path,
        gif_format=gif_format
    )


class AnimationBuilder:
    """
    Builder class for creating complex animations with various configuration options.
    """
    
    def __init__(self):
        self.grids = []
        self.interval = 200
        self.dpi = 300
        self.figsize = (12, 8)
        self.gif_format = False
        self.frame_skip = 1
        self.output_path = None
        self.cleanup = True
        
    def add_grid(self, grid: List[List[Callable]]) -> 'AnimationBuilder':
        """Add a grid layout to the animation sequence."""
        self.grids.append(grid)
        return self
        
    def set_interval(self, interval: int) -> 'AnimationBuilder':
        """Set the time between frames in milliseconds."""
        self.interval = interval
        return self
        
    def set_dpi(self, dpi: int) -> 'AnimationBuilder':
        """Set the resolution for rendered frames."""
        self.dpi = dpi
        return self
        
    def set_figsize(self, figsize: tuple) -> 'AnimationBuilder':
        """Set the figure size for the animation."""
        self.figsize = figsize
        return self
        
    def set_output_format(self, gif_format: bool) -> 'AnimationBuilder':
        """Set whether to output as GIF (True) or MP4 (False)."""
        self.gif_format = gif_format
        return self
        
    def set_frame_skip(self, frame_skip: int) -> 'AnimationBuilder':
        """Set frame skipping (1 = no skip, 2 = every other frame, etc.)."""
        self.frame_skip = frame_skip
        return self
        
    def set_output_path(self, output_path: str) -> 'AnimationBuilder':
        """Set the output file path."""
        self.output_path = output_path
        return self
        
    def set_cleanup(self, cleanup: bool) -> 'AnimationBuilder':
        """Set whether to clean up temporary files."""
        self.cleanup = cleanup
        return self
        
    def build(self) -> animation.ArtistAnimation:
        """Build and return the animation."""
        if not self.grids:
            raise ValueError("No grids added to animation")
            
        return animate_grids(
            self.grids,
            interval=self.interval,
            dpi=self.dpi,
            gif_format=self.gif_format,
            frame_skip=self.frame_skip,
            output_path=self.output_path,
            cleanup=self.cleanup
        )


def check_animation_requirements() -> dict:
    """
    Check if required libraries and tools are available for animation creation.
    
    Returns:
        Dictionary with availability status of different animation backends
    """
    requirements = {
        'matplotlib': False,
        'PIL': False,
        'ffmpeg': False,
        'imagemagick': False
    }
    
    # Check matplotlib
    try:
        import matplotlib.animation
        requirements['matplotlib'] = True
    except ImportError:
        pass
    
    # Check PIL
    try:
        from PIL import Image
        requirements['PIL'] = True
    except ImportError:
        pass
    
    # Check ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        requirements['ffmpeg'] = result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check imagemagick
    try:
        import subprocess
        result = subprocess.run(['convert', '-version'], 
                              capture_output=True, text=True, timeout=5)
        requirements['imagemagick'] = result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return requirements


def get_recommended_settings(num_frames: int, file_size_limit_mb: float = 50) -> dict:
    """
    Get recommended animation settings based on number of frames and file size constraints.
    
    Args:
        num_frames: Number of frames in the animation
        file_size_limit_mb: Target maximum file size in MB
        
    Returns:
        Dictionary with recommended settings
    """
    settings = {
        'interval': 200,
        'dpi': 150,
        'frame_skip': 1,
        'gif_format': False
    }
    
    # Adjust settings based on number of frames
    if num_frames > 100:
        settings['frame_skip'] = max(1, num_frames // 50)  # Limit to ~50 frames
        settings['interval'] = 150
    elif num_frames > 50:
        settings['interval'] = 300
    
    # Adjust for file size
    estimated_size_mb = num_frames * settings['dpi'] / 1000  # Rough estimate
    if estimated_size_mb > file_size_limit_mb:
        # Reduce DPI to meet file size constraint
        settings['dpi'] = max(75, int(settings['dpi'] * file_size_limit_mb / estimated_size_mb))
        
        # If still too large, prefer GIF format
        if estimated_size_mb > file_size_limit_mb * 1.5:
            settings['gif_format'] = True
    
    return settings


# Utility functions for backward compatibility
def animate_grids_legacy(grids, interval=100, dpi=300, gif_format=False, frame_skip=1):
    """Legacy function for backward compatibility."""
    return animate_grids(
        grids=grids,
        interval=interval,
        dpi=dpi,
        gif_format=gif_format,
        frame_skip=frame_skip
    )


if __name__ == "__main__":
    # Example usage and testing
    print("=== Animation Utils Testing ===")
    
    # Check requirements
    requirements = check_animation_requirements()
    print("Animation requirements:")
    for requirement, available in requirements.items():
        status = "✓" if available else "✗"
        print(f"  {status} {requirement}")
    
    # Test recommended settings
    settings = get_recommended_settings(100, 25)
    print(f"\nRecommended settings for 100 frames, 25MB limit:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    print("\n=== Testing Complete ===")