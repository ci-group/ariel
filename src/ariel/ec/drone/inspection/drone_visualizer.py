"""
Modern, clean interface for drone genome visualization.

This module provides a high-level, object-oriented interface for visualizing
drone genomes with support for multiple coordinate systems, cylinder visualization,
and clean APIs.
"""

from __future__ import annotations
from typing import Optional, Union, Dict, Any, Tuple, List
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import gc

from . import utils as u

try:
    import trimesh
    import multiprocessing
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from ariel.ec.drone.genome_handlers.conversions.arm_conversions import Cylinder

# Launch in separate process to avoid blocking
def show_scene(scene_obj):
    scene_obj.show(viewer='gl')

@dataclass
class VisualizationConfig:
    """Configuration for drone visualization styling and display options."""

    # Basic display options
    circle_radius: float = 0.0254  # 2-inch propeller radius in meters
    scale_factor: float = 1.2
    
    # 3D view options
    elevation: float = 30
    azimuth: float = 30
    include_motor_orientation: bool = True
    
    # Axis and labeling options
    show_axis: bool = True
    show_axis_ticks: bool = True
    axis_labels: bool = True
    fontsize: int = 10
    
    # 2D specific options
    show_limits: bool = True
    include_motor_orientation_2d: int = 0  # 0=none, 1=arrows, 2=arrows+labels
    
    # Colors
    arm_color: str = 'k'
    motor_color: str = 'm' 
    orientation_color: str = 'r'
    limit_circle_color: str = 'black'
    cylinder_color: str = 'b'
    constraint_min_color: str = 'r'
    constraint_max_color: str = 'g'
    
    # Line styles
    arm_linestyle: str = '--'
    limit_linestyle: str = ':'
    
    # Cylinder visualization options
    cylinder_alpha: float = 0.6
    cylinder_num_lines: int = 6
    show_constraints: bool = True
    constraint_alpha: float = 0.1


class DroneVisualizer:
    """
    High-level interface for visualizing drone genomes and cylinder arrangements.
    
    Supports both Cartesian and polar coordinate genome formats with automatic
    detection and conversion. Provides clean APIs for 2D, 3D, blueprint, and
    cylinder visualizations.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the drone visualizer.
        
        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config if config is not None else VisualizationConfig()
    
    def plot_3d(
        self,
        genome_data: Any,
        ax: Optional[Axes3D] = None,
        title: Optional[str] = None,
        fitness: Optional[float] = None,
        generation: Optional[int] = None,
        **kwargs
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Create a 3D visualization of a drone genome.
        
        Args:
            genome_data: Genome handler object or numpy array
            ax: Existing 3D axes to plot on. Creates new if None.
            title: Custom title for the plot
            fitness: Fitness value to display in title
            generation: Generation number to display in title
            **kwargs: Additional config overrides
            
        Returns:
            Tuple of (figure, axes)
        """
        # Extract standardized data
        data = u.auto_extract_genome_data(genome_data)

        # Create figure/axes if needed
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Apply config overrides
        config = self._apply_config_overrides(**kwargs)
        
        # Plot each motor/arm
        for i, (pos, orient, direction) in enumerate(zip(
            data['positions'], 
            data['orientations'], 
            data['directions']
        )):
            x, y, z = pos
            
            # Draw arm to origin
            ax.plot([0, x], [0, y], [0, z], 
                   color=config.arm_color, 
                   linestyle=config.arm_linestyle)
            
            # Draw motor orientation if enabled
            if config.include_motor_orientation:
                self._draw_motor_orientation_3d(ax, pos, orient, config)
            
            # Draw propeller circle and direction
            self._draw_propeller_3d(ax, pos, orient, direction, config)
        
        # Set up axes and limits
        self._setup_3d_axes(ax, data['positions'], config)
        
        # Set title
        self._set_title(ax, title, fitness, generation, config)
        
        # Set view angle
        ax.view_init(elev=config.elevation, azim=config.azimuth)
        
        return fig, ax
    
    def plot_2d(
        self,
        genome_data: Any,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        fitness: Optional[float] = None,
        generation: Optional[int] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a 2D top-down visualization of a drone genome.
        
        Args:
            genome_data: Genome handler object or numpy array
            ax: Existing 2D axes to plot on. Creates new if None.
            title: Custom title for the plot
            fitness: Fitness value to display in title
            generation: Generation number to display in title
            xlim: X-axis limits (auto-computed if None)
            ylim: Y-axis limits (auto-computed if None)
            **kwargs: Additional config overrides
            
        Returns:
            Tuple of (figure, axes)
        """
        # Extract standardized data
        data = u.auto_extract_genome_data(genome_data)
        
        # Create figure/axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Apply config overrides
        config = self._apply_config_overrides(**kwargs)
        
        # Plot each motor/arm
        for i, (pos, orient, direction) in enumerate(zip(
            data['positions'], 
            data['orientations'], 
            data['directions']
        )):
            x, y, z = pos
            
            # Draw arm to origin (2D projection)
            ax.plot([0, x], [0, y], 
                   color=config.arm_color, 
                   linestyle=config.arm_linestyle)
            
            # Draw motor circle
            circle = plt.Circle((x, y), config.circle_radius, 
                              edgecolor=config.motor_color, 
                              facecolor='none')
            ax.add_patch(circle)
            
            # Draw propeller direction arrow
            self._draw_propeller_direction_2d(ax, (x, y), direction, config)
            
            # Draw motor orientation if enabled
            if config.include_motor_orientation_2d > 0:
                self._draw_motor_orientation_2d(ax, pos, orient, config)
        
        # Draw limit circles if enabled
        if config.show_limits:
            self._draw_limit_circles(ax, config)
        
        # Set up axes and limits
        self._setup_2d_axes(ax, data['positions'], xlim, ylim, config)
        
        # Set title
        self._set_title(ax, title, fitness, generation, config)
        
        return fig, ax
    
    def plot_blueprint(
        self,
        genome_data: Any,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 12),
        **kwargs
    ) -> Tuple[plt.Figure, List[Axes3D]]:
        """
        Create a blueprint-style visualization with 4 different views.
        
        Args:
            genome_data: Genome handler object or numpy array
            title: Title for the entire figure
            figsize: Figure size
            **kwargs: Additional config overrides
            
        Returns:
            Tuple of (figure, list of axes)
        """
        # Define views: (elevation, azimuth, label)
        views = [
            (0, 90, "Front View"),    # Front
            (0, 0, "Side View"),      # Side  
            (90, 0, "Top View"),      # Top
            (30, 45, "Isometric")     # Isometric
        ]
        
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        
        axes = []
        for i, (elev, azim, view_label) in enumerate(views):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            # Plot with specific view angle
            config_override = {**kwargs, 'elevation': elev, 'azimuth': azim}
            self.plot_3d(genome_data, ax=ax, title=view_label, **config_override)
            
            axes.append(ax)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        return fig, axes


    def plot_cylinders_3d(
        self,
        cylinders: List[Cylinder],
        ax: Optional[Axes3D] = None,
        title: Optional[str] = None,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
        color_map: Optional[str] = 'jet',
        **kwargs
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Create a 3D visualization of cylinder arrangements.
        
        Args:
            cylinders: List of Cylinder objects to visualize
            ax: Existing 3D axes to plot on. Creates new if None.
            title: Custom title for the plot
            d_min: Minimum distance constraint (for visualization)
            d_max: Maximum distance constraint (for visualization)
            color_map: Colormap name for cylinder colors
            **kwargs: Additional config overrides
            
        Returns:
            Tuple of (figure, axes)
        """
        # Create figure/axes if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Apply config overrides
        config = self._apply_config_overrides(**kwargs)
        
        # Create colormap for cylinders
        if len(cylinders) > 1:
            colors = plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(cylinders)))
        else:
            colors = [config.cylinder_color]
        
        # Draw constraint spheres if provided
        if config.show_constraints and (d_min is not None or d_max is not None):
            self._draw_constraint_spheres(ax, d_min, d_max, config)
        
        # Draw cylinders
        max_extent = 0
        for i, cylinder in enumerate(cylinders):
            color = colors[i] if len(cylinders) > 1 else config.cylinder_color
            self._draw_cylinder_3d(ax, cylinder, color, config)
            
            # Track extent for bounds
            p1, p2 = cylinder.get_endpoints()
            max_extent = max(max_extent, 
                           np.linalg.norm(p1), 
                           np.linalg.norm(p2))
        
        # Set up axes
        if d_max is not None:
            extent = d_max * 1.1
        else:
            extent = max_extent * config.scale_factor
            
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        
        if config.axis_labels:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Add legend for constraints
        if config.show_constraints and (d_min is not None or d_max is not None):
            self._add_constraint_legend(ax, d_min, d_max)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=config.fontsize)
        
        # Set view angle
        ax.view_init(elev=config.elevation, azim=config.azimuth)
        
        return fig, ax
    
    def plot_cylinders_trimesh(self, cylinders: List[Cylinder]) -> None:
        """
        Visualize cylinders using Trimesh interactive viewer.
        
        Args:
            cylinders: List of Cylinder objects to visualize
            
        Raises:
            ImportError: If trimesh is not available
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("Trimesh is required for interactive 3D visualization. "
                            "Install with: pip install trimesh[easy]")
        
        cylinder_meshes = []
        
        for cyl in cylinders:
            # Create cylinder mesh along X-axis (trimesh default)
            mesh = trimesh.creation.cylinder(
                radius=cyl.radius,
                height=cyl.height,
                sections=32,
                segment=None,
                transform=np.eye(4)
            )
            
            # Apply the cylinder's transformation matrix
            mesh.apply_transform(cyl.transform)
            cylinder_meshes.append(mesh)
        
        # Create scene and show
        scene = trimesh.Scene(cylinder_meshes)
        
        p = multiprocessing.Process(target=show_scene, args=(scene,))
        p.start()
        
        return p  # Return process handle in case user wants to manage it
    
    def _apply_config_overrides(self, **kwargs) -> VisualizationConfig:
        """Apply configuration overrides to current config."""
        import copy
        config = copy.deepcopy(self.config)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _draw_motor_orientation_3d(
        self, 
        ax: Axes3D, 
        position: npt.NDArray, 
        orientation: npt.NDArray, 
        config: VisualizationConfig
    ) -> None:
        """Draw motor orientation vector in 3D."""
        x, y, z = position
        roll, pitch, yaw = orientation
        
        # Create rotation matrix using drone's coordinate system
        R = u.create_rotation_matrix_euler(roll, pitch, yaw)
        
        # Create unit z-axis vector and rotate it
        z_axis = np.array([0, 0, config.circle_radius * 2])
        rotated_z = R @ z_axis
        
        # Draw orientation vector
        ax.plot([x, x + rotated_z[0]], 
               [y, y + rotated_z[1]], 
               [z, z + rotated_z[2]], 
               color=config.orientation_color)
    
    def _draw_propeller_3d(
        self, 
        ax: Axes3D, 
        position: npt.NDArray, 
        orientation: npt.NDArray, 
        direction: int, 
        config: VisualizationConfig
    ) -> None:
        """Draw propeller circle and direction arrow in 3D."""
        x, y, z = position
        roll, pitch, yaw = orientation
        
        # Create rotation matrix using drone's coordinate system
        R = u.create_rotation_matrix_euler(roll, pitch, yaw)

        # Create circle points in local XY plane (perpendicular to motor axis)
        num_points = 100
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_local = np.array([
            config.circle_radius * np.cos(theta),
            config.circle_radius * np.sin(theta),
            np.zeros_like(theta)
        ]).T
        
        # Rotate and translate circle
        circle_world = (R @ circle_local.T).T + position

        ax.plot(circle_world[:, 0], circle_world[:, 1], circle_world[:, 2], 
               color=config.motor_color)
        
        # Draw direction arrow
        if direction == 1:  # Clockwise
            arrow_start_idx = 75
            arrow_end_idx = 85
        else:  # Counter-clockwise
            arrow_start_idx = 25
            arrow_end_idx = 15
        
        arrow_start = circle_world[arrow_start_idx]
        arrow_end = circle_world[arrow_end_idx]
        arrow_vec = arrow_end - arrow_start
        
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                 arrow_vec[0], arrow_vec[1], arrow_vec[2],
                 color=config.motor_color, arrow_length_ratio=0.3)
    
    def _draw_propeller_direction_2d(
        self, 
        ax: plt.Axes, 
        position: Tuple[float, float], 
        direction: int, 
        config: VisualizationConfig
    ) -> None:
        """Draw propeller direction arrow in 2D."""
        x, y = position
        
        # Create arrow points on circle
        if direction == 1:  # Clockwise
            start_angle = 3 * np.pi / 4
            end_angle = np.pi / 2
        else:  # Counter-clockwise
            start_angle = np.pi / 4
            end_angle = np.pi / 2
        
        arrow_start = np.array([
            x + config.circle_radius * np.cos(start_angle),
            y + config.circle_radius * np.sin(start_angle)
        ])
        arrow_end = np.array([
            x + config.circle_radius * np.cos(end_angle),
            y + config.circle_radius * np.sin(end_angle)
        ])
        
        # Draw arrow
        arrowstyle = matplotlib.patches.ArrowStyle(
            "Fancy", head_length=0.4, head_width=0.4, tail_width=0.1
        )
        ax.annotate('', xy=arrow_end, xytext=arrow_start,
                   arrowprops=dict(facecolor=config.motor_color, 
                                 edgecolor=config.motor_color, 
                                 arrowstyle=arrowstyle))
    
    def _draw_motor_orientation_2d(
        self, 
        ax: plt.Axes, 
        position: npt.NDArray, 
        orientation: npt.NDArray, 
        config: VisualizationConfig
    ) -> None:
        """Draw motor orientation in 2D."""
        x, y, z = position
        roll, pitch, yaw = orientation
        
        # Project 3D orientation to 2D
        scale = config.circle_radius * 1.25
        x_proj = scale * np.sin(pitch) * np.cos(yaw)
        y_proj = scale * np.sin(pitch) * np.sin(yaw)
        
        # Draw orientation vector
        ax.quiver(x, y, x_proj, y_proj, 
                 angles='xy', scale_units='xy', scale=1, 
                 color='b')
        
        # Add angle labels if requested
        if config.include_motor_orientation_2d == 2:
            label = f"Y:{np.round(np.degrees(yaw),0)}°, P:{np.round(np.degrees(pitch),0)}°"
            ax.text(x - config.circle_radius * 1.05, y, label, 
                   fontsize=6, ha='right')
    
    def _draw_limit_circles(self, ax: plt.Axes, config: VisualizationConfig) -> None:
        """Draw constraint limit circles in 2D."""
        # Inner limit circle
        inner_circle = plt.Circle((0, 0), 0.09, 
                                linestyle=config.limit_linestyle, 
                                edgecolor=config.limit_circle_color, 
                                facecolor='none')
        ax.add_patch(inner_circle)
        
        # Outer limit circle  
        outer_circle = plt.Circle((0, 0), 0.4, 
                                linestyle=config.limit_linestyle, 
                                edgecolor=config.limit_circle_color, 
                                facecolor='none')
        ax.add_patch(outer_circle)
    
    def _draw_cylinder_3d(
        self, 
        ax: Axes3D, 
        cylinder: Cylinder, 
        color: str, 
        config: VisualizationConfig
    ) -> None:
        """Draw a single cylinder in 3D using the correct drone orientation system."""
        # Get cylinder endpoints
        p1, p2 = cylinder.get_endpoints()
        
        # Generate circle points using drone's coordinate system
        # The cylinder orientation should match the drone motor orientation
        top_circle_points = self._generate_oriented_circle_points(
            cylinder.radius, cylinder.orientation, p1
        )
        bottom_circle_points = self._generate_oriented_circle_points(
            cylinder.radius, cylinder.orientation, p2
        )
        
        # Draw top and bottom circles
        ax.plot(top_circle_points[:, 0], top_circle_points[:, 1], 
               top_circle_points[:, 2], color=color)
        ax.plot(bottom_circle_points[:, 0], bottom_circle_points[:, 1], 
               bottom_circle_points[:, 2], color=color)
        
        # Draw connecting lines
        indices = np.arange(0, len(top_circle_points), 
                          len(top_circle_points) // config.cylinder_num_lines)
        for i in indices:
            ax.plot([top_circle_points[i, 0], bottom_circle_points[i, 0]], 
                   [top_circle_points[i, 1], bottom_circle_points[i, 1]], 
                   [top_circle_points[i, 2], bottom_circle_points[i, 2]], 
                   color=color)
    
    def _generate_oriented_circle_points(
        self, 
        radius: float, 
        orientation: npt.NDArray, 
        center: npt.NDArray,
        num_points: int = 100
    ) -> npt.NDArray:
        """Generate circle points with proper orientation matching drone system."""
        # Convert quaternion to rotation matrix if needed
        # Quaternion [qx, qy, qz, qw] 
        R = u.create_rotation_matrix_quaternion(orientation)
        
        # Create circle points in local XY plane (consistent with drone motors)
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_local = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta), 
            np.zeros_like(theta)
        ]).T
        
        # Apply rotation and translation
        circle_world = (R @ circle_local.T).T + center
        
        return circle_world
    
    def _draw_constraint_spheres(
        self, 
        ax: Axes3D, 
        d_min: Optional[float], 
        d_max: Optional[float], 
        config: VisualizationConfig
    ) -> None:
        """Draw constraint spheres for distance limits."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        if d_min is not None:
            x = d_min * np.outer(np.cos(u), np.sin(v))
            y = d_min * np.outer(np.sin(u), np.sin(v))
            z = d_min * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color=config.constraint_min_color, 
                          alpha=config.constraint_alpha)
        
        if d_max is not None:
            x = d_max * np.outer(np.cos(u), np.sin(v))
            y = d_max * np.outer(np.sin(u), np.sin(v))
            z = d_max * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color=config.constraint_max_color, 
                          alpha=config.constraint_alpha)
    
    def _add_constraint_legend(
        self, 
        ax: Axes3D, 
        d_min: Optional[float], 
        d_max: Optional[float]
    ) -> None:
        """Add legend for constraint visualization."""
        from matplotlib.lines import Line2D
        
        legend_elements = []
        if d_min is not None:
            legend_elements.append(
                Line2D([0], [0], color=self.config.constraint_min_color, lw=2, 
                      label=f'Min Distance ({d_min:.3f})')
            )
        if d_max is not None:
            legend_elements.append(
                Line2D([0], [0], color=self.config.constraint_max_color, lw=2,
                      label=f'Max Distance ({d_max:.3f})')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
    
    def _setup_3d_axes(
        self, 
        ax: Axes3D, 
        positions: npt.NDArray, 
        config: VisualizationConfig
    ) -> None:
        """Set up 3D axes properties and limits."""
        bounds = u.compute_visualization_bounds(positions, config.scale_factor)
        
        ax.set_xlim(bounds['xlim'])
        ax.set_ylim(bounds['ylim'])
        ax.set_zlim(bounds['zlim'])
        
        if config.axis_labels:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        if not config.show_axis:
            ax.set_axis_off()
        
        if not config.show_axis_ticks:
            ax.tick_params(labelbottom=False, labeltop=False, 
                          labelleft=False, labelright=False)
    
    def _setup_2d_axes(
        self, 
        ax: plt.Axes, 
        positions: npt.NDArray, 
        xlim: Optional[Tuple[float, float]], 
        ylim: Optional[Tuple[float, float]], 
        config: VisualizationConfig
    ) -> None:
        """Set up 2D axes properties and limits."""
        if xlim is None or ylim is None:
            bounds = u.compute_visualization_bounds(positions, config.scale_factor)
            xlim = xlim or bounds['xlim']
            ylim = ylim or bounds['ylim']
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True)
        
        ax.tick_params(labelbottom=config.show_axis_ticks, 
                      labelleft=config.show_axis_ticks)
    
    def _set_title(
        self, 
        ax: Union[Axes3D, plt.Axes], 
        title: Optional[str], 
        fitness: Optional[float], 
        generation: Optional[int], 
        config: VisualizationConfig
    ) -> None:
        """Set plot title with fitness and generation info."""
        if title:
            ax.set_title(title, fontsize=config.fontsize)
        elif fitness is not None or generation is not None:
            title_parts = []
            if generation is not None:
                title_parts.append(f'G: {generation}')
            if fitness is not None:
                title_parts.append(f'F: {np.round(fitness, 2)}')
            
            if title_parts:
                ax.set_title(', '.join(title_parts), fontsize=config.fontsize)


# Convenience functions for backward compatibility and ease of use
def visualize_cylinders(
    cylinders: List[Cylinder],
    title: Optional[str] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    use_trimesh: bool = False,
    config: Optional[VisualizationConfig] = None
) -> Union[Tuple[plt.Figure, Axes3D], multiprocessing.Process]:
    """
    Convenience function to visualize a single set of cylinders.
    
    Args:
        cylinders: List of Cylinder objects
        title: Plot title
        d_min: Minimum distance constraint
        d_max: Maximum distance constraint 
        use_trimesh: Whether to use interactive Trimesh viewer
        config: Visualization configuration
        
    Returns:
        Either (figure, axes) tuple for matplotlib or Process handle for trimesh
    """
    gc.collect()  # Clean up memory before visualization
    
    visualizer = DroneVisualizer(config)
    
    if use_trimesh:
        return visualizer.plot_cylinders_trimesh(cylinders)
    else:
        return visualizer.plot_cylinders_3d(
            cylinders, title=title, d_min=d_min, d_max=d_max
        )


def compare_cylinder_arrangements(
    arrangements: List[List[Cylinder]],
    labels: Optional[List[str]] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    figsize: Tuple[float, float] = (15, 7),
    config: Optional[VisualizationConfig] = None
) -> Tuple[plt.Figure, List[Axes3D]]:
    """
    Compare multiple cylinder arrangements side by side.
    
    Args:
        arrangements: List of cylinder arrangements to compare
        labels: Labels for each arrangement
        d_min: Minimum distance constraint
        d_max: Maximum distance constraint
        figsize: Figure size
        config: Visualization configuration
        
    Returns:
        Tuple of (figure, list of axes)
    """
    visualizer = DroneVisualizer(config)
    
    n_arrangements = len(arrangements)
    cols = min(n_arrangements, 3)  # Max 3 columns
    rows = (n_arrangements + cols - 1) // cols  # Ceiling division
    
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    
    axes = []
    for i, cylinders in enumerate(arrangements):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Generate title
        if labels and i < len(labels):
            title = labels[i]
        else:
            title = f"Arrangement {i+1}"
        
        # Plot the arrangement
        visualizer.plot_cylinders_3d(
            cylinders, ax=ax, title=title, d_min=d_min, d_max=d_max
        )
        
        axes.append(ax)
    
    return fig, axes


def create_drone_cylinder_combo(
    genome_data: Any,
    cylinders: List[Cylinder],
    title: Optional[str] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    figsize: Tuple[float, float] = (15, 7),
    config: Optional[VisualizationConfig] = None
) -> Tuple[plt.Figure, Tuple[Axes3D, Axes3D]]:
    """
    Create side-by-side visualization of drone genome and cylinder arrangement.
    
    Args:
        genome_data: Drone genome data
        cylinders: List of Cylinder objects
        title: Overall figure title
        d_min: Minimum distance constraint
        d_max: Maximum distance constraint
        figsize: Figure size
        config: Visualization configuration
        
    Returns:
        Tuple of (figure, (drone_axes, cylinder_axes))
    """
    visualizer = DroneVisualizer(config)
    
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.1)
    
    # Plot drone genome
    ax1 = fig.add_subplot(121, projection='3d')
    visualizer.plot_3d(genome_data, ax=ax1, title="Drone Configuration")
    
    # Plot cylinders
    ax2 = fig.add_subplot(122, projection='3d')
    visualizer.plot_cylinders_3d(
        cylinders, ax=ax2, title="Cylinder Arrangement", 
        d_min=d_min, d_max=d_max
    )
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    return fig, (ax1, ax2)