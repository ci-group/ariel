"""
Animation and Real-time Visualization

This module provides OpenCV-based animation and real-time visualization
capabilities for drone simulation including trajectory playback and recording.

Functions:
    view: Real-time drone state visualization with interactive controls
    animate: Trajectory playback with timeline control and recording
    get_drone_state_zero: Default drone state for testing
    nothing: Placeholder function for OpenCV trackbars
"""

import numpy as np
import cv2
import time
import os
import copy

from .graphics_3d import Camera
from .drone_visualization import create_grid, create_path, create_drone, set_thrust

# Screen and visualization constants
DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 700
DEFAULT_FPS = 100
THRUST_SCALE = 0.2
ZOOM_FACTOR = 1.05
GATE_SIZE = 1.5
GRID_SIZE = 20
DRONE_BOX_SIZE = [0.02, 0.02, 0.02]
PROP_RADIUS = 0.0254  # 2-inch propeller radius in meters
PATH_SUBSAMPLE = 5
THRUST_BASE_LEN = 0.0  # Base arrow length so direction is visible at low thrust

# Camera constants
CAMERA_MATRIX = np.array([[1.e+3, 0., DEFAULT_WIDTH/2], [0., 1.e+3, DEFAULT_HEIGHT/2], [0., 0., 1.]])
DIST_COEFFS = np.array([0., 0., 0., 0., 0.])

# Color mappings for OpenCV (BGR format)
COLORS_BGR = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'brown': (42, 42, 165),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (200, 200, 200),
    'gate_orange': (0, 140, 255)
}

def nothing(x):
    """Placeholder function for OpenCV trackbar callbacks."""
    pass


_OVERLAY_POSITIONS = {
    'upper left', 'upper right', 'lower left', 'lower right',
    'top left', 'top right', 'bottom left', 'bottom right',
}


def _overlay_anchor(position, text_w, text_h, frame_w, frame_h, pad=15):
    """Resolve a corner keyword to an (x, y) anchor for cv2.putText.

    cv2.putText's origin is the text's baseline-left, so ``y`` must be below
    the top of the text box.
    """
    position = (position or 'lower right').lower()
    if position not in _OVERLAY_POSITIONS:
        position = 'lower right'
    top = position.startswith('upper') or position.startswith('top')
    left = position.endswith('left')
    x = pad if left else (frame_w - text_w - pad)
    y = (pad + text_h) if top else (frame_h - pad)
    return x, y

def create_camera(view_type='iso', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """
    Create a camera with predefined settings based on view type.
    
    Args:
        view_type: Camera view type ('top', 'iso', 'isometric', or other)
        width: Screen width
        height: Screen height
        
    Returns:
        Camera: Configured camera object
    """
    if view_type == 'top':
        cam = Camera(
            pos=np.array([-12., 0., 0.]),
            theta=[0., -np.pi/2, 0.],
            cameraMatrix=CAMERA_MATRIX,
            distCoeffs=DIST_COEFFS
        )
        cam.r = [-3.8, 0, 0]
    elif view_type in ['isometric', 'iso']:
        cam = Camera(
            pos=np.array([0., 0., 0.]),
            theta=[0,-np.pi/8,-np.pi/4],
            cameraMatrix=CAMERA_MATRIX,
            distCoeffs=DIST_COEFFS
        )
        cam.r = [-2.6, 0, 0]
    else:
        cam = Camera(
            pos=np.array([0., 0., 0.]),
            theta=[0,-np.pi/8,-np.pi/4],
            cameraMatrix=CAMERA_MATRIX,
            distCoeffs=DIST_COEFFS
        )
        cam.r = [-15, 0, 0]
    
    return cam

def create_gate_geometry(gate_size=None):
    """
    Create gate geometry objects for visualization.

    Args:
        gate_size: Gate size in meters. If None, uses GATE_SIZE constant.

    Returns:
        tuple: (gate, gate_collision_box) path objects
    """
    n = gate_size if gate_size is not None else GATE_SIZE
    gate = create_path(np.array([
        [0, n/2, n/2],
        [0, n/2, -n/2],
        [0, -n/2, -n/2],
        [0, -n/2, n/2]
    ]), loop=True)

    gate_collision_box = create_path(np.array([
        [0, 1., 1.],
        [0, 1., -1.],
        [0, -1., -1.],
        [0, -1., 1.]
    ]), loop=True)

    return gate, gate_collision_box

def handle_keyboard_input_view(key, cam, auto_play, follow, draw_forces, draw_path, record, out, record_file):
    """
    Handle keyboard input for the view function.
    
    Returns:
        tuple: (auto_play, follow, draw_forces, draw_path, record, should_exit)
    """
    should_exit = False
    
    if key == 27:  # ESC - exit
        if record:
            print('recording ended')
            out.release()
            print('recording saved in ' + record_file)
        should_exit = True
    elif key == ord('p'):  # P - toggle auto-play
        auto_play = not auto_play
    elif key == ord('f'):  # F - toggle follow mode
        follow = not follow
    elif key == ord('s'):  # S - toggle force display
        draw_forces = not draw_forces
    elif key == ord('d'):  # D - toggle path display
        draw_path = not draw_path
    elif key == ord('1'):  # 1 - zoom in
        cam.zoom(ZOOM_FACTOR)
    elif key == ord('2'):  # 2 - zoom out
        cam.zoom(1/ZOOM_FACTOR)
    elif key == ord('r'):  # R - toggle recording
        if record:
            print('recording ended')
            out.release()
            print('recording saved in ' + record_file)
        else:
            print('recording started')
        record = not record
        
    return auto_play, follow, draw_forces, draw_path, record, should_exit

def draw_drone_and_forces(drone, forces, pos, ori, u, state, draw_forces, scl, frame, cam):
    """
    Draw drone(s) and force vectors.

    Note: ``u`` here is the policy action in [-1, 1]. Under the reference-form
    motor model (Session 5 dynamics migration), the physical motor thrust
    magnitude scales with U = (u + 1) / 2 ∈ [0, 1] (motor never reverses;
    minimum is the idle speed w_min). We therefore render arrow length using
    that mapping so arrows point in the actual thrust direction at all
    throttles, rather than flipping backward whenever u < 0.
    """
    if len(pos.shape) == 1:  # Single drone
        drone.translate(pos-drone.pos)
        drone.rotate(ori)
        thrust_magnitude = (np.asarray(u, dtype=float) + 1.0) / 2.0
        set_thrust(drone, forces, thrust_magnitude * scl, base_len=THRUST_BASE_LEN)
        drone.draw(frame, cam, color=COLORS_BGR['black'], pt=2)

        if draw_forces:
            for force in forces:
                c = force.color if force.color is not None else COLORS_BGR['red']
                force.draw(frame, cam, color=c, pt=2)
    else:  # Multiple drones
        for i in range(pos.shape[0]):
            drone.translate(pos[i]-drone.pos)
            drone.rotate(ori[i])
            thrust_magnitude_i = (np.asarray(u[i], dtype=float) + 1.0) / 2.0
            set_thrust(drone, forces, thrust_magnitude_i * scl, base_len=THRUST_BASE_LEN)

            # Draw drone with custom color if available
            if 'color' in state and len(state['color']) > i:
                drone.draw(frame, cam, color=state['color'][i], pt=2)
            else:
                drone.draw(frame, cam, color=COLORS_BGR['black'], pt=2)

            if draw_forces:
                for force in forces:
                    c = force.color if force.color is not None else COLORS_BGR['red']
                    force.draw(frame, cam, color=c, pt=2)

def get_drone_state_zero():
    """
    Return default zero drone state for testing and initialization.
    
    Returns:
        dict: Default drone state with position, orientation, and control inputs
    """
    return {
        'x': 0,
        'y': 0,
        'z': 0,
        'phi': 0,
        'theta': 0,
        'psi': 0,
        'u1': 0,
        'u2': 0,
        'u3': 0,
        'u4': 0
    }

def view(propellers,
         get_drone_state=get_drone_state_zero,
         fps=100,
         gate_pos=[],
         gate_yaw=[],
         gate_size=None,
         record_steps=0,
         record_file='output.mp4',
         show_window=True,
         view_type='iso',
         follow=False,
         draw_forces=True,
         draw_path=False,
         auto_play=True,
         record=False,
         motor_colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'],
         overlay_text_position='lower right',
         overlay_text_scale=6.0,
         ):
    """
    Real-time 3D visualization of drone state with interactive controls.
    
    This function creates a real-time 3D viewer that continuously updates
    the drone visualization based on the provided state function. Supports
    interactive camera controls, force visualization, and video recording.
    
    Args:
        propellers: List of propeller dictionaries defining drone configuration
        get_drone_state: Function that returns current drone state dict
        fps: Target frames per second for display/recording
        gate_pos: List of gate positions for course visualization
        gate_yaw: List of gate orientations
        gate_size: Gate size in meters. If None, uses GATE_SIZE constant.
        record_steps: Number of steps to record (0 = no recording)
        record_file: Output video filename
        show_window: Whether to display the window
        view_type: Camera view type ('top', 'iso', 'isometric')
        follow: Whether camera should follow the drone
        draw_forces: Whether to display thrust force vectors
        draw_path: Whether to display trajectory path
        auto_play: Whether animation auto-plays
        record: Whether to start recording immediately
        motor_colors: List of color names for motor visualization
        
    Controls:
        Mouse: Rotate camera view
        P: Toggle auto-play
        F: Toggle follow mode (camera tracks drone)
        S: Toggle force vector display
        D: Toggle path display
        1/2: Zoom in/out
        R: Toggle recording
        ESC: Exit
    """
    num_arms = len(propellers)
    
    # Screen resolution
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT

    # Initialize camera
    cam = create_camera(view_type, width, height)

    # Create scene elements
    big_grid = create_grid(GRID_SIZE, GRID_SIZE, 1)
    
    # Convert motor colors to BGR format for OpenCV
    motor_colors_bgr = [COLORS_BGR.get(color, COLORS_BGR['white']) for color in motor_colors]
    
    drone, forces = create_drone(propellers, box_size=DRONE_BOX_SIZE, prop_radius=PROP_RADIUS, scale=1, motor_colors=motor_colors_bgr)

    # Create gate geometry
    gate, gate_collision_box = create_gate_geometry(gate_size)

    # Visualization parameters
    scl = THRUST_SCALE  # Thrust vector scale
    
    # Control flags (parameters passed in as initial values)
    
    # Target point visualization
    target = create_path(np.array([[0,0,0],[0,0,0.01]]))

    # Video recording setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if '/' in record_file:
        os.makedirs(os.path.dirname(record_file), exist_ok=True)             
    out = cv2.VideoWriter(record_file, fourcc, fps=fps, frameSize=(width, height))
    
    # Auto-recording setup
    steps = 0
    if record_steps > 0:
        record = True

    # Window setup
    if show_window:
        cv2.namedWindow('animation')
        cv2.setMouseCallback('animation', cam.mouse_control)

    # Main visualization loop
    while True:
        # Update step counter
        if auto_play:
            steps += 1
            if 0 < record_steps < steps:
                out.release()
                break
            
            # Get current drone state
            state = get_drone_state()
            prev_state = copy.deepcopy(state)
        else:
            state = prev_state

        # Extract state information
        pos = np.stack([state['x'], state['y'], state['z']]).T
        ori = np.stack([state['phi'], state['theta'], state['psi']]).T
        u = np.stack([state[f'u{i}'] for i in range(1,num_arms+1)]).T
        
        # Update camera
        if follow:
            cam.set_center(drone.pos)
        else:
            cam.set_center(np.zeros(3))

        # Create frame
        frame = 255*np.ones((height, width, 3), dtype=np.uint8)
    
        # Draw grid
        big_grid.draw(frame, cam, color=COLORS_BGR['gray'], pt=1)
        
        # Draw trajectory targets if available
        if 'traj_x' in state:
            target_pos = np.stack([state['traj_x'], state['traj_y'], state['traj_z']]).T
            
            # Draw target trajectory ghost line
            if len(target_pos) > 1 and draw_path:
                target_path = create_path([tp for tp in target_pos[0::PATH_SUBSAMPLE]])  # Subsample for performance
                target_path.draw(frame, cam, color=COLORS_BGR['green'], pt=1)  # Green ghost line
            
            # Draw current target points
            for tp in target_pos:
                target.translate(tp-target.pos)
                target.draw(frame, cam, color=COLORS_BGR['green'], pt=10)

        # Draw drone(s) and forces
        draw_drone_and_forces(drone, forces, pos, ori, u, state, draw_forces, scl, frame, cam)
                        
        # Draw gates
        for pos_gate, yaw in zip(gate_pos, gate_yaw):
            gate.translate(pos_gate-gate.pos)
            gate_collision_box.translate(pos_gate-gate_collision_box.pos)
            gate.rotate([0,0,yaw])
            gate_collision_box.rotate([0,0,yaw])
            gate.draw(frame, cam, color=(0,140,255), pt=4)

        # Gates-passed counter (drawn before the recording write so it appears
        # in the output file).
        if 'gates_passed' in state:
            label = f"{state['gates_passed']}"
            thickness = max(1, int(round(overlay_text_scale * 3)))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          overlay_text_scale, thickness)
            tx, ty = _overlay_anchor(overlay_text_position, tw, th, width, height)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        overlay_text_scale, COLORS_BGR['black'],
                        thickness=thickness)

        # Recording indicator
        if record:
            out.write(frame)
            cv2.putText(frame, '[recording]', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR['black'])

        # Handle keyboard input
        key = cv2.waitKeyEx(1)
        auto_play, follow, draw_forces, draw_path, record, should_exit = handle_keyboard_input_view(
            key, cam, auto_play, follow, draw_forces, draw_path, record, out, record_file)
        
        if should_exit:
            break
        
        # Display frame
        if show_window:
            cv2.imshow('animation', frame)
    
    cv2.destroyAllWindows()

def animate(t, x, y, z, phi, theta, psi, u,
            autopilot_mode=[],
            target=[],
            waypoints=[],
            file='output.mp4',
            multiple_trajectories=False,
            simultaneous=False,
            colors=[],
            names=[],
            alpha=0,
            step=1,
            gate_pos=[],
            gate_yaw=[],
            **kwargs):
    """
    Trajectory playback animation with timeline control and recording.
    
    This function creates an interactive animation player for pre-recorded
    trajectories with timeline scrubbing, path visualization, and recording capabilities.
    
    Args:
        t: Time array or list of time arrays
        x, y, z: Position arrays or lists of position arrays  
        phi, theta, psi: Attitude arrays or lists of attitude arrays
        u: Control input arrays or lists of control arrays
        autopilot_mode: Array indicating autopilot status
        target: Target trajectory points
        waypoints: Waypoint markers to display
        file: Output video filename
        multiple_trajectories: Whether data contains multiple trajectories
        simultaneous: Whether to show multiple trajectories simultaneously
        colors: Colors for multiple trajectories
        names: Names for trajectory labels
        gate_pos, gate_yaw: Gate positions and orientations
        **kwargs: Additional options (follow, auto_play, draw_path, draw_forces, etc.)
        
    Controls:
        Mouse: Rotate camera view
        Timeline slider: Scrub through time
        SPACE: Toggle auto-play
        F: Toggle follow mode
        P: Toggle path display
        S: Toggle force display
        R: Toggle recording
        J/L: Previous/next trajectory (if multiple)
        1/2: Zoom in/out
        ESC: Exit
    """
    # Extract control options
    follow = kwargs.get('follow', False)
    auto_play = kwargs.get('auto_play', False)
    draw_path = kwargs.get('draw_path', False)
    draw_forces = kwargs.get('draw_forces', False)
    record = False
    
    traj_index = 0
    
    # Handle multiple trajectories
    if simultaneous:
        traj_index = np.argmax(np.array([ti[-1] for ti in t]))
    
    if multiple_trajectories:
        t_ = t[traj_index]
        pos = np.stack([x[traj_index],y[traj_index],z[traj_index]]).T
        ori = np.stack([phi[traj_index],theta[traj_index],psi[traj_index]]).T
        u_ = u[traj_index]
    else:
        t_ = t
        pos = np.stack([x,y,z]).T
        ori = np.stack([phi,theta,psi]).T
        u_ = u
    
    # Create default quadrotor for animation if no specific configuration provided
    propellers = kwargs.get('propellers', [
        {"loc": [0.5*np.cos(np.pi/4), 0.5*np.sin(np.pi/4), 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
        {"loc": [0.5*np.cos(3*np.pi/4), 0.5*np.sin(3*np.pi/4), 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
        {"loc": [0.5*np.cos(5*np.pi/4), 0.5*np.sin(5*np.pi/4), 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
        {"loc": [0.5*np.cos(7*np.pi/4), 0.5*np.sin(7*np.pi/4), 0], "dir": [0, 0, -1, "cw"], "propsize": 5}
    ])
    
    # Screen resolution
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT
    
    # Initialize camera (default view for animation)
    cam = create_camera('top', width, height)
    cam.pos = np.array([-5., 0., 0.])
    cam.theta = np.zeros(3)
    cam.r[0] = -12.
    
    # Create scene elements
    big_grid = create_grid(10, 10, 1)
    drone, forces = create_drone(propellers, box_size=DRONE_BOX_SIZE, prop_radius=PROP_RADIUS, scale=1)
    
    # Gate setup
    gate, gate_collision_box = create_gate_geometry(kwargs.get('gate_size'))

    scl = THRUST_SCALE  # Thrust scale

    # Window setup
    cv2.namedWindow('animation')
    cv2.setMouseCallback('animation', cam.mouse_control)
    cv2.createTrackbar('t', 'animation', 0, t_.shape[0]-1, nothing)
    
    # Create path visualizations
    paths = []
    if simultaneous:
        for i in range(len(t)):
            p = np.stack([x[i],y[i],z[i]]).T
            paths.append(create_path([pi for pi in p[0::PATH_SUBSAMPLE]]))
    
    path = create_path([p for p in pos[0::PATH_SUBSAMPLE]])
    
    # Create waypoint markers
    waypoints = [create_path([v,v+[0,0,0.01]]) for v in waypoints]
    
    # Animation state
    start_time = time.time()
    time_index = 0
    video_step = 1
    
    # Main animation loop
    while True:
        
        # Update time index
        if auto_play:
            if record:
                if time_index < len(t_) - video_step:
                    time_index += video_step
            else:
                current_time = time.time() - start_time
                for i in range(len(t_)):
                    if t_[i] > current_time:
                        time_index = i
                        break
                if time_index == -1:
                    current_time = t_[time_index]
        else:
            time_index = cv2.getTrackbarPos('t', 'animation')
            current_time = t_[time_index]

        # Check for recording time limit
        if 'record_time' in kwargs:
            if t_[time_index] >= kwargs['record_time']:
                if record:
                    print('recording ended')
                    out.release()
                    print('recording saved in ' + file)
                break
        
        # Update drone state
        drone.translate(pos[time_index] - drone.pos)
        drone.rotate(ori[time_index])

        T = u_[time_index]  # Thrust values
        set_thrust(drone, forces, T*scl)

        # Update camera
        if follow:
            cam.set_center(drone.pos)
        else:
            cam.set_center(np.zeros(3))

        # Create frame
        frame = 255*np.ones((height, width, 3), dtype=np.uint8)

        # Draw scene elements
        big_grid.draw(frame, cam, color=COLORS_BGR['gray'], pt=1)
        
        # Draw waypoints
        for w in waypoints:
            w.draw(frame, cam, color=COLORS_BGR['red'], pt=4)
            
        # Draw target trajectory ghost line
        if len(target) > 1:
            target_path = create_path([t for t in target[0::PATH_SUBSAMPLE]])  # Subsample for performance
            target_path.draw(frame, cam, color=COLORS_BGR['green'], pt=1)  # Green ghost line
        
        # Draw current target
        if time_index < len(target):
            tt = target[time_index]
            tt_graphic = create_path([tt,tt+[0,0,0.01]])
            tt_graphic.draw(frame, cam, color=COLORS_BGR['green'], pt=10)

        # Draw trajectory paths
        if draw_path and not simultaneous:
            path.draw(frame, cam, color=COLORS_BGR['green'], pt=2)
            
        # Handle simultaneous trajectories
        if simultaneous:
            if draw_path:
                for i in range(len(t)):
                    color = colors[i] if len(colors) > i else COLORS_BGR['green']
                    paths[i].draw(frame, cam, color=color, pt=1)
                    
            for i in range(len(t)):
                pos_i = np.stack([x[i],y[i],z[i]]).T
                ori_i = np.stack([phi[i],theta[i],psi[i]]).T
                u_i = u[i]
                time_index_i = 0
                for j in range(len(t[i])):
                    time_index_i = j
                    if t[i][j] > t_[time_index]:
                        break
                        
                drone.translate(pos_i[time_index_i] - drone.pos)
                drone.rotate(ori_i[time_index_i])
                T_i = u_i[time_index_i]
                set_thrust(drone, forces, T_i*scl)
                
                color = colors[i] if len(colors) > i else COLORS_BGR['blue']
                drone.draw(frame, cam, color=color, pt=2)
                
                if draw_forces:
                    for force in forces:
                        force.draw(frame, cam, pt=2)
        else:
            # Single trajectory rendering
            if multiple_trajectories and len(colors) > traj_index:
                drone.draw(frame, cam, color=colors[traj_index], pt=2)
            elif pos[time_index][2] > 0:
                drone.draw(frame, cam, color=COLORS_BGR['red'], pt=2)
            elif len(autopilot_mode) > 0:
                if autopilot_mode[time_index] == 0 or True:
                    drone.draw(frame, cam, color=COLORS_BGR['blue'], pt=2)
                else:
                    drone.draw(frame, cam, color=COLORS_BGR['green'], pt=2)
                    cv2.putText(frame, '[gcnet active]', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR['green'])                
            else:
                drone.draw(frame, cam, color=COLORS_BGR['blue'], pt=2)
                
            if draw_forces and not simultaneous:
                for force in forces:
                    force.draw(frame, cam, pt=2)

        # Draw gates
        for gpos, gyaw in zip(gate_pos, gate_yaw):
            gate.translate(gpos-gate.pos)
            gate_collision_box.translate(gpos-gate_collision_box.pos)
            gate.rotate([0,0,gyaw])
            gate_collision_box.rotate([0,0,gyaw])
            gate.draw(frame, cam, color=(0,140,255), pt=4)
        
        # Add text overlays
        cv2.putText(frame, "t = " + str(round(t_[time_index], 2)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR['black'])
        
        if 'title' in kwargs:
            cv2.putText(frame, kwargs['title'], (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR['black'], thickness=2)
        
        for i in range(len(names)):
            color = colors[i] if len(colors) > i else COLORS_BGR['black']
            cv2.putText(frame, names[i], (700, 20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
            
        if record:
            out.write(frame)
            cv2.putText(frame, '[recording]', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR['black'])

        # Handle keyboard input
        control = cv2.waitKeyEx(1)
        
        # Trajectory switching (J/L keys)
        if control == 106 and multiple_trajectories:  # J - previous trajectory
            time_index = 0
            start_time = time.time() - t_[time_index]
            traj_index = max(0, traj_index-step)
            t_ = t[traj_index]
            pos = np.stack([x[traj_index],y[traj_index],z[traj_index]]).T
            ori = np.stack([phi[traj_index],theta[traj_index],psi[traj_index]]).T
            u_ = u[traj_index]
            path = create_path([p for p in pos[0::5]])
        if control == 108 and multiple_trajectories:  # L - next trajectory
            time_index = 0
            start_time = time.time() - t_[time_index]
            traj_index = min(len(t)-1, traj_index+step)
            t_ = t[traj_index]
            pos = np.stack([x[traj_index],y[traj_index],z[traj_index]]).T
            ori = np.stack([phi[traj_index],theta[traj_index],psi[traj_index]]).T
            u_ = u[traj_index]
            path = create_path([p for p in pos[0::5]])
            
        # Auto-record trigger
        if 'record' in kwargs:
            if kwargs['record'] and not record:
                control = 114  # Trigger record
                
        if control == 114:  # R - toggle recording
            if record:
                print('recording ended')
                out.release()
                print('recording saved in ' + file)
            else:
                print('recording started')
                # Setup video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # Calculate appropriate framerate (max 100 fps)
                dt = np.mean(t_[1:]-t_[:-1])
                fps = 1/dt
                while fps > 100:
                    video_step += 1
                    fps = 1/(dt*video_step)
                fps = int(fps)
                print(f"Recording at {fps} fps")                    
                out = cv2.VideoWriter(file, fourcc, fps=fps, frameSize=(width, height))
            record = not record
        elif control == 102:  # F - follow mode
            follow = not follow
        elif control == 112:  # P - path display
            draw_path = not draw_path
        elif control == 115:  # S - force display
            draw_forces = not draw_forces
        elif control == 32:   # SPACE - auto-play
            auto_play = not auto_play
            start_time = time.time() - t_[time_index]
        elif control == 49:   # 1 - zoom in
            cam.zoom(ZOOM_FACTOR)
        elif control == 50:   # 2 - zoom out
            cam.zoom(1/ZOOM_FACTOR)
        elif control == 27:   # ESC - exit
            break
        
        cv2.imshow('animation', frame)
    
    cv2.destroyAllWindows()