# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import os
from datetime import datetime
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

from . import rotation_conversion, quaternion_functions

numFrames = 8

def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave, orient="NED", gate_pos=None, gate_yaw=None, gate_size=1.0, bspline_traj=None, save_path=None):

    x = pos_all[:,0]
    y = pos_all[:,1]
    z = pos_all[:,2]

    xDes = sDes_tr_all[:,0]
    yDes = sDes_tr_all[:,1]
    zDes = sDes_tr_all[:,2]

    x_wp = waypoints[:,0]
    y_wp = waypoints[:,1]
    z_wp = waypoints[:,2]

    if (orient == "NED"):
        z = -z
        zDes = -zDes
        z_wp = -z_wp

    fig = plt.figure()
    ax = p3.Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    line1, = ax.plot([], [], [], lw=2, color='red')
    line2, = ax.plot([], [], [], lw=2, color='blue')
    line3, = ax.plot([], [], [], '--', lw=1, color='blue')

    # Setting the axes properties
    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    mid_z = 0.5*(z.max()+z.min())
    
    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    if (orient == "NED"):
        ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
    elif (orient == "ENU"):
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    # Only B-spline gate trajectories are supported (xyzType == 15)
    if (xyzType == 15):
        trajType = 'B-spline Gate Trajectory'
        # Plot desired trajectory path
        ax.plot(xDes, yDes, zDes, ':', lw=1.3, color='green')
    else:
        # Legacy trajectory types are no longer supported
        trajType = f'Unsupported (xyzType={xyzType})'

    # Draw gates if provided
    if gate_pos is not None and gate_yaw is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for gate_idx, (gpos, gyaw) in enumerate(zip(gate_pos, gate_yaw)):
            # Create gate corners as a vertical square (standing up)
            # In the gate's local frame, the gate is perpendicular to the forward direction
            half_size = gate_size / 2.0

            # Gate corners in local frame: gate is vertical, perpendicular to X-axis
            # Y-axis goes left-right, Z-axis goes up-down
            local_corners = np.array([
                [0, -half_size, -half_size],  # Bottom left
                [0,  half_size, -half_size],  # Bottom right
                [0,  half_size,  half_size],  # Top right
                [0, -half_size,  half_size]   # Top left
            ])

            # Rotation matrix for yaw (rotation around Z-axis in NED)
            cos_yaw = np.cos(gyaw)
            sin_yaw = np.sin(gyaw)

            # Rotation matrix (Z-axis rotation)
            R = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [0,        0,       1]
            ])

            # Transform corners to world frame
            gate_corners_3d = []
            for corner in local_corners:
                # Rotate corner
                rotated = R @ corner

                # Translate to gate position
                world_pos = rotated + gpos

                # Adjust for NED orientation if needed (flip Z)
                if orient == "NED":
                    gate_corners_3d.append([world_pos[0], world_pos[1], -world_pos[2]])
                else:
                    gate_corners_3d.append([world_pos[0], world_pos[1], world_pos[2]])

            # Draw gate as a polygon
            gate_poly = Poly3DCollection([gate_corners_3d], alpha=0.3, facecolor='orange', edgecolor='darkorange', linewidths=2)
            ax.add_collection3d(gate_poly)

            # Draw gate center marker with order number
            if orient == "NED":
                gate_z_pos = -gpos[2]
            else:
                gate_z_pos = gpos[2]

            ax.scatter([gpos[0]], [gpos[1]], [gate_z_pos], color='darkorange', marker='o', s=100, edgecolors='black', linewidths=1.5, zorder=6)

            # Add order number on the gate (1-indexed)
            gate_num = gate_idx + 1
            ax.text(gpos[0], gpos[1], gate_z_pos, f'{gate_num}', fontsize=12,
                   color='white', weight='bold', ha='center', va='center', zorder=7,
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='darkorange',
                            edgecolor='black', linewidth=2))

    # Draw B-spline control points if provided
    if bspline_traj is not None:
        # Get all control points (all are gate control points)
        gate_cps = bspline_traj.get_all_control_points()

        # Visualize gate control points
        gate_x = gate_cps[:, 0]
        gate_y = gate_cps[:, 1]
        gate_z = gate_cps[:, 2]

        if orient == "NED":
            gate_z = -gate_z

        # Draw gate control points
        ax.scatter(gate_x, gate_y, gate_z, color='purple', marker='o', s=100,
                   alpha=0.7, label='Gate Control Points', edgecolors='black', linewidths=1.5, zorder=5)

        # Draw lines connecting gate control points (periodic - close the loop back to first gate)
        gate_x_closed = np.append(gate_x, gate_x[0])
        gate_y_closed = np.append(gate_y, gate_y[0])
        gate_z_closed = np.append(gate_z, gate_z[0])
        ax.plot(gate_x_closed, gate_y_closed, gate_z_closed, color='purple', linestyle='--',
                linewidth=1.5, alpha=0.4, zorder=4)

        # Add legend with better positioning to avoid overlap
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95), fontsize=8,
                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'



    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   
    
    def updateLines(i):

        time = t_all[i*numFrames]
        pos = pos_all[i*numFrames]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_from0 = pos_all[0:i*numFrames,0]
        y_from0 = pos_all[0:i*numFrames,1]
        z_from0 = pos_all[0:i*numFrames,2]
    
        dxm = params["dxm"]
        dym = params["dym"]
        dzm = params["dzm"]
        
        quat = quat_all[i*numFrames]
    
        if (orient == "NED"):
            z = -z
            z_from0 = -z_from0
            quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])
    
        R = rotation_conversion.quat2Dcm(quat)    
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z 
        
        line1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        line1.set_3d_properties(motorPoints[2,0:3])
        line2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        line2.set_3d_properties(motorPoints[2,3:6])
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time))
        
        return line1, line2


    def ini_plot():

        line1.set_data(np.empty([1]), np.empty([1]))
        line1.set_3d_properties(np.empty([1]))
        line2.set_data(np.empty([1]), np.empty([1]))
        line2.set_3d_properties(np.empty([1]))
        line3.set_data(np.empty([1]), np.empty([1]))
        line3.set_3d_properties(np.empty([1]))

        return line1, line2, line3

        
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    if (ifsave):
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = '__data__/animations/animation_{0:.0f}_{1:.0f}_{2}.mp4'.format(
                xyzType, yawType, timestamp)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        line_ani.save(save_path, dpi=80, writer='ffmpeg', fps=25)
        print(f"Animation saved to: {save_path}")

    return line_ani