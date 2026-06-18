"""
3D Graphics Engine for Drone Visualization

This module provides the core 3D graphics infrastructure including camera controls,
mesh representation, and force visualization for drone simulation.

Classes:
    Camera: 3D camera with projection and mouse controls
    Mesh: 3D mesh representation with drawing capabilities  
    Force: Force vector visualization

Global Variables:
    dragging: Mouse drag state for camera control
    xi, yi: Mouse position tracking
"""

import numpy as np
import cv2

# Global variables for mouse interaction
dragging = False
xi, yi = -1, -1

def rotation_matrix(theta):
    """
    Calculate rotation matrix given euler angles.
    
    Args:
        theta: Array of [roll, pitch, yaw] angles in radians
        
    Returns:
        3x3 rotation matrix
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

class Camera:
    """
    3D camera with projection and interactive mouse controls.
    
    Provides 3D to 2D projection, mouse-based rotation and zoom,
    and camera positioning relative to a center point.
    """
    
    def __init__(self, pos, theta, cameraMatrix, distCoeffs):
        """
        Initialize camera.
        
        Args:
            pos: Initial camera position in world frame
            theta: Initial Euler angles [roll, pitch, yaw]
            cameraMatrix: OpenCV camera intrinsic matrix
            distCoeffs: OpenCV distortion coefficients
        """
        # pose
        self.pos = pos                          # wrt world frame
        self.theta = theta                      # Euler angles: roll pitch yaw
        self.rMat = rotation_matrix(theta)

        self.center = np.zeros(3)               # camera rotates around center
        self.r = np.array([-8., 0., 0.])

        # intrinsic camera parameters
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def set_center(self, vector):
        """Set the center point that the camera rotates around."""
        self.center = vector
        self.pos = np.dot(self.rMat, self.r) + self.center

    def rotate(self, theta):
        """
        Rotate camera by given angles.
        
        Args:
            theta: [roll, pitch, yaw] rotation increments
        """
        self.theta += theta
        self.rMat = rotation_matrix(self.theta)
        self.pos = np.dot(self.rMat, self.r) + self.center
        
    def zoom(self, scl):
        """
        Zoom camera by scaling distance to center.
        
        Args:
            scl: Scale factor (>1 zoom out, <1 zoom in)
        """
        self.r *= scl
        self.pos = self.rMat @ self.r + self.center

    def project(self, points):
        """
        Project 3D points to 2D camera image.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            tuple: (projected_points, in_frame_mask)
                projected_points: Nx2 array of 2D pixel coordinates
                in_frame_mask: Boolean array indicating which points are visible
        """
        # points in frame (in front of the camera) given by a boolean array
        in_frame = np.dot(points - self.pos, self.rMat[:, 0]) > 0.01

        # x-axis is used as projection axis
        M = np.dot(self.rMat, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

        tvec = -np.dot(np.transpose(M), self.pos)
        rvec = cv2.Rodrigues(np.transpose(M))[0]

        projected_points = cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)
        return projected_points, in_frame

    def mouse_control(self, event, x, y, flags, params):
        """
        Handle mouse events for interactive camera control.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse coordinates
            flags: Mouse event flags
            params: Additional parameters (unused)
        """
        global xi, yi, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            xi, yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                yaw = 2*np.pi * (x - xi) / 1536
                pitch = -np.pi * (y - yi) / 864
                self.rotate([0, pitch, yaw])
                xi, yi = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

class Mesh:
    """
    3D mesh representation with drawing capabilities.
    
    Represents a 3D object as vertices and edges that can be
    transformed and rendered to a 2D image.
    """
    
    def __init__(self, vertices, edges):
        """
        Initialize mesh.
        
        Args:
            vertices: Nx3 array of 3D vertex positions
            edges: Mx2 array of edge indices connecting vertices
        """
        self.vertices = vertices
        self.edges = edges
        self.pos = np.array([0., 0., 0.])
        self.theta = np.array([0., 0., 0.])
        self.edge_colors = None

    def draw(self, img, cam, color=(100, 100, 100), pt=1, arrow=False):
        """
        Draw mesh to image using camera projection.

        Args:
            img: Image to draw on
            cam: Camera object for projection
            color: RGB color tuple
            pt: Line thickness
            arrow: Whether to draw as arrows
        """
        pvertices, in_frame = cam.project(self.vertices)
        edge_colors = self.edge_colors
        for i, edge in enumerate(self.edges):
            if in_frame[edge[0]] and in_frame[edge[1]]:
                pt1 = tuple(pvertices[edge[0]][0])
                pt2 = tuple(pvertices[edge[1]][0])
                c = color
                if edge_colors is not None and edge_colors[i] is not None:
                    c = edge_colors[i]
                if arrow:
                    cv2.arrowedLine(img, pt1, pt2, c, pt)
                else:
                    cv2.line(img, pt1, pt2, c, pt)

    def translate(self, vector):
        """
        Translate mesh by given vector.
        
        Args:
            vector: 3D translation vector
        """
        self.pos += vector
        for vertex in self.vertices:
            vertex += vector

    def rotate(self, theta):
        """
        Rotate mesh to given orientation.
        
        Args:
            theta: Target Euler angles [roll, pitch, yaw]
        """
        M1 = np.transpose(rotation_matrix(self.theta))
        M2 = rotation_matrix(theta)
        R = np.dot(M2, M1)
        for vertex in self.vertices:
            delta = self.pos + np.dot(R, vertex - self.pos) - vertex
            vertex += delta
        self.theta = theta

class Force:
    """
    Force vector visualization.
    
    Represents and visualizes force vectors as arrows in 3D space.
    """
    
    def __init__(self, vertex):
        """
        Initialize force at given position.

        Args:
            vertex: 3D position where force is applied
        """
        self.vertex = vertex
        self.F = np.array([0., 0., 0.])
        self.body_dir = None  # Unit thrust direction in drone body frame
        self.color = None     # BGR tuple, overrides caller's default draw color

    def draw(self, img, cam, color=(0, 0, 255), pt=1):
        """
        Draw force vector as arrow.
        
        Args:
            img: Image to draw on
            cam: Camera object for projection
            color: RGB color tuple
            pt: Line thickness
        """
        pt1, in_frame1 = cam.project(np.array([self.vertex]))
        pt2, in_frame2 = cam.project(np.array([self.vertex + self.F]))
        pt1 = tuple(pt1[0][0])
        pt2 = tuple(pt2[0][0])
        # check if pt1 or pt2 are in frame
        if in_frame1[0] and in_frame2[0]:
            cv2.arrowedLine(img, pt1, pt2, color, pt)