import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class CupModule(Module):
    
    ATTACH = 0

    def __init__(self, name="cup", radius=0.05, height=0.08, ball_radius=0.02, ball_mass=0.01):
        self.name = name
        self.radius = radius
        self.height = height
        self.cup_thickness = 0.005
        self.ball_radius = ball_radius
        self.ball_mass = ball_mass
        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                # Attach by the wall: offset by radius + thickness in X
                offset=Vector3([self.radius + self.cup_thickness, 0.0, self.height / 2]),
                orientation=Quaternion([0,0,0,1]),
            ),
        }
        super().__init__(attachment_points)
    
    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat) -> list:
        body_pos = attachment_point_pos + Vector3([-self.height / 2, 0.0, self.radius + self.cup_thickness])
        pos_list = ensure_list(body_pos)
        mj_quat = attachment_point_quat.to_mujoco_format()
        
        # Create cup body
        # Rotate the cup body 90 degrees around Y axis so it's horizontal
        # Original orientation is Z-up. We want it to be X-out (or similar)
        # Let's rotate 90 deg around Y: [0, 0.7071, 0, 0.7071]
        q_rot = Quaternion.from_axis_angle(Vector3([0, 1, 0]), np.pi/2)
        final_quat = attachment_point_quat * q_rot
        
        cup_body = entry_point.add("body", name=self.name, pos=pos_list, quat=final_quat.to_mujoco_format())
        
        # Cup base
        cup_body.add("geom", name="cup_base", type="box", size=[self.radius, self.radius, self.cup_thickness], 
                     pos=[0, 0, self.cup_thickness], rgba=[0.1, 0.1, 0.8, 1], mass=0.1, contype=1, conaffinity=1)
        
        # Cup walls
        h2 = self.height / 2
        cup_body.add("geom", name="cup_wall1", type="box", size=[self.radius, self.cup_thickness, h2], 
                     pos=[0, self.radius, h2], rgba=[0.1, 0.1, 0.8, 0.5], mass=0.05, contype=1, conaffinity=1)
        cup_body.add("geom", name="cup_wall2", type="box", size=[self.radius, self.cup_thickness, h2], 
                     pos=[0, -self.radius, h2], rgba=[0.1, 0.1, 0.8, 0.5], mass=0.05, contype=1, conaffinity=1)
        cup_body.add("geom", name="cup_wall3", type="box", size=[self.cup_thickness, self.radius, h2], 
                     pos=[self.radius, 0, h2], rgba=[0.1, 0.1, 0.8, 0.5], mass=0.05, contype=1, conaffinity=1)
        cup_body.add("geom", name="cup_wall4", type="box", size=[self.cup_thickness, self.radius, h2], 
                     pos=[-self.radius, 0, h2], rgba=[0.1, 0.1, 0.8, 0.5], mass=0.05, contype=1, conaffinity=1)
        
        cup_body.add("site", name="cup_site", pos=[0, 0, h2], size=[0.01], rgba=[0, 1, 0, 1])

        # String and Ball - implemented as a chain of bodies with ball joints
        num_segments = 20
        segment_length = 0.015  # Shorter segments for smoother curve
        string_radius = 0.001
        
        # String connects to the cup base (which is at z=0 in local coords)
        # Build a chain of bodies manually for the string
        prev_body = cup_body
        
        # Wave/Spiral parameters
        amplitude = 0.005
        frequency = 1.5
        
        for i in range(num_segments):
            # Calculate curved offset for each segment to create a wave/spiral shape
            # This helps the string maintain a more stable, non-linear rest pose
            dx = amplitude * np.sin(frequency * i)
            dy = amplitude * np.cos(frequency * i)
            dz = -segment_length
            
            if i == 0:
                seg_pos = [dx, dy, -0.005 + dz]
            else:
                seg_pos = [dx, dy, dz]
                
            seg_body = prev_body.add("body", name=f"string_seg_{i}", pos=seg_pos)
            # Increased damping and added small stiffness to stabilize the string
            # The curved structure combined with stiffness makes it more like a spring/rope
            seg_body.add("joint", name=f"string_joint_{i}", type="ball", damping=0.01, stiffness=0.005)
            
            # Visual capsule connecting back to the previous body
            # Since seg_body is at [dx, dy, dz] relative to prev_body, 
            # the vector from seg_body to prev_body is [-dx, -dy, -dz]
            seg_body.add("geom", name=f"string_vis_{i}", type="capsule",
                         fromto=[0, 0, 0, -dx, -dy, -dz],
                         size=[string_radius], rgba=[0.8, 0.8, 0.8, 1],
                         contype=0, conaffinity=0, group=1, mass=0)
            
            # Small sphere for collision and mass
            seg_body.add("geom", name=f"string_geom_{i}", type="sphere",
                         size=[string_radius], rgba=[0.8, 0.8, 0.8, 1],
                         contype=2, conaffinity=1, mass=0.002)
            prev_body = seg_body
        
        # The last segment body is the attachment point for the ball
        ball = prev_body.add("body", name="ball", pos=[0, 0, 0])
        # Increased damping for the ball joint as well
        ball.add("joint", name="ball_joint", type="ball", damping=0.1, stiffness=0.01)
        ball.add("geom", name="ball_geom", type="sphere", size=[self.ball_radius], mass=self.ball_mass, 
                 rgba=[0.8, 0.1, 0.1, 1], friction=[1, 0.005, 0.001], contype=1, conaffinity=1)
        ball.add("site", name="ball_site", pos=[0, 0, 0], size=[0.01], rgba=[1, 0, 0, 0])

        return []
