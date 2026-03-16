from typing import Callable
from collections import deque
from dm_control import mjcf
from typing import Dict
import os 
from .math_utils import Quaternion

from .unbuilt_child import UnbuiltChild
from .module import Module

from .mj_default_sim_setup import mujoco_setup_sim
from ..scenes.plane import plane

def build_modular_robot(env_mjcf : mjcf.Element, body : Module, body_pos : list, body_ori : Quaternion):

    # Build Modular Robot
    root = UnbuiltChild(module=body, mj_body=env_mjcf.worldbody)
    root.make_pose(position=body_pos, orientation=body_ori)

    queue = deque([root])
    while len(queue) > 0:
        unbuilt = queue.popleft()
        new_tasks = unbuilt.module.build(env_mjcf, entry_point=unbuilt.mj_body, attachment_point_pos=unbuilt.position, attachment_point_quat=unbuilt.orientation)
        queue.extend(new_tasks)

def build_mjcf(bodies : list, body_poss : list, body_oris : list, terrain_builder : Callable, sim_setup : Callable, ts=0.01):
    # Initialize model
    env_mjcf = mjcf.RootElement(model="scene")

    # Setup Model
    sim_setup(env_mjcf, ts=ts)
    
    # Build Terrain
    terrain_builder(env_mjcf)

    # Build Modular Robots
    for body, body_pos, body_ori in zip(bodies, body_poss, body_oris):
    
        build_modular_robot(env_mjcf, body, body_pos, body_ori)
    

    xml = env_mjcf.to_xml_string()

    return xml

def build_single_robot_mjcf(body, body_ori):
    # Initialize model
    env_mjcf = mjcf.RootElement(model="scene")

    # Build Modular Robots
    build_modular_robot(env_mjcf, body, body_pos=[0,0,0], body_ori=body_ori)

    return env_mjcf

def make_xml_file(make_body, morphological_parameters : Dict = {}, terrain_builder = plane, sim_setup = mujoco_setup_sim, ts=0.01):

    body = make_body(**morphological_parameters)

    xml = build_mjcf([body], [[0,0,0]], [Quaternion()], terrain_builder=terrain_builder, sim_setup=sim_setup, ts=ts)
    model_path = "./assets/ant.xml"
    if not(os.path.isdir("./assets")):
        os.mkdir("./assets")
    with open(model_path, "w") as text_file:
        text_file.write(xml)

    return model_path

class URDF_STR():
    def __init__(self):
        self.urdf_str = """<?xml version="1.0"?>\n<robot name="multirotor">\n"""

    def __str__(self) -> str:
        return self.urdf_str
    
    def add(self, string):
        self.urdf_str = self.urdf_str + string

def build_urdf_str(body):
    urdf_str = URDF_STR()
    # Build Modular Robot
    root = UnbuiltChild(module=body, mj_body=urdf_str)
    root.make_pose(position=[0,0,0], orientation=Quaternion)

    queue = deque([root])
    while len(queue) > 0:
        unbuilt = queue.popleft()
        new_tasks = unbuilt.module.build(None, entry_point=unbuilt.mj_body, attachment_point_pos=None, attachment_point_quat=None)
        queue.extend(new_tasks)

    urdf_str.add("\n</robot>")
    return urdf_str