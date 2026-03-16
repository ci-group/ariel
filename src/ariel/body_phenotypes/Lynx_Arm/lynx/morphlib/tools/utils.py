import ast, re
import numpy as np

import time
import mujoco
from mujoco.viewer import launch_passive

def find_and_remove(data, start, end):
    start_idx = data.find(start)
    end_idx = data.find(end)

    if end_idx == -1 or start_idx == -1:
        return 'None', 'None'

    variable_found = data[data.find(start)+len(start):end_idx]
    data = data[end_idx+1:]

    return data, variable_found

def find_and_remove_jinja_variable(data):
    data, key = find_and_remove(data, '{%set', '=')
    data, value = find_and_remove(data, 'default(', ')%')
    
    if value == 'false':
        value = 'False'
    elif value == 'true':
        value = 'True'
    value_eval = ast.literal_eval(value)
    if value_eval == 'inf' or value_eval == '-inf': 
        value_eval = float(value_eval)

    return data, key, value_eval

def get_default_values(dir_file:str) -> dict:
    '''
    :param dir_file: the file to get the variables from
    :returns: dictionary of variable names and values in a file specified by the jinja syntax
    :rtype: dict
    '''
    dict = {}
    with open(dir_file, 'r') as file:
        data = file.read().replace('\n', '').replace(' ', '')

    data, key, value = find_and_remove_jinja_variable(data)
    while value != None and key != 'None':
        dict[key] = value
        data, key, value = find_and_remove_jinja_variable(data)

    return dict

# print(get_default_values('/home/Jed/Documents/uniwork/thesis/gym-adapt/gym_adapt/core/test.xml'))

def dict2nparray(dict):
    length = len(dict)
    keys = np.empty(length, dtype=object)
    values = np.empty(length, dtype=object)

    for i, (k, v) in enumerate(dict.items()):
        keys[i] = k
        values[i] = v

    return keys, values 

def nparray2dict(keys, values):
    dict = {}
    for idx in range(len(keys)):
        dict[keys[idx]] = values[idx]
    return dict

def simulate_XML_mujoco(XML : str, duration : float = 10.0, ASSETS : dict = None):
    '''Simulate a mujoco XML string for a given duration. Taken from mujoco python binding example.'''

    m = mujoco.MjModel.from_xml_string(XML, ASSETS)
    d = mujoco.MjData(m)

    with launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            # mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # close the viewer
        print('Closing viewer...')
        viewer.close()
        print('Viewer closed.')