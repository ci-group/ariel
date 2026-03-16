

def mujoco_setup_sim(env_mjcf, ts=0.01, gravity=[0, 0, -9.81]):
    env_mjcf.compiler.angle = "radian"
    env_mjcf.compiler.inertiafromgeom = True
    env_mjcf.option.timestep = ts
    env_mjcf.option.integrator = "implicitfast"
    env_mjcf.option.solver = "CG"
    env_mjcf.option.gravity = gravity


    env_mjcf.asset.add("texture", builtin="gradient", type="skybox", rgb1=[0.7,0.7,0.7], rgb2=[0.5,0.5,0.5], width=512, height=512)
    env_mjcf.asset.add("texture", name="texgeom", builtin="flat", rgb1=[0.8,0.6,0.4], rgb2=[0.8,0.6,0.4], width=127, height=1278, type="cube", random="0.01", mark="cross", markrgb=[1, 1, 1])
    env_mjcf.asset.add("texture", name="texplane", builtin="checker", height=100, rgb1=[0.137,0.00,0.275], rgb2=[0.2,0.0,0.39], type="2d", width=100)
    env_mjcf.asset.add("material", name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[60, 60], texture="texplane")
    env_mjcf.asset.add("material", name="geom", texture="texgeom", texuniform=True)

    return env_mjcf