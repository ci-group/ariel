

def plane(env_mjcf):
    '''Builds the terrain for the simulation. Includes a floor plane and a light source.'''
    env_mjcf.worldbody.add(
        "light",
        cutoff=100,
        diffuse=[1,1,1],
        dir=[0,0,-1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
        castshadow=False,
    )

    env_mjcf.worldbody.add("geom", friction=[0.7, 0.1, 0.1], conaffinity=1, condim=3, name="floor", pos=[0,0,0], rgba=[0.8, 0.9, 0.8, 1], size=[40,40,40], type="plane", material="MatPlane")
