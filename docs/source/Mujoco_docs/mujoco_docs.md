# MuJoCo Data Class Variables Documentation
This page holds the general information about some Mujoco variables used in ARIEL. This info is, in general, the same as what can be found on the [Mujoco documentation](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#tydataenums). But some things have been added for better understanding.  

## Matrix-Related Variables
- `bind`: Class variable for binding operations. can be used to get data in each step of the simulation. Example: 
```python
geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
```
<!--- `B_colind`, `B_rowadr`, `B_rownnz`: Sparse matrix indexing for constraint Jacobian
- `D_colind`, `D_rowadr`, `D_rownnz`, `D_diag`: Sparse matrix indexing for constraint motion space-->
- `M`: Inertia matrix
<!--- `M_colind`, `M_rowadr`, `M_rownnz`: Sparse inertia matrix indexing-->

## Actuator States
- `act`: Actuator activation states
- `act_dot`: Time-derivative of actuator activations
- `actuator_force`: Forces applied by actuators
- `actuator_length`: Current length of actuators
<!--- `actuator_moment`: Moments applied by actuators-->
- `actuator_velocity`: Velocities of actuators

## Spatial Properties
<!--- `bvh_aabb_dyn`: Bounding volume hierarchy dynamic AABB
- `bvh_active`: Active flags for bounding volume hierarchy nodes
- `cacc`: Center acceleration-->
- `cam_xmat`, `cam_xpos`: Camera orientation matrix and position
<!--- `cdof`: Constraint degrees of freedom
- `cdof_dot`: Time-derivative of constraint DOFs-->
- `cfrc_ext`, `cfrc_int`: External and internal contact forces
<!--- `cinert`: Composite inertia tensors
- `crb`: Composite rigid body inertias-->
- `ctrl`: Control signals
<!--- `cvel`: Center velocities-->

## Energy and State
- `energy`: Potential and kinetic energy
<!--- `eq_active`: Active equality constraints-->

<!-- ## Flex Elements -->
<!--- `flexedge_J`, `flexedge_J_colind`, `flexedge_J_rowadr`, `flexedge_J_rownnz`: Flex edge Jacobian and indexing
- `flexedge_length`, `flexedge_velocity`: Flex edge lengths and velocities
- `flexelem_aabb`: Flex element axis-aligned bounding boxes
- `flexvert_xpos`: Flex vertex positions-->

## Geometric Properties
- `geom_xmat`, `geom_xpos`: Geometry orientation matrices and positions
<!--- `light_xdir`, `light_xpos`: Light direction and position-->

## Mapping Arrays
<!--- `mapD2M`, `mapM2D`, `mapM2M`: Various mapping arrays between different representations-->

<!-- ## System Statistics -->
<!--- `maxuse_arena`: Maximum arena memory usage
- `maxuse_con`: Maximum number of contacts
- `maxuse_efc`: Maximum number of constraint rows
- `maxuse_stack`: Maximum stack size
- `maxuse_threadstack`: Maximum thread stack sizes-->

## Motion Capture
- `mocap_pos`, `mocap_quat`: Motion capture positions and quaternions

<!-- ## System Dimensions -->
<!--- `nA`, `nJ`: Number of rows in constraint matrices
- `narena`, `nbuffer`: Memory arena and buffer sizes
- `ncon`, `nefc`, `nf`: Counts of contacts, constraints, and frames
- `nidof`, `nisland`, `nl`: Various system dimension counts
- `nplugin`: Number of plugins-->

<!-- ## Plugin Management -->
<!--- `plugin`, `plugin_data`, `plugin_state`: Plugin-related data arrays-->

## State Variables
- **`qpos`**: Generalized position state vector.  
  Contains the positions of the core and all hinge joints, in the order they are defined in the XML.  
  - **Core (free joint)**:   
    - `[x, y, z]` : 3D position of the core body frame  
    - `[qw, qx, qy, qz]` : Orientation quaternion of the core  
  - **Hinge joints (bricks)**: `[angle]` for each hinge, representing the rotation angle in radians  
  
  Actuator inputs and states are *not* stored here (see `data.ctrl` and `data.act` instead).

- `qvel`: Velocity state vector. Holds the current velocity of each actuator. 
- `qacc`: Acceleration state vector
<!--- `qDeriv`: Position derivatives
- `qH`, `qHDiagInv`: Matrices for numerical optimization
- `qLD`, `qLDiagInv`, `qLU`: Decomposition matrices
- `qM`: Mass matrix-->

## Forces
- `qfrc_actuator`: Actuator forces
- `qfrc_applied`: Externally applied forces
- `qfrc_bias`: Bias forces
- `qfrc_constraint`: Constraint forces
<!--- `qfrc_damper`: Damping forces
- `qfrc_fluid`: Fluid forces
- `qfrc_gravcomp`: Gravity compensation forces-->
- `qfrc_passive`: Passive forces
- `qfrc_spring`: Spring forces

## Sensors and Sites
- `sensordata`: Sensor measurements
- `site_xmat`, `site_xpos`: Site orientation matrices and positions

<!-- ## Solver Information -->
<!--- `solver_fwdinv`: Forward/inverse dynamics solver data
- `solver_niter`: Solver iteration counts
- `solver_nnz`: Number of non-zero elements-->

<!-- ## Tendon Properties -->
<!--- `ten_J`, `ten_length`, `ten_velocity`: Tendon Jacobians, lengths, and velocities
- `ten_wrapadr`, `ten_wrapnum`: Tendon wrapping information-->

## Spatial Transforms
<!--- `xanchor`: Joint anchor points
- `xaxis`: Joint axes-->
- `xfrc_applied`: Applied Cartesian forces
<!--- `ximat`, `xipos`: Internal orientation matrices and positions-->
- `xmat`, `xpos`: Orientation matrices and positions
- `xquat`: Quaternion orientations

