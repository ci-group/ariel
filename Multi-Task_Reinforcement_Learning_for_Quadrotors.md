IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 10, NO. 3, MARCH 2025 

2112 

## Multi-Task Reinforcement Learning for Quadrotors 

Jiaxu Xing , Ismail Geles , Yunlong Song , Elie Aljalbout , and Davide Scaramuzza _, Senior Member, IEEE_ 

_**Abstract**_ **—Reinforcement learning (RL) has shown great effectiveness in quadrotor control, enabling specialized policies to develop even human-champion-level performance in single-task scenarios. However, these specialized policies often struggle with novel tasks, requiring a complete retraining of the policy from scratch. To address this limitation, this paper presents a novel multi-task reinforcement learning (MTRL) framework tailored for quadrotor control, leveraging the shared physical dynamics of the platform to enhance sample efficiency and task performance. By employing a multi-critic architecture and shared task encoders, our framework facilitates knowledge transfer across tasks, enabling a single policy to execute diverse maneuvers, including high-speed stabilization, velocity tracking, and autonomous racing. Our experimental results, validated both in simulation and real-world scenarios, demonstrate that our framework outperforms baseline approaches in terms of sample efficiency and overall task performance. Video is available at https://youtu.be/HfK9UT1OVnY.** 

_**Index Terms**_ **—Reinforcement learning, machine learning for robot control, aerial systems: perception and autonomy.** 

## I. INTRODUCTION 

EAL world quadrotor applications typically involve mul- **R** tiple tasks and skills. For example, in search and rescue scenarios or inspection, quadrotors are required to perform a range of specific tasks within a single mission, such as tracking moving objects, maintaining stable hover positions, and precisely following designated paths or targets. To meet these diverse demands, a generalist control policy that can effectively manage these tasks can greatly enhance the versatility and adaptability of quadrotors, making them more effective in real-world applications. However, developing a generalist controller for multiple tasks is a challenging problem since different tasks often have different objectives and state spaces. For instance, in quadrotor control, the objective for hovering is to stabilize the vehiclebyreducingitsvelocitytozero,whereasquadrotorracing requires maximizing speed while avoiding collisions with gates. These tasks inherently conflict with their goals and demand different observations and strategies. 

Fig. 1. The proposed approach performs three distinct tasks for quadrotor control in the real world. The resulting _single MTRL policy_ can ( _Top_ ) stabilize the quadrotor from high speed, ( _Middle_ ) autonomously race through a fixed track, and ( _Bottom_ ) track randomly generated velocities. 

Received 3 September 2024; accepted 26 November 2024. Date of publication 23 December 2024; date of current version 22 January 2025. This article was recommended for publication by Associate Editor J. Panerati and Editor G. Loianno upon evaluation of the reviewers’ comments. (http://rpg.ifi.uzh.ch). This work was supported in part by the European Union’s Horizon Europe Research and Innovation Programme under Grant Agreement 101120732 (AUTOASSESS) and in part by the European Research Council (ERC) under Grant Agreement 864042 (AGILEFLIGHT). _(Corresponding author: Jiaxu Xing.)_ 

The authors are with the Robotics and Perception Group, Department of Informatics, University of Zurich, 8006 Zürich, Switzerland, and also with the Department of Neuroinformatics, University of Zurich and ETH Zurich, 8006 Zürich, Switzerland (e-mail: jixing@ifi.uzh.ch). 

This letter has supplementary downloadable material available at https://doi.org/10.1109/LRA.2024.3520894, provided by the authors. Digital Object Identifier 10.1109/LRA.2024.3520894 

In this work, we tackle the multi-task quadrotor control problem using deep reinforcement learning (RL), which offers the advantage of automatically optimizing parametric controllers through trial and error. RL is particularly effective in handling highly non-linear dynamical systems and non-convex, nondifferentiable objectives—challenges that are typically difficult for conventional optimization-based methods such as Model Predictive Control (MPC) [1], [2]. RL has demonstrated significant success in different domains of agile quadrotor control, ranging from time-optimal drone racing to obstacle avoidance [1], [3], [4]. 

2377-3766 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information. 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

XING et al.: MULTI-TASK REINFORCEMENT LEARNING FOR QUADROTORS 

2113 

State-of-the-art RL approaches have demonstrated specializationwithgreat performance, evenreachingthehuman-champion level in single-task scenarios such as drone racing [5]. However, these specialized policies often struggle to perform novel, outof-distribution tasks and require retraining from scratch when faced with even minor changes in task configuration [6]. Consequently, the commonly used task-specific training approach is unsuitable for multi-task scenarios, making developing a multi-task RL policy for quadrotors a significant challenge. A natural solution to MTRL problems is to train a network jointly on all tasks to uncover shared structures that can improve efficiency and performance beyond what individual task solutions can achieve. However, learning multiple tasks simultaneously often poses a challenging optimization problem involving conflicting objectives [7], sometimes resulting in poorer overall performance and data efficiency compared to individual task learning. 

Despite these challenges, previous work on MTRL has demonstrated the potential of integrating shared task structures to perform various manipulation skills, such as lifting and pickand-place, particularly in fixed-base manipulation scenarios [8], [9], [10], [11]. While MTRL has shown promise in manipulation on a simulation benchmark [12], its application for quadrotors remains largely unexplored. 

## _I. Contributions_ 

In this paper, we propose the first MTRL framework for learning various quadrotor control tasks efficiently. A key advantage of MTRL is its ability to share knowledge across different tasks, thereby enhancing learning efficiency. Although the reward objectives of RL may vary between tasks, the underlying physical dynamics of the quadrotor, such as mass, inertia, and other physical properties, remain constant throughout the learning process. Leveraging these invariant conditions, our framework efficiently utilizes information sharing to learn multiple tasks within a single policy. 

The reinforcement learning framework employs a multi-critic setup, and we propose a shared task encoder for observations containing dynamical information while handling task-specific observations using task-specific encoding networks. By sharing common information and isolating task-specific data, we have developed a high-performance policy capable of executing maneuvers such as stabilization, velocity tracking, and racing. 

We validate our MTRL policy in both simulation and realworld settings. We demonstrate that our framework outperforms baseline approaches in terms of sample efficiency and overall task performance. We believe this advancement is an important step toward developing a generalist quadrotor control policy, which could enable robots to handle diverse tasks in real-world scenarios more effectively. 

## II. RELATED WORKS 

## _A. Reinforcement Learning for Quadrotor Control_ 

In recent years, Reinforcement Learning (RL) has gained significant attention for the control of quadrotors. The work [13] 

is one of the first successful applications of RL to quadrotor control by tracking waypoints and recovering from challenging initial conditions. In [14], the performance gains of using RL for low-level control over classical methods are demonstrated. Besides obstacle avoidance [15], [16], drone racing represents an important benchmark task for agile quadrotor flight, where the impact of reinforcement learning is very significant [1], [5], even outperforming state-of-the-art model-based control [1]. By optimizing several aspects of the training framework, in [17] it is demonstrated RL policies may be trained within seconds. Several works focus on vision-based flight with reinforcement learning, such as [18], which uses a CAD model to produce discrete velocity commands directly from pictures. In [19], it is shown how RL bootstrapped imitation learning [3] benefits vision-based agile flight. In [5], the combination of visionbased state estimation and RL-based control enabled surpassing human world champions in drone racing. In [20], RL is used to learn drone racing from pixels without explicit state estimation. 

## _B. Multi-Task Reinforcement Learning for Robotics_ 

In [8], a large-scale collective robotic learning system, can rapidly acquire diverse skills by sharing exploration, experience, and representations across tasks, improving overall performance and capabilities. The work [9] demonstrates using MTRL to perform complex robotic skills like in-hand manipulation autonomously, significantly reducing the need for human resets during training sessions. In [21], modularization in network design is explored to facilitate MTRL, improving both sample efficiency and the performance of robotic tasks by dynamically configuring network modules according to the task requirements. In [22], a sequential multi-task learning scenario where a robot incrementally learns various tasks is demonstrated, using experiences from previous tasks to reduce the need for relearning and improve efficiency. Recently multi-task world models in proposed in [11], leveraging language model embeddings as task representations for model-based reinforcement learning of multiple robotic tasks. However, most existing MTRL research focuses on manipulation tasks with varying underlying dynamics across tasks, limiting opportunities for knowledge sharing. In contrast, our work leverages the consistent dynamics of the quadrotor, enabling us to propose a novel framework that enhances knowledge sharing for more efficient multi-task learning. 

## III. METHODOLOGY 

## _A. Notation_ 

In this manuscript, we define two reference frames. The first _W_ is the fixed world frame with its _z_ -axis aligned with gravity. The second frame _B_ is the quadrotor body frame. These reference frames are illustrated in Fig. 2. Vectors and matrices are represented as bold quantities, with capital letters denoting matrices. Vectors include a suffix indicating the frame in which they are expressed and their endpoint. For example, the quantity _**p** W B_ represents the position of the body frame _B_ relative to the 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 10, NO. 3, MARCH 2025 

2114 

**==> picture [85 x 45] intentionally omitted <==**

Fig. 2. Diagram of quadrotor model with the world and body frames and propeller numbering convention. 

world frame _W_ . The rotation matrix that transforms a vector from frame _B_ to _W_ is denoted by _**R** W B_ . 

## _B. Quadrotor Dynamics_ 

Let _**p** W B_ , _**q** W B_ , and _**v** W B_ represent the position, orientation, and linear velocity of the quadrotor, respectively, expressed in the world frame _W_ . Let _**ω** B_ denote the angular velocity of the quadrotor expressed in the body frame _B_ . Additionally, let _c_ = Σ _ici_ represent the body’s collective thrust, where _ci_ is the thrust produced by the _i_ -th motor, and let _**c**_ = [0 0 _c_ ][⊺] denote the collective thrust vector. Here, _m_ represents the mass of the quadrotor, and _**g** W_ is the gravity vector. Finally, let _**J**_ be the diagonal moment of inertia matrix, and _**τ** B_ the body’s collective torque. The quadrotor’s dynamic model is 

**==> picture [223 x 55] intentionally omitted <==**

## _C. Policy Learning_ 

_1) Problem Formulation:_ In the MTRL setting, we have _N_ tasks _T_ = _{T_ 1 _, . . . , Ti, . . . , TN }_ , where each task _Ti_ has a specific reward function _ri_ . Each task is defined by a Markov Decision Process (MDP) _Mi_ = ( _Si, Ai, Pi, Ri, γi_ ), where _Si_ is the state space, _Ai_ is the action space, _Pi_ is the transition probability, _Ri_ is the reward function, and _γi_ is the discount factor. In the MTRL setting, the goal is to learn a policy _π_ that maximizes the expected return _J_ ( _π_ ) across all tasks. Given uniformly sampled tasks, the optimization objective is defined as 

**==> picture [204 x 31] intentionally omitted <==**

In our work, our multi-task configurations are primarily distinguished by the _reward settings R_ as model dynamics _P_ remain identical duetotheunchangingphysical properties of thedrones. In the following sections, we detail the tasks considered in this work and the policy learning approach. 

_2) Autonomous Racing:_ The autonomous racing task can be formulated as an optimization problem that aims to minimize 

the time required for an agile quadrotor to navigate through a predefined sequence of gates [23], as shown in Fig. 5. In this task, we use the observations _**o**_ = [ _**p** ,_ _**R**_[˜] _,_ _**v** ,_ _**ω** , a_ prev _, δ_ _**p**_ 1 _, δ_ _**p**_ 2], where _**p** ∈_ R[3] denotes the drone’s position, _**R**_[˜] _∈_ R[6] is a vector comprising the first two columns of _**R** W B_ [24], _**v** ∈_ R[3] and _**ω** ∈_ R[3] denote the linear and angular velocity of the drone, _a_ prev represents the previous action from the actor policy, _δ_ _**p**_ 1 _∈_ R[12] represents the relative position differences of the four upcoming gate corners on the race track with respect to the drone agent, with each corner specified by a 3D position in the world frame. Similarly, _δ_ _**p**_ 2 _∈_ R[12] represents the relative position differences of the corners from the next gate to the gate after that. Here _δ_ _**p**_ 1 represents the difference of the 4 corners of the next gate to pass between the current quadrotor position. And _δ_ _**p**_ 2 represents the positional difference of the corners between the next gate to pass and the gate after the next gate to pass on the race track. The RL policy training rewards are adjusted based on [1]. The reward at time _t_ , denoted as _rt_ , is defined as the sum of various components 

**==> picture [225 x 13] intentionally omitted <==**

where _rt_[prog] encourages progress towards the next gate to be passed [1], _rt_[perc] encodes perception awareness by adjusting the quadrotor’s attitude such that the optical axis of its camera points towards the next gate’s center, _rt_[act][penalizes action changes from] the last time step, _rt_[br][penalizes][body][rates][and][consequently] reduces motion blur, _rt_[pass] is a binary reward that is active when the robot successfully passes the next gate, _rt_[crash] is a binary penalty that is only active when a collision happens, which also ends the episode. The reward components are formulated as follows 

**==> picture [216 x 106] intentionally omitted <==**

_3) Stabilization From High Speed:_ In the stabilization task, the quadrotor is expected to recover to the static status, given randomized poses and high initial velocities. Here, successful stabilization is defined as achieving near-zero velocities around a predefined height _zd_ , starting from random initial positions and velocities. In this setting, the quadrotor is initialized with a random position, orientation, and linear and angular velocities. The observation contains _**o** h_ = [ _**p** ,_ _**R**_[˜] _,_ _**v** ,_ _**ω** , a_ prev _,_ ¨ _**p** W B, zd_ ]. Here _**p**_ ¨ _W B_ is defined as the acceleration of the quadrotor agent in the world frame, and _zd_ represents a predefined constant height in meters to stabilize at. The reward function _rt_[stabilize] is defined as 

**==> picture [243 x 25] intentionally omitted <==**

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

2115 

XING et al.: MULTI-TASK REINFORCEMENT LEARNING FOR QUADROTORS 

Here _rt_[height] rewards the quadrotor for maintaining a constant height, _rt_[attitude] rewards the quadrotor for maintaining a constant orientation, _rt_[velocity] and _rt_[angular] penalizes the non-zero linear and angular velocities, _rt_[act] penalizes the non-smooth actions, and _rt_[success] is a discrete reward when the quadrotor is stabilized. All the rewards are formulated here using the _L_ 2 norm multiplied by a corresponding constant coefficient. The detailed rewards 

**==> picture [207 x 107] intentionally omitted <==**

During training, we applied a curriculum to gradually increase the difficulty of the tasks. We increase the initial speed of the quadrotor in the _x_ , _y_ , and _z_ axes by 10% for every 100,000 data samples. The curriculum will stop once the predefined upper limits in each direction are reached. 

_4) Velocity Tracking:_ In the velocity tracking task, the quadrotor is required to track a randomly generated velocity profile. The observation contains _**o** v_ = [ _**p** ,_ _**R**_[˜] _,_ _**v** ,_ _**ω** , a_ prev _,_ _**v** d,_ ¨ _**p** W B_ ], where _**v** d_ represent the desired linear velocity. The reward function _rt_[tracking] is defined as _rt_[tracking] = _rt_[velocity] + _rt_[br][+] _[ r] t_[act] _[,]_ (7) 

where _rt_[velocity] rewards the quadrotor for tracking the desired velocity, _rt_[br][penalizes the quadrotor for non-zero angular veloc-] ities, and _rt_[act] penalizes the quadrotor for excessive actions. All the rewards are formulated here using the _L_ 2 norm multiplied by a corresponding constant coefficient, 

**==> picture [180 x 50] intentionally omitted <==**

During training, we implemented a curriculum to progressively increase the difficulty of the tasks. The desired speed of the quadrotor inthe _x_ , _y_ , and _z_ axes was increasedby1 m _/_ s for every 100,000 data samples. This gradual increase will continue until the predefined upper-speed limits in each direction are achieved. 

## _D. Multi-Task Learning Framework_ 

Since the control or navigation tasks are typically not contactrich for a fixed quadrotor platform, we generally do not expect changes in the dynamics equations (see (1)), whether during different phases of a single task or across various tasks. Many previous works also present this valid assumption, from agile drone racing to obstacle avoidance [1], [4], [5]. Hence, in the quadrotor setting, we can utilize the consistent physical property to simplify the multi-task scenario by assuming an identical transition probability across all tasks. This assumption serves 

as the primary motivation for our proposed information-sharing structure in the MTRL setup, allowing us to share data samples across tasks to efficiently learn an encoding network for the observation information related to the transition dynamics. To leveragethesharedinformationacrosstasks,weproposeamultitask learning framework that consists of shared and task-specific modules. 

As we aim to have a single policy capable of solving multiple tasks, our actor needs to be common to all tasks. As shown in the previous sections, the observation space in different tasks shares some common features, namely the position, orientation, linear and angular velocities, and the previous action. The selection of the shared features is based on the fact that these represent the essential dynamical properties outlined in (1), which remain consistent across different tasks. Other features can be task-specific and have different dimensions, which would lead to different tasks having different policy input sizes. To distinguish observations from various potential tasks, we use an identifier based on the task-specific observation length, which allows us to incorporateone-hotencodingintothetask-specificobservations. To overcome this, we propose a feature encoding architecture including a shared encoder network among different tasks to extract the shared features from the observation. Meanwhile, we use task-specific encoders to generate the task-specific policy inputs. Although the task-specific observations’ dimensions vary from task to task, the task-specific encoder is designed to map the specific observation to the same latent space. 

For the MTRL part, we employ a shared actor policy with multiple individual critic networks, each corresponding to a specific task. The actor policy takes the concatenated latent features from the shared encoder and the task-specific encoder as input and outputs the action, namely the collective thrust and the body rates [4]. The overall architecture is shown in Fig. 3. We train the shared actor policy to maximize the expected return across all tasks, while the individual critic networks are trained to evaluate the value function for each task. In contrast, the task-specific critic networks take directly the full observation as the input and output of the value function for each task. Based on [25], we went for the choice not sharing the policy feature encoder with the critic networks, as it has been shown to improve the performance of the policy. 

## IV. EXPERIMENTS 

Using the individual tasks described in the previous section, we evaluate the performance of the proposed MTRL approach. Our experiments are designed to answer the following research questions: (i) How sample-efficient is our MTRL approach compared to the single-task RL baselines? (ii) How does the MTRL policy’s performance compare to the single-task RL policies? (iii) How do different knowledge-sharing strategies affect the MTRL performance? (iv) Does the result transfer to a real-world scenario? 

## _A. Training Configurations_ 

For thepolicytraining, weemployapolicynetworkconsisting of a two-layer MLP, each layer containing 256 neurons, with 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 10, NO. 3, MARCH 2025 

2116 

**==> picture [92 x 115] intentionally omitted <==**

**==> picture [95 x 43] intentionally omitted <==**

Fig. 3. Our MTRL framework utilizes a shared encoder for observations related to the quadrotor dynamics across all tasks. The embedding output from the shared encoder is then merged with the task-specific observation (e.g., the gate observation from the racing task and the desired velocity from the tracking task) to create a task-specific embedding. The policy uses both the concatenated embedding (64) from the shared embedding (32) and the task-specific embedding (32) to generate control commands. A separate critic function is used for each task, which is not employed during deployment. 

TABLE I 

## _B. Baselines_ 

REWARD PARAMETERS FOR MTRL TRAINING 

**==> picture [189 x 76] intentionally omitted <==**

TABLE II 

OVERVIEW OF THE DRONE PARAMETERS FOR BOTH SIMULATION AND REAL-WORLD EXPERIMENTS 

**==> picture [173 x 72] intentionally omitted <==**

a final layer outputting a 4-dimensional vector using a _tanh_ activation function. In our experiments, for the shared dynamic encoder, we use a three-layer MLP with 19 neurons in the input layers and 128 neurons in the hidden layers to generate a 32-dimensional latent embedding. For the task-specific encoder, we use a three-layer MLP with task-dependent input dimensions and 128 neurons in the hidden layer to generate a 32-dimensional latent embedding. Table I shows the detailed task reward parameters for the MTRL training. In our setting, we optimized our hyperparameters solely based on the single-task performance. For MTRL training, we employ model-free reinforcement learning approach using Proximal Policy Optimization (PPO) [26]. For the quadrotor platform used for both training and deployment, we detail the information regarding components and physical parameters in Table II. We use the Flightmare simulator [27] for policy training in simulation. 

In our experiments, we compare our approach with the following baselines: (i) **Single-task RL** : We train a separate policy for each task using the same RL algorithm as the MTRL approach. (ii) **MTRL-Actor** : For this baseline, we keep the _Actor_ as the only network shared among different tasks. And all the other encoder networks and critic networks are different among all tasks. We train a shared actor policy with multiple individual critic networks, each corresponding to a specific task. (iii) **MTRLSeperate** : To ablate the design choice of fusing shared and taskspecific observations, we train a shared encoder and task-specific encoders, but the task encoder does not receive the shared observation as input. The observation encoders are different for each task, and the shared actor policy takes the latent embedding directly as input to output the action. For all of the approaches, we performed 10 runs of the same training configurations using different random seeds, and we report the average evaluation metrics in this section. Since the primary contribution of our approach is to enhance learning efficiency and task performance of the RL policies using MTRL, we have chosen not to include baselines that rely solely on classical control approaches in our comparison. 

## _C. Sample Efficiency Analysis_ 

To evaluate the sample efficiency of our MTRL approach, we compare the return curves of the MTRL approach with the aforementioned baselines. All the policies are trained using the same hyperparameters and the same number of training steps. The number of training steps in our setting is 40 M, where all the policies’ performances converge. As shown in Fig. 4, the MTRL approach outperforms the single-task RL baselines in all three different tasks. The MTRL approach achieves a higher average return in the same number of training steps than single-task RL baselines. Notably, the single-task RL policies still perform closely to the MTRL approach when only sharing the actor network. This highlights the necessity of information sharing in our framework. However, although information sharing is beneficial, the MTRL-Seperate cannot fly at all in the 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

XING et al.: MULTI-TASK REINFORCEMENT LEARNING FOR QUADROTORS 

2117 

Fig. 4. Overview of the average return comparison of different tasks. It is clearly shown that our proposed MTRL approach achieves a higher average return within the same number of training steps compared to single-task RL baselines. Notably, single-task RL policies still perform comparably to the MTRL approach when only the actor network is shared. 

racing task. This is likely because if we do not fuse the shared information with task-specific information, it becomes difficult when the task requirements are conflicting, e.g. racing and stabilization from high speed. The policy will then prefer learning the rather easier task. Hence, this highlights the importance of fusing the shared and task-specific information in the MTRL framework. 

## _D. Individual Task Performance of MTRL Policy_ 

In this section, we showcase the individual task performance of our MTRL policy. We evaluate the performance of the MTRL policy in three different tasks: stabilization from high speed, racing, and velocity tracking. The MTRL policy is trained using the same hyperparameters as the single-task RL policies. 

_1) Racing Performance:_ In the racing task, we evaluate the performance of the MTRL policy by racing on a predefined race track. The race track contains 6 gates, and the quadrotor is required to pass these gates in a fixed order. The policy is evaluated in 64 different starting positions in uniformly sampled starting positions. Fig. 5 visualizes one sample rollout that successfully completes the race track. 

_2) Stabilization Performance:_ As shown in Fig. 6(a), we evaluate the high-speed stabilization task by randomly initializing the quadrotor with various positions, orientations, and velocities. The initial speeds in the _x_ and _y_ directions are randomly sampledfrom [ _−_ 20 _,_ 20]m _/_ s,whilethevelocityinthe _z_ direction is sampled from [ _−_ 4 _,_ 4] m _/_ s. To prevent the quadrotor from crashing into the ground when the initial _z_ -direction velocity is high and negative, we adjust the initial height accordingly to ensure the quadrotor will not crash within one second, even if no control input is applied. From the results, we observe that the MTRL policy can successfully stabilize the quadrotor in the hovering condition from high speed task within seconds, even when the initial conditions are challenging; the maximum initial speed from our experiments is 82.57 kmh _[−]_[1] . This demonstrates the robustness of the MTRL policy in stabilization from highspeed tasks. 

Fig. 5. Illustration of one racing policy rollout. The policy successfully completes a Figure-8 race track, which consists of six gates, with a 100% success rate. 

_3) Velocity Tracking Performance:_ In the tracking task, we evaluate the performance of the MTRL policy by tracking a randomly generated velocity command, where we apply random walks in the acceleration space. We randomize the velocity references in _x_ and _y_ directions up to 54 kmh _[−]_[1] and 18kmh _[−]_[1] in _z_ direction. As shown in Fig. 6(b), the MTRL policy can successfully track the challenging commanded velocities, where they can go up to 50 kmh _[−]_[1] in a very short time, and we did not even include the non-holonomic constraints of the quadrotor to generate velocity command. 

## _E. Quantitative Analysis_ 

_1) Evaluation Metrics:_ For the autonomous racing task, we use three evaluation metrics: success rate (SR), mean-gatepassing-error (MGE), and lap time (LT). SR is the ratio of completed laps to total trials. MGE measures the distance between the drone’s position and the gate center when passing through; here, the inner gate size used for experiments is 1.5 m. LT indicates the duration of completing a full race track and 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 10, NO. 3, MARCH 2025 

2118 

**==> picture [205 x 221] intentionally omitted <==**

**==> picture [205 x 241] intentionally omitted <==**

Fig. 6. Visualizations of the MTRL policy on individual task performance. (a) The MTRL policy successfully stabilizes the quadrotor in a hover within seconds, even from high-speed tasks and challenging initial conditions. (b) The MTRL policy successfully tracks the commanded velocity, even for challenging trajectories with speeds reaching up to 50 kmh _[−]_[1] . 

TABLE III 

INDIVIDUAL TASK PERFORMANCE AFTER 20 M AND 40 M TRAINING SAMPLES 

**==> picture [449 x 78] intentionally omitted <==**

flying through all gates. For the stabilization task, we employ two evaluation metrics, namely _t_ half and _t_ full, to evaluate the time usage of the RL controller to stabilize the quadrotor. Here _t_ half is the time usage to reduce the actual velocity to half of the initial velocity, and _t_ full is the time usage to control a quadrotor to hover condition; in our experiment, we determine this when the quadrotor’s linear velocity is smaller than 0 _._ 5 m _/_ s. For the velocity tracking task, we simply use a tracking error metric _**ev**_ , which computes the averaged velocity difference over time. 

_2) Analysis:_ In Table III, we present a quantitative comparison of our approaches with the baselines. To demonstrate the effectiveness of the policy trained with different numbers of samples, we list the performance of the MTRL policies in two different timesteps, namely 20 M and 40 M. First of all, for the policy trained with 20 M steps, our MTRL approach demonstrates a much better task performance than all of the baseline approaches. This strongly demonstrates the sample efficiency of our approach. Secondly, when the policy is trained till convergence, the policy’s performance of our approach is still not worse than learning individually in all the tasks and all the metrics. This further indicates the claim of our approach: our 

MTRL framework could learn multiple tasks efficiently without trading off performance. 

## _F. Real World Performance_ 

To demonstrate the effectiveness of our policy improvements, we conducted validation tests in real-world scenarios. We utilized an Agilicious quadrotor platform [28] with the identical properties presented in I with state estimation provided by a VICON motion capture system to feed accurate inputs to the policy. For low-level control, the BetaFlight2 firmware was employed to track the commanded collective thrusts and body rates. We conducted five individual runs for each task, varying the starting conditions in each run. Remarkably, our MTRL policy achieved a 100% success rate across all tasks in these realworld experiments, as illustrated in Fig. 1. These results clearly indicate that our policy effectively transfers to and performs reliably in real-world scenarios. The detailed evaluation results are presented in Table IV. The metrics for each task indicate that our policy consistently delivers stable performance both in simulation and the real world using identical configurations. 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

XING et al.: MULTI-TASK REINFORCEMENT LEARNING FOR QUADROTORS 

2119 

TABLE IV 

COMPARISON OF OUR MTRL POLICY’S TASK PERFORMANCE BETWEEN SIMULATION AND REAL WORLD 

**==> picture [216 x 35] intentionally omitted <==**

## _G. Discussion_ 

Our experiments thoroughly evaluated the proposed MTRL framework, addressing its critical performance aspects. First, we demonstrated that the MTRL approach significantly improves sample efficiency and task performance compared to single-task RL baselines. After training for half the amount of total samples (20 M), our MTRL method reduced stabilization time by 18% ( _t_ half) and gate passing error (MGE) by 16% compared even to individual RL methods. Furthermore, even when trained to convergence, our MTRL approach consistently performs better than, or as well as other baseline methods. Notably, it achieved a 6.7% reduction in gate passing error in racing tasks compared to the next best approach. In contrast, the MTRL-Separate baseline without the integration of shared and task-specific information resulted in a 0% success rate in the racing task, as it could not even complete the task. This failure underscores the importance of properly integrating shared and task-specific elements within our MTRL framework. Finally, the deployment of our MTRL policy in real-world scenarios confirmed its robustness, with performance closely matching that observed in simulations. 

## V. CONCLUSION 

In this work, we introduced the first Multi-Task Reinforcement Learning (MTRL) framework specifically designed for quadrotor control, addressing the challenges posed by diverse task requirements in real-world scenarios. By leveraging the shared physical dynamics of the quadrotor and employing a novel multi-critic setup with a shared task-agnostic observation encoder,ourapproachsuccessfullyintegratesinformationacross different tasks while maintaining high performance. The experimental results, both in simulation and real-world applications, demonstrated the effectiveness and efficiency of our method, particularly in enhancing sample efficiency when learning different tasks while maintaining strong task performance. This advancement paves the way for more versatile quadrotor control systems capable of performing a wide range of tasks within a single mission, thereby significantly contributing to the broader application of quadrotors in critical areas like search and rescue and infrastructure inspection. 

## REFERENCES 

- [1] Y. Song, A. Romero, M. Mueller, V. Koltun, and D. Scaramuzza, “Reaching the limit in autonomous racing: Optimal control versus reinforcement learning,” _Sci. Robot._ , vol. 8, 2023, Art. no. eadg1462. 

- [2] J. Xing, G. Cioffi, J. Hidalgo-Carrió, and D. Scaramuzza, “Autonomous power line inspection with drones via perception-aware MPC,” in _Proc. 2023 IEEE/RSJ Int. Conf. Intell. Robots Syst._ , 2023, pp. 1086–1093. 

- [3] J. Xing, L. Bauersfeld, Y. Song, C. Xing, and D. Scaramuzza, “Contrastive learning for enhancing robust scene transfer in vision-based agile flight,” in _Proc. 2024 IEEE Int. Conf. Robot. Automat._ , 2024, pp. 5330–5337. 

- [4] E. Kaufmann, L. Bauersfeld, and D. Scaramuzza, “A benchmark comparison of learned control policies for agile quadrotor flight,” in _Proc. Int. Conf. Robot. Automat._ , 2022, pp. 10504–10510. 

- [5] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Müller, V. Koltun, and D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” _Nature_ , vol. 620, no. 7976, pp. 982–987, Aug. 2023. 

- [6] H. Wang, J. Xing, N. Messikommer, and D. Scaramuzza, “Environment as policy: Learning to race in unseen tracks,” 2024, _arXiv:2410.22308_ . 

- [7] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu, “Conflict-averse gradient descent for multi-task learning,” in _Proc. Adv. Neural Inf. Process. Syst._ , 2021, vol. 34, pp. 18878–18890. 

- [8] D. Kalashnikov et al., “Scaling up multi-task robotic reinforcement learning,” in _Proc. Conf. Robot Learn._ , 2022. 

- [9] A. Gupta et al., “Reset-free reinforcement learning via multi-task learning: Learning dexterous manipulation behaviors without human intervention,” in _Proc. 2021 IEEE Int. Conf. Robot. Automat._ , 2021, pp. 6664–6671. 

- [10] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “Sharing knowledge in multi-task deep reinforcement learning,” in _Proc. Int. Conf. Learn. Representations_ , 2020. 

- [11] E. Aljalbout, N. Sotirakis, P. van der Smagt, M. Karl, and N. Chen, “LIMT: Language-informed multi-task visual world models,” 2024, _arXiv:2407.13466_ . 

- [12] T. Yu et al., “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning,” in _Proc. Conf. Robot Learn._ , 2020. 

- [13] J. Hwangbo, I. Sa, R. Siegwart, and M. Hutter, “Control of a quadrotor with reinforcement learning,” _IEEE Robot. Automat. Lett._ , vol. 2, no. 4, pp. 2096–2103, Oct. 2017. 

- [14] N. O. Lambert, D. S. Drew, J. Yaconelli, S. Levine, R. Calandra, and K. S. Pister, “Low-level control of a quadrotor with deep model-based reinforcement learning,” _IEEE Robot. Automat. Lett._ , vol. 4, no. 4, pp. 4224–4230, Oct. 2019. 

- [15] G. Zhao, T. Wu, Y. Chen, and F. Gao, “Learning speed adaptation for flight in clutter,” _IEEE Robot. Automat. Lett._ , vol. 9, no. 8, pp. 7222–7229, Aug. 2024. 

- [16] Z. Huang, Z. Yang, R. Krupani, B. ¸Senba¸slar, S. Batra, and G. S. Sukhatme, “Collision avoidance and navigation for a quadrotor swarm using endto-end deep reinforcement learning,” in _Proc. IEEE Int. Conf. Robot. Automat._ , 2024, pp. 300–306. 

- [17] J. Eschmann, D. Albani, and G. Loianno, “Learning to fly in seconds,” _IEEE Robot. Automat. Lett._ , vol. 9, no. 7, pp. 6336–6343, Jul. 2024. 

- [18] F. Sadeghi and S. Levine, “CAD2RL: Real single-image flight without a single real image,” in _Proc. Robot., Sci. Syst._ , 2017. 

- [19] J. Xing, A. Romero, L. Bauersfeld, and D. Scaramuzza, “Bootstrapping reinforcement learning with imitation for vision-based agile flight,” in _Proc. Conf. Robot Learn._ , 2024. 

- [20] I. Geles, L. Bauersfeld, A. Romero, J. Xing, and D. Scaramuzza, “Demonstrating agile flight from pixels without state estimation,” in _Proc. Robot., Sci. Syst._ , 2024. 

- [21] R. Yang, H. Xu, Y. Wu, and X. Wang, “Multi-task reinforcement learning with soft modularization,” in _Proc. Adv. Neural Inf. Process. Syst._ , 2020, vol. 33, pp. 4767–4777. 

- [22] A. Xie and C. Finn, “Lifelong robotic reinforcement learning by retaining experiences,” in _Proc. Conf. Lifelong Learn. Agents_ , 2022. 

- [23] Y. Song, M. Steinweg, E. Kaufmann, and D. Scaramuzza, “Autonomous drone racing with deep reinforcement learning,” in _Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst._ , 2021, pp. 1205–1212. 

- [24] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representationsinneuralnetworks,”in _Proc.IEEE/CVFConf.Comput.Vis. Pattern Recognit._ , 2019, pp. 5745–5753. 

- [25] M. Andrychowicz et al., “What matters in on-policy reinforcement learning? a large-scale empirical study,” 2020, _arXiv:2006.05990_ . 

- [26] J.Schulman,F.Wolski,P.Dhariwal,A.Radford,andO.Klimov,“Proximal policy optimization algorithms,” 2017, _arXiv:1707.06347_ . 

- [27] Y. Song, S. Naji, E. Kaufmann, A. Loquercio, and D. Scaramuzza, “Flightmare: A flexible quadrotor simulator,” in _Proc. Conf. Robot Learn._ , 2020. 

- [28] P. Foehn et al., “Agilicious: Open-source and open-hardware agile quadrotor for vision-based flight,” _Sci. Robot._ , vol. 7, no. 67, 2022. 

Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on June 15,2026 at 14:03:17 UTC from IEEE Xplore.  Restrictions apply. 

