IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED FEBRUARY, 2026 

1 

## Efficient Knowledge Transfer for Jump-Starting Control Policy Learning of Multirotors through Physics-Aware Neural Architectures 

Welf Rehberg, Mihir Kulkarni, Philipp Weiss and Kostas Alexis 

_**Abstract**_ **—Efficiently training control policies for robots is a major challenge that can greatly benefit from utilizing knowledge gained from training similar systems through cross-embodiment knowledge transfer. In this work, we focus on accelerating policy training using a library-based initialization scheme that enables effective knowledge transfer across multirotor configurations. By leveraging a physics-aware neural control architecture that combines a reinforcement learning-based controller and a supervised control allocation network, we enable the reuse of previously trained policies. To this end, we utilize a policy evaluationbased similarity measure that identifies suitable policies for initialization from a library. We demonstrate that this measure correlates with the reduction in environment interactions needed to reach target performance and is therefore suited for initialization. Extensive simulation and real-world experiments confirm that our control architecture achieves state-of-the-art control performance, and that our initialization scheme saves on average up to** 73 _._ 5% **of environment interactions (compared to training a policy from scratch) across diverse quadrotor and hexarotor designs, paving the way for efficient cross-embodiment transfer in reinforcement learning.** 

_**Index Terms**_ **—Aerial Systems: Mechanics and Control, Reinforcement Learning.** 

**==> picture [213 x 168] intentionally omitted <==**

**----- Start of picture text -----**<br>
Library of Policies (     )<br>New<br>Configuration<br>~~CixLibraryco  > of Policies (£os we) oa Configuration New<br>SK OK X Cn<br>_<br>-j ow—— 1:<br>Similarity Measure Similarity Measure Control Allocation Control Allocation<br>Choose configuration for initialization using  reward-based  o™<br>similarity measure: Supervised<br>Learning<br>Cn Te; . ai,is<br>Network Initialization Network Initialization Position Controller Position Controller<br>Initialize network weights of  o™<br>controller network:<br>Reinforcement<br>Learning<br>**----- End of picture text -----**<br>


Fig. 1: The proposed library-based initialization scheme: For training a policy for a new configuration, we first train a configuration-specific control allocation network. Subsequently, a suitable policy for initializating the training of the control policy is picked from a library of policies using a rewardbased similarity measure. The resulting training time is significantly lower than training a policy from scratch. 

## I. INTRODUCTION 

ULTIROTORS are popular for their agility and simple **M** design, with recent advances in onboard autonomy leading to deployments in diverse environments [1]–[3] and even outperforming expert-level pilots in drone racing [4]. Yet, design still follows the traditional sequence: airframe first, then control policy. An emerging paradigm frames autonomous multirotor design as a computational co-design problem, jointly optimizing airframe and control policy [5]– [7]. This approach envisions evaluating thousands of candidate designs by training a policy for each, but is fundamentally limited by the prohibitive cost of training so many policies. Training reinforcement learning (RL) policies is computationally intensive, and thus, suitable initialization can greatly 

Manuscript received: October, 01, 2025; Revised December, 09, 2025; Accepted February, 13, 2026. 

This paper was recommended for publication by Editor Giuseppe Loianno upon evaluation of the Associate Editor and Reviewers’ comments. This work was supported by the Horizon Europe Grant Agreement No. 101119774 and the NVIDIA Academic Grant Program using NVIDIA RTX PRO 6000 Blackwell Max-Q and A100 GPU-Hours on Saturn Cloud. _(Corresponding author: Welf Rehberg)_ 

All authors are with the Department of Engineering Cybernetics at the Norwegian University of Science and Technology, O.S. Bragstads Plass 2D, 7034, Trondheim, Norway (e-mails: _{_ welf.rehberg, mihir.kulkarni, philipp.weiss, konstantinos.alexis _}_ @ntnu.no). Digital Object Identifier (DOI): see top of the page. 

accelerate convergence. A promising strategy is to leverage policies pre-trained on related configurations, either to initialize parameters [8] or guide exploration [9]. This relies on control policies generalizing across similar systems, a premise supported, among others, by domain randomization, which produces controllers effective not only for a nominal design in simulation but also for nearby system variations in reality. 

We propose an efficient knowledge-transfer procedure for policy learning across diverse multirotor configurations, enabling fast training of large numbers of robots within the same system family. This challenge is not unique to cooptimization and also arises in general control-policy learning settings [10]. Our physics-aware neural architecture separates control into two parts: an RL-trained feedback policy mapping states to wrench vectors, and an allocation network trained via supervised learning guided by a Quadratic Programming (QP) expert. This physics-aware architecture enables separate verification of controller and control allocation and supports reusing policies to jump-start learning for new airframe designs. To accelerate learning, we maintain a library of airframe–policy pairs. Using a similarity measure, a suitable prior configuration is selected whose policy and optimizer states are used to initialize training for new airframes. In turn, this greatly reduces the RL effort of training the controller, while 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED FEBRUARY, 2026 

2 

the allocation network is always exclusively trained for the system at hand. We validate our approach through simulations, ablation studies, and real-world experiments, verifying its core contributions: 

- A library-based initialization scheme for control policy learning, saving on average up to 73 _._ 5% of environment interactions (compared to training a policy from scratch) across diverse quadrotor and hexarotor designs 

- A physics-aware neural control architecture, facilitating policy initialization and improved interpretability 

The remainder of this paper is structured as follows: Section II reviews related work; Section III presents the methods, including the multirotor dynamics, policy architecture, and initialization scheme; Section IV reports real and simulated evaluations; and Section V provides a summary and outlook. 

## II. RELATED WORK 

Work on cross-embodiment policy transfer has focused mainly on manipulation tasks. Chen et al. [11] propose zeroshot transfer, while Wang et al. [12] retrain network components to align action and state spaces. Uchendu et al. [9] reuse suboptimal policies as guides in a curriculum to accelerate exploration, and Julian et al. [8] show that fine-tuning policies initialized from other tasks reduces training time. While these works demonstrate transfer across configurations, they do not address how to select an initialization policy. Earlier work on embodiment transfer in design optimization includes training general policies for efficient initialization [13], [14], transferring via intermediate embodiments [6], [15], and selecting from a discrete set of pretrained policies [6], [16]. Generalpolicy initialization requires either pretraining a broad policy [14] or training multiple networks in parallel [13], with both approaches constrained by the need for identical network architectures, limiting individual policy size. While general policies for multirotor control have not, to our knowledge, been applied to design optimization, they exhibit similar shortcomings [10] and are targeted to planar quadrotors so far [10], [17]. While using a curriculum of transitioning configurations in training is able to transfer policies for complex systems and tasks like object manipulation with robotic hands [15], they do not mention how the policy for initialization is picked among multiple already trained configurations, and the method adds significant additional complexity to the training process by making continuous changes to the robot model necessary. Earlier works on library-based approaches [6], [16] mitigate these issues by using the original policies trained for individual configurations and reusing the experience gained with already trained policies. While [16] reuses the best policy for each actuator-sensor combination, they do not consider the dynamical differences due to the different placement and orientation of the actuators and sensors. In [6], the authors take the differences into account by approximating how similar two configurations are based on the difference of the parameter vector fully defining the configuration. Although this method offers a computationally efficient way to approximate how suitable a configuration is for initialization, it does not necessarily correlate well with improvements in sample efficiency when used for policy selection, as we will 

**==> picture [115 x 76] intentionally omitted <==**

Fig. 2: Description of arbitrary airframes considered in this work. 

demonstrate later in this work. Additionally, no evaluation of the initialization scheme was conducted. Last, we note that prior work on reinforcement learning for multirotors [18], [19] often uses end-to-end policies mapping states directly to motor commands. Drawing inspiration from classical control, we propose a modular framework in which one module generates wrench commands while another allocates actuator inputs. This separation enhances interpretability, particularly for arbitrary multirotor configurations. Building on this architecture, we further introduce a library-based initialization scheme that leverages a similarity measure and present an evaluation of its efficiency. 

## III. APPROACH 

## _A. Dynamical Systems Representation_ 

The morphology of the individuals of the multirotor family considered in this work is defined by the parameter vector: 

**==> picture [215 x 11] intentionally omitted <==**

with _nm ∈_ R being the motor number and _ti ∈_ R[3] and **R** _i ∈ SO_ (3) being translations and orientations of the _i_ -th rotor with respect to the body frame _B_ placed at the center of mass as shown in Fig. 2. _vec_ () describes flattening the lower-triangle matrix into a vector. In this work, we consider systems with _nm ∈{_ 4 _,_ 6 _}_ (i.e., quadrotors and hexarotors). The system state is described by _x_ = [ _p, v, q, ωB_ ] _[T] ∈_ R[13] , with _p ∈_ R[3] being the position, _v ∈_ R[3] being the velocity (both expressed in the world frame _W_ ), _q ∈_ H being the unit quaternion rotation (parametrizing the rotation matrix **R** ( _q_ )), and _ωB ∈_ R[3] being the rotational velocity in the body frame. The system dynamics can therefore be written as [20]: 

**==> picture [241 x 47] intentionally omitted <==**

where **J** _∈_ R[3] _[×]_[3] denotes the inertia matrix of the system in _B_ depending on the motor positions, _m ∈_ R its mass, _g_ = [0 _,_ 0 _, −_ 9 _._ 81 _m/s_[2] ] _[T]_ the gravity vector, _F ∈_ R[3] the applied combined force, _τ ∈_ R[3] the applied torque in _B_ , and _⊗_ denotes the quaternion product. We assume all systems to have the same payload and assume that the mass difference resulting from different arm lengths is negligible. The applied thrust and torque are related to the commanded motor thrusts _u ∈_ R _[n][m]_ as follows: 

**==> picture [156 x 25] intentionally omitted <==**

REHBERG _et al._ : EFFICIENT KNOWLEDGE TRANSFER FOR JUMP-STARTING CONTROL POLICY LEARNING OF MULTIROTORS 

3 

**==> picture [208 x 72] intentionally omitted <==**

**----- Start of picture text -----**<br>
Controller Network Controller Network Scaling Scaling Allocation Allocation Network Network<br>> “11 = [Finins maz] SS<br>Training Data for<br>BRO § NN-allocation Ges pis<br>Admissible Wrench<br>a Set<br>Normalized Wrench Command Physical Wrench Command<br>**----- End of picture text -----**<br>


Fig. 3: Proposed physics-aware neural control architecture. The architecture is split into a controller and an allocation network, which are trained separately. The control network is trained using RL, while the allocation network is trained via supervised learning. 

**==> picture [243 x 50] intentionally omitted <==**

with **F** _∈_ R[6] _[×][n][m]_ representing the allocation matrix of the system, _αi ∈{_ 1 _, −_ 1 _}_ representing the rotation direction of rotor _i_ , _cq ∈_ R being the torque to thrust ratio of the motors and _zmi ∈_ R[3] representing the thrust direction of motor _i_ in motor frame ([0 _,_ 0 _,_ 1] _[T]_ for all motors). The relation between the commanded motor thrust and the rotational velocity ( _ωi_ ) of motor _i_ is defined by _ui_ = _ctωi_[2][with] _[c][t][∈]_[R][being][the] thrust coefficient. 

## _B. Physics-aware Control Policy Learning_ 

In this work, we depart from earlier efforts on learning position controllers that directly output motor commands through a single neural network [18], [19] and propose a twostep approach that employs a physics-aware neural architecture trained with a combination of reinforcement and supervised learning. The proposed neural architecture is depicted in Fig. 3. As shown, the method involves first learning the control allocation (from desired wrench to motor commands) in a supervised manner and then fixing that learned network while training a position controller outputting wrenches ( _w_ = [ _F, τ_ ] _[T]_ ) through reinforcement learning. Splitting the controller into two parts leaves us with a network where the first part depends on the admissible wrench set, the mass, and the inertia of the system, while the second part is specific to the system’s allocation matrix. Since the first part of the controller commands forces and torques to a rigid body, this is hypothesized and demonstrated in practice that it generalizes between different system configurations (including different motor numbers) and can, therefore, be initialized efficiently from already trained policies for other configurations. Additionally, the state between the two networks (wrench) is physically meaningful, thus adding interpretability to the controller. 

## _1) Learning-based Control Allocation_ 

Learning the control allocation, compared to using the pseudo-inverse of the allocation matrix or solving an optimization problem, comes with three advantages. First, compared to pseudo-inverse methods that find a solution solving the unconstrained allocation [21], the learned control allocation can be trained to mimic an expert that accounts for constraints as described in this section. This is increasingly important for 

**==> picture [120 x 63] intentionally omitted <==**

**----- Start of picture text -----**<br>
Admissible<br>Wrench Set<br>Minimum Enclosing<br>S Bounding Box<br>**----- End of picture text -----**<br>


Fig. 4: Transformation of wrench commands. 

systems with a low thrust-to-weight ratio. Second, solving the constrained allocation optimization problem at each timestep during training slows down the training process (solving the optimization problem is significantly more expensive than the forward path through a network), which is especially problematic when large numbers of different systems have to be trained. Third, at runtime, performing inference through that network comes with predictable computation time as opposed to algorithms that iteratively solve for the constrained control allocation [22]. To learn the control allocation, we generate training data by sampling wrenches uniformly from a hypercube containing the admissible set of the configuration. Choosing the space from which the training data is sampled to be a hypercube is motivated by the fact that normalized commands (between _−_ 1 and 1) generated by the control network can be mapped directly (without solving a linear problem) onto the training envelope, guaranteeing the allocation network is not confronted with data unseen in training. Our experiments have shown that the achievable accuracy of the allocation network is highly dependent on the hyper-volume ratio between the chosen sample space and the admissible set. To minimize this ratio, we calculate an axis-aligned minimum enclosing bounding box that is aligned with the principal components of the vertices of the convex hull of the admissible set. During inference, the normalized commands of the controller network are scaled according to the limits of the bounding box in the frame _P_ defined by the principal components as shown in Fig. 4. Then, the commands are transformed from _P_ to the frame _S_ of the admissible wrench space. 

For generating the corresponding motor forces from the sampled wrenches, we formulate the control allocation as a slack-constrained quadratic program following [23], minimizing projection error and motor forces: 

**==> picture [184 x 38] intentionally omitted <==**

with _wd ∈_ R[6] being the commanded wrench, _umin ∈_ R _[n][m]_ and _umax ∈_ R _[n][m]_ being the minimum and maximum motor thrust and _s ∈_ R[6] being a slack variable. Introducing _s_ is necessary since it is not guaranteed that the commanded wrench is within the admissible wrench set. The optimization problem is solved for multiple data points in parallel using the qpth-solver [24] to speed up the data generation. The allocation network is a simple MLP with different layer sizes, network depth and activation function depending on whether constraints should be enforced. Since a network imitating the unconstrained control allocation QP is equivalent to the solution obtained with the pseudo-inverse, small layer sizes are sufficient and no non-linear activation functions are needed. 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED FEBRUARY, 2026 

4 

For unconstrained control allocation, a network with one layer and 32 neurons is chosen. Introducing constraints renders the problem highly non-linear and requires larger networks and non-linear activations. Therefore, a network with 3 layers with 100 neurons each and _Leaky ReLU_ as an activation function is chosen. 

## _2) Learned Position Controller_ 

For training the controller network, we formulate the control problem as a Markov Decision Process (MDP) using reinforcement learning. As input, the network receives the state representation _x_ where we replace the position with the position error _pe_ and the quaternions with the 6-D rotation representation ( _R_ 6 _D_ ) introduced in [25] to avoid double coverage of _SO_ (3) without introducing redundancy. This results in the observations _o_ = [ _pe, v, R_ 6 _D, ω_ ] _∈_ R[15] . 

The used reward is based on the following quantities: 

**==> picture [226 x 26] intentionally omitted <==**

**==> picture [236 x 11] intentionally omitted <==**

**==> picture [230 x 12] intentionally omitted <==**

The quantities _hp_ , _hv_ and _h_ Ω are terms based on the position offset from the provided reference, the linear velocity and the angular rates. The term _huabs_ is based on the difference between each commanded motor thrust and the thrust necessary for hovering ( _u_ ˆ). To incentivize smooth actions, we add temporal smoothness regularization terms and spatial smoothness terms to the policy loss as described in [26]. The quantity _hup_ is based on the alignment of the _z_ -axis of _B_ and the _z_ -axis of _W_ . This is calculated by rotating the corresponding unit vector _e_ 3 by the quaternion representing the orientation of the system _q_ and evaluating its alignment, where _hup_ = 0 for perfect alignment. _hforw_ is equivalently based on the alignment of the _x_ -axis of _W_ and _B_ . Note that strictly speaking, only _hp_ is relevant for the position tracking task and the other terms are only to ensure a desired behavior from a sim-to-real standpoint. The terms in equation 9 - 12 described quantities _h_ which are used to calculate the individual parts _rk_ of the reward function using exponential kernels as follows: 

**==> picture [162 x 23] intentionally omitted <==**

where the coefficients _a ∈_ R and _b ∈_ R of the exponential kernel are adequately chosen weights defining its magnitude and width and _k_ indicates the quantity the reward is based on ( _p, v,_ Ω _, ..._ ). The obtained individual reward terms are combined into the following cumulative reward: 

**==> picture [239 x 12] intentionally omitted <==**

Multiplying _rforw_ , _rv_ and _r_ Ω with _rp_ leads to a weighted reward allowing for high linear velocities, angular rates and orientation offsets far away from the target location. The controller network consists of two layers with 32 and 24 neurons with _tanh_ as activation function. Additionally, we employ _tanh_ -squashing after the last layer to bound the generated outputs to a [ _−_ 1 _,_ 1]-hypercube. During training, no specific trajectories are sampled. Instead, we sample an 

initial state of the system and the reward function penalizes deviations from the origin of the state space. 

## _C. Initializing Policies to Jump-Start Policy Learning_ 

To accelerate the training of a new configuration, we propose a library-based initialization scheme shown in Fig. 1 and Algorithm 1, maintaining a fixed number of candidate policies. For a new configuration defined by the parameter vector _cn_ , first, a network for control allocation is trained. To pick the initial weights for the controller network, all configurations _ck ∈L_ = _{c_ 1 _, ..., cL}_ in the library _L_ are evaluated regarding their suitability for initialization based on a similarity measure _m_ ( _ci, ck_ ). The policy weights of the configuration with the best score are picked as initial values for the training of the new configuration. Initializing the actor weights alone is not sufficient and will result in a poor learning signal provided by the untrained critic, as described in [9]. We therefore initialize the critic in the same way as the actor, similar to [14]. Additionally, we found that transferring the current optimizer state (first and second moments of the gradients for Adam) of the policy used for initialization resulted in significantly better performance. Subsequently, the configuration is finetuned until convergence. Our approach requires an efficient **Algorithm 1** Library-Based Initialization 

**Given:** Parameter vector of the configuration to train _c_ , library of already trained configurations _L_ , similarity measure m( _ci_ , _ck_ ), RL training method train( _c_ ,Θ, _γ_ ). 

initialize empty list _M_ 

**for** _ck_ in _L_ **do** Append _m_ ( _c, ck_ ) to _M ▷_ compute similarity measure 

**end for** 

|_cinit_ =_L_(arg max(_M_))<br>Θ = Θ_init_<br>_γ_ =_γinit_<br>train(_c_,Θ,_γ_)|_▷_pick confg for initialization<br>(arg min for _mc, mwd_)<br>_▷_initialize policy parameters<br>_▷_initialize optimizer states<br>_▷_train confg _c_ until convergence|
|---|---|



method for selecting a policy for initialization from the library. In this work, two avenues are explored to select a configuration for initialization: first, selecting a configuration based on a measure over the physical properties of the airframe, and second, evaluating the properties of the policy. This led to the 3 following similarity measures considered in this work: 

- 1) A straightforward choice for comparing the physical properties of two systems is to evaluate the norm of the difference of their augmented parameter vectors, like in [6]. The augmented parameter vector is defined as _caug_ = [ _c, vec_ ( **J** )] and the corresponding measure is defined as: _mc_ ( _caugi, caugk_ ) = _||caugi − caugk ||_ 2 _._ (15) 

- 2) The admissible set of linear and angular accelerations that a configuration can produce instantaneously closely relates to the interface between the policy and the system, while implicitly providing information on all physical properties of the airframe (mass, inertia, and motor position and orientation). However, comparing the admissible 

5 

REHBERG _et al._ : EFFICIENT KNOWLEDGE TRANSFER FOR JUMP-STARTING CONTROL POLICY LEARNING OF MULTIROTORS 

acceleration sets for two arbitrary configurations is not straightforward (e.g., collapsing overlapping volume of the convex hulls). Instead, we propose a measure based on the Wasserstein distance [27] between two distributions ( _X, Y_ ) representing the admissible acceleration sets ( _A_ ( _c_ )): 

**==> picture [224 x 44] intentionally omitted <==**

where the infimum is over all permutations _π_ with _n_ being the number of samples drawn from a uniform distribution bounded by the convex hull of the admissible set. Intuitively, the measure can be interpreted as the “effort” necessary to transform one set into the other. 

Fig. 5: Sampling range for sampling the configurations in the pool. With ° _lmin_ = 0 _._ 1 _m_ , _lmax_ = 0 _._ 35 _m_ , _γ_ = 60 and _ϕ_ = 20°. 

Fig. 6: Example configurations from the sampled library of configurations with two of their closely sampled configurations in greyscale (quadcopters left and hexacopters right). 

3) Instead of choosing a configuration based on the physical 

properties, we also explore directly evaluating the policies of the configurations in the library. To approximate how well a configuration is suited for initialization, we deploy its policy on the newly sampled configuration and evaluate the accumulated reward. Using modern massively parallelized simulators, this can be done for large numbers of configurations in parallel. The resulting measure can be formulated as: 

**==> picture [5 x 6] intentionally omitted <==**

**==> picture [223 x 35] intentionally omitted <==**

with _T_ being the number of evaluation time steps, _πck_ being the policy corresponding to the configuration defined by the parameter vector _ck_ , and _xinit_ being the initial state. 

## IV. EVALUATION 

To show the validity of our proposed initialization scheme and controller structure, we run multiple simulations and realworld experiments. First, we perform extensive simulation studies to evaluate the efficiency of our library-based initialization scheme and how it leads to computational improvements. Subsequently, we evaluate the accuracy of the learned control allocation networks by comparing them to the QP-expert used for training. Finally, we demonstrate the proposed controller structure, consisting of the controller network and the allocation network, in real-world scenarios. 

## _A. Evaluation of the Library-based Initialization Scheme_ 

The experiments to evaluate the proposed initialization scheme are twofold. First, the correlation of the similarity measures with the number of saved environment interactions are evaluated to decide which one should be used in the initialization scheme. Second, the initialization scheme based on the selected similarity measure is evaluated. 

To evaluate the proposed initialization scheme, a library of 40 configurations is sampled with 4 motors and with 6 motors, while a control policy providing wrench commands to an unconstrained control allocation network is trained 

for each configuration from scratch until convergence for 3 different seeds. We choose to evaluate the initialization scheme using unconstrained networks because training the constrained control allocation takes significantly more time. The motor positions are uniformly sampled from a cone stump centered at the nominal motor position of the standard quad- and hexacopter. The two faces are defined by the maximum and minimum arm length ( _lmin_ , _lmax_ ), while the cone angle _γ_ defines the angle limits of the arm. The motor orientations are uniformly sampled such that the motor _z_ -axis does not deviate from the body _z_ -axis by more than _ϕ_ . This is illustrated by Fig. 5. The high-dimensional sampling space induced by the large number of morphology parameters implies that dense sampling across all of it is computationally expensive. For the purposes of the presented evaluation, we restrict the sampling space to a localized region to obtain meaningfully close configurations for initialization. In order to still be able to evaluate diverse airframes, we sample 3 additional configurations uniformly from a distribution centered around each of the 40 configurations, allowing for a maximum deviation of 5° in motor angle and 5 _cm_ in motor position (examples shown in Fig. 6). Those configurations are added to the library (160 configurations in total). Compared to simulating the real system, the configurations in the pool are simulated with a lower fidelity simulation. This includes calculating the system’s inertia from a point mass model and removing sensor noise and the motor model from the simulation. Additionally, the reward function is condensed to include only terms that matter for the task of position control (purely based on the position error). The lower fidelity simulation and only using position terms facilitates using the same reward function across the whole system family. Additionally, we check that the sampled configurations are able to hover and have control authority around all axes. Configurations that are still not able to learn in any of the tried seeds are discarded after training. 

## _1) Similarity Measures_ 

We assess the effectiveness of the proposed similarity measures in predicting whether a configuration is suitable for initialization by examining the correlation between the similarity measure and the change in the number of environment interactions necessary to reach a goal reward. To this end, we 

6 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED FEBRUARY, 2026 

TABLE I: Statistical evaluation of the similarity metric. 

|**Similarity**<br>**Measure**<br>_mc_<br>_mwd_|**Abs. Corr. Coefficient**<br>_Quad_<br>_Hex_<br>0_._11<br>0_._23<br>039<br>010|_p_**-value**<br>_Quad_<br>_Hex_<br>0_._54<br>0_._26<br>003<br>065|
|---|---|---|



retrain every configuration in the pool while initializing the training with randomly picked configurations from the other configurations in the pool. We then compute the Spearman correlation coefficient and corresponding _p_ -value to assess the relationship between the similarity measure calculated for two configurations and the resulting change in required environment interactions when one is used to initialize the other. Since with a flattening reward curve, the randomness in the training process has an increasing influence on the exact number of saved environment interactions, the correlation becomes less strong. We therefore evaluate the correlation and statistical significance at multiple reward goals over the full reward range and report the values for the highest reward a _p_ -value _≤_ 0 _._ 01 was obtained to ensure high significance. The corresponding results are summarized in Table I. The results show that the two measures based on evaluating physical properties of the configuration correlate less well than the measure _mr_ based on evaluating the policy directly. Note that the _p_ -values of the evaluations of _mc_ and _mwd_ show no statistical significance and have a smaller correlation coefficient over the whole range. 

## _2) Initialization Scheme_ 

Subsequently, each configuration in the pool is retrained while picking a policy for initialization based on the rewardbased similarity measure. All training runs were repeated using 3 different seeds. To show that choosing a configuration based on the proposed similarity measure is important, we compare our library-based initialization scheme to randomly choosing configurations from the library. This shows that the constraints on the configuration space we impose do not render the configurations so close to each other that picking arbitrary configurations will yield similar results as using the proposed metric. Additionally, we conduct experiments initializing only from the original (sparse) library, not containing close neighboring configurations and initializing from policies trained for a different motor number. In Fig. 7, the median of the reward curves for all systems and the mean of the standard deviation over seeds for each configuration are reported. The experiments show that the proposed initialization scheme reduces required environment interactions by _≈_ 71 _/_ 76% (quadrotor/hexarotor) with close configurations, _≈_ 50 _/_ 65% with the sparse library, and _≈_ 49 _/_ 65% when using a policy from a system with a different motor number. Note that computing the similarity measure increases the environment interactions (accounted for in the figure), but choosing a system afterward is trivial. Fig. 7 shows a significant difference when selecting a random configuration for initializing the hexacopter training. This difference can be attributed to the greater number of randomly sampled motors, which, on average, results in an admissible set that more uniformly covers the wrench space. As a result, the overlap between the admissible sets of the two configurations increases, facilitating smoother initialization. 

**==> picture [191 x 191] intentionally omitted <==**

**----- Start of picture text -----**<br>
Quadcopter<br>18000<br>15000<br>10000<br>5000<br>1000 /~ YN [Ss<br>2 4 6 8<br>Hexacopter × 10 [7]<br>18000<br>15000<br>10000 HAR<br>5000<br>1000<br>2 4 6 8<br># Environment Interactions × 10 [7]<br>From Scratch Initialized (Random)<br>Initialized (Similarity Measure)<br>Initialized (Similarity Measure - Sparse Library)<br>Initialized (Similarity Measure - Cross Library)<br>Reward<br>Reward<br>**----- End of picture text -----**<br>


Fig. 7: Results of the policy initialization experiments. The blue line shows training from scratch, the green line shows initialization with a random configuration, the red line shows initialization using the proposed similarity measure, selecting policies from the original 40 configurations without the added close neighbors and the orange line shows initialization using the similarity measure, selecting policies from the full library of systems, including close neighbors. The violet line shows initialization using the proposed measure, picking policies for initialization from the library of systems with another motor number. 

## _B. Evaluation of the Learned Control Allocation_ 

In the following, the learned control allocation is compared to the QP-expert solving the constrained and unconstrained control allocation. For evaluation, 3 configurations having 4 motors and 3 configurations having 6 motors were used. For both motor numbers, the standard configuration (all thrust axes parallel), a configuration where all motors are tilted by 10 _[◦]_ outwards, and a random configuration were chosen. Table II shows the average and maximum errors normalized to the thrust range between the output of the learned allocation model and the solution to the quadratic problem over 1 _e_ 5 random samples. In practice, this gives an intuition of how close the outputs of the controller network will be to physically meaningful wrenches. Since the unconstrained control allocation is a linear mapping, the network is able to reproduce the expert outputs with very high precision. The constrained control allocation, on the other hand, is a nonlinear relation requiring more complex networks. Our experiments show that the accuracy of the learned control allocation drops for higher motor numbers. This is likely due to the higher-dimensional output space (reflecting the actual control allocation of the system) and, therefore, more complex mapping. Learning the constrained control allocation can still be beneficial for arbitrary designs since it allows for constraint-aware allocation and, therefore, redistribution of effort in the case of saturation. The observed errors, particularly during training of the constrained control allocation network, do not hinder the controller network, as it learns to compensate for these inaccuracies by adapting its control strategy accordingly. Additionally, sensitivity curves of the closed-loop tracking performance of a frozen control network with respect to the allocation error are shown in Fig. 8. It is observed that even without allowing the network 

REHBERG _et al._ : EFFICIENT KNOWLEDGE TRANSFER FOR JUMP-STARTING CONTROL POLICY LEARNING OF MULTIROTORS 

7 

TABLE II: Maximum ( _|e|_ max) and average ( _|_[¯] _e|_ ) absolute errors of the control allocation normalized to the thrust range 

**==> picture [233 x 239] intentionally omitted <==**

**----- Start of picture text -----**<br>
Constrained Unconstrained<br>|e| max, | [¯] e| |e| max, | [¯] e|<br>Quadrotor 2 . 7e − 2, 1 . 2e − 3 4 . 4e − 5, 5 . 2e − 6<br>TiltedQuadrotorPropellerswith(10 [◦] ) 2 . 5e − 2, 1 . 3e − 3 4 . 0e − 5, 5 . 0e − 6<br>Random Quadrotor 4 . 3e − 2, 1 . 8e − 3 8 . 4e − 5, 8 . 1e − 6<br>Hexarotor 9 . 0e − 2, 6 . 0e − 3 5 . 9e − 4, 1 . 4e − 5<br>TiltedHexarotorPropellerswith(10 [◦] ) 15 . 0e − 1, 7e − 3 1 . 7e − 3, 3 . 8e − 4<br>one Random Hexarotor 11 . 0e − 1, 4e − 3 8 . 7 e− 5, 7 . 4e − 6<br>Hex (Non-Planar) - Unconstrained Hex - Unconstrained<br>5 . 0 Mean 5 . 0 Mean<br>Max Max<br>2 . 5 2 . 5<br>0 . 0 0 . 0<br>10 [−] [5] 10 [−] [3] 10 [−] [1] 10 [−] [5] 10 [−] [3] 10 [−] [1]<br>Quad - Constrained Quad - Unconstrained<br>5 . 0 Mean 5 . 0 Mean<br>Max Max<br>2 . 5 2 . 5<br>0 . 0 0 . 0<br>10 [−] [5] 10 [−] [3] 10 [−] [1] 10 [−] [5] 10 [−] [3] 10 [−] [1]<br>Normalized | [¯] e| Normalized | [¯] e|<br>[m]<br>¯ pe<br>[m]<br>¯ pe<br>**----- End of picture text -----**<br>


Fig. 8: Closed-loop trajectory tracking error with respect to the mean absolute allocation error normalized to the thrust range of the networks. 

to adapt to the allocation error, there is a significant margin before the allocation error significantly influences the closedloop tracking error. 

## _C. Evaluation of the Proposed Controller Architecture_ 

We train our RL policies with Proximal Policy Optimization (PPO) [28] using a customized version of Sample-Factory [29] and simulate the system in the Aerial Gym Simulator [30]. All policies are trained on a Lenovo ThinkPad P1 Gen 6 from 2024 with a GeForce RTX 4090 GPU and deployed on a system with a ModalAI Voxl 2 Mini board and a ModalAI Voxl ESC. To guarantee robust sim-to-real transfer, we incorporate sensor noise and highly accurate approximations of the robot’s inertia obtained from the CAD model and the true robot mass into the simulator. We additionally simulate the motor dynamics using a first-order model with a time constant obtained from RPM step-response experiments on the real system. All policies are trained with a physics time step size of 0 _._ 01 _s_ and deployed with a control frequency of 250Hz. The trained networks were deployed on the compute board with a custom PX4 module. We evaluate the proposed controller architecture in 2 different experiments. First, we deploy a trained policy to track a Lissajous trajectory with a loop time of 5 _._ 5 _s_ for a planar hexacopter (14 _cm_ arm length and a total weight of 421 _g_ ) and quadcopter (23 _cm_ arm length and a total weight of 373 _g_ ) configuration and a non-planar system from the library (14 _cm_ arm length on average and a total weight of 402 _g_ ) with an unconstrained allocation network. Note that we slightly adjust the non-planar system’s arm angles and lengths to avoid selfcollision and fit the arms to the body of an existing frame. For dynamic trajectories, the network receives the velocity error to a setpoint instead of the system velocity. The resulting 

**==> picture [207 x 225] intentionally omitted <==**

**----- Start of picture text -----**<br>
Hex<br>0 . 5 Setpoints<br>1 . 75<br>0 . 0<br>− 0 . 5<br>1 . 50<br>Hex (Non-Planar)<br>1 . 25<br>0 . 5<br>0 . 0 1 . 00<br>− 0 . 5<br>0 . 75<br>Quad<br>0 . 50<br>0 . 5<br>0 . 0<br>0 . 25<br>− 0 . 5<br>− 1 0 1<br>x [m]<br>[m]<br>y<br>[m/s]<br>[m]<br>y<br>Velocity<br>[m]<br>y<br>**----- End of picture text -----**<br>


Fig. 9: Results of the real-world trajectory tracking experiments using an unconstrained control allocation network. 

trajectories are shown in Fig. 9. Subsequently, we deploy a trained policy for a standard quadcopter configuration with a constrained allocation network in a static setpoint tracking task (shown in Fig. 10). Fig. 10 shows a constant offset in the commanded thrust values, which is not seen in simulation. This is likely due to unmodeled shifts in the center of mass or differing thrust coefficients between motors. We achieve state-of-the-art [30] steady state position error _≤_ 0 _._ 055 _m_ and a mean trajectory tracking error of 0 _._ 26 _m_ for the quadrotor, 0 _._ 17 _m_ for the planar hexacopter and 0 _._ 24 _m_ for the non-planar hexacopter. 

## V. CONCLUSION AND FUTURE WORK 

In this work, we presented a library-based policy initialization scheme aimed at accelerating the training of control policies for diverse multirotor configurations. By leveraging a physics-aware, modular control architecture and introducing a policy evaluation-based similarity measure, we enable the efficient cross-embodiment transfer of previously trained policies to jump-start learning for novel designs. Our results in simulation and the real world a) demonstrate substantial improvements in sample efficiency and training time and b) showcase that the proposed control architecture is capable of state-of-the-art performance. We further show that the similarity measure based on policy behavior significantly outperforms those based purely on physical configuration parameters in predicting initialization success. This highlights the importance of task-specific evaluation criteria for policy transfer. In the future, the presented work could be extended by initializing navigation policies for systems with differently placed exteroceptive sensors. 

## REFERENCES 

- [1] A. Pretto, S. Aravecchia, W. Burgard, N. Chebrolu, C. Dornhege, T. Falck, F. Fleckenstein, A. Fontenla, M. Imperoli, R. Khanna _et al._ , 

IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED FEBRUARY, 2026 

8 

**==> picture [271 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
Position Motor<br>Error [ m ] Velocity [ [m] s []] Thrust [ N ]<br>—<br>2<br>1 . 25<br>1 px<br>py 1 1 . 00<br>pz<br>0 0 0 . 75 u 1<br>vx u 2<br>− 1 − 1 vy 0 . 50 u 3<br>vz u 4<br>− 2 0 . 25<br>0 20 40 0 20 40 0 20 40<br>Time [ s ] Time [ s ] Time [ s ]<br>**----- End of picture text -----**<br>


Fig. 10: Left: results of the real-world experiments using a constrained control allocation network. Right: systems used for real-world experiments ( a) planar hexarotor, b) non-planar hexarotor, c) planar quadrotor). 

“Building an aerial–ground robotics system for precision farming: an adaptable solution,” _IEEE Robotics & Automation Magazine_ , vol. 28, no. 3, pp. 29–49, 2020. 

- [2] T. H. Chung, V. Orekhov, and A. Maio, “Into the robotic depths: Analysis and insights from the darpa subterranean challenge,” _Annual Review of Control, Robotics, and Autonomous Systems_ , vol. 6, no. 1, pp. 477–502, 2023. 

- [3] M. Dharmadhikari and K. Alexis, “Semantics-aware predictive inspection path planning,” _IEEE Transactions on Field Robotics_ , 2025. 

- [4] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. M¨uller, V. Koltun, and D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” _Nature_ , vol. 620, no. 7976, pp. 982–987, Aug. 2023. [Online]. Available: https://www.nature.com/articles/s41586-023-06419-4 

- [5] A. Gupta, S. Savarese, S. Ganguli, and L. Fei-Fei, “Embodied Intelligence via Learning and Evolution,” _Nature Communications_ , vol. 12, no. 1, p. 5721, Oct. 2021. [Online]. Available: http: //arxiv.org/abs/2102.02202 

- [6] P. Mannam, X. Liu, D. Zhao, J. Oh, and N. Pollard, “Design and Control Co-Optimization for Automated Design Iteration of Dexterous Anthropomorphic Soft Robotic Hands,” in _2024 IEEE 7th International Conference on Soft Robotics (RoboSoft)_ , Apr. 2024, pp. 332–339, iSSN: 2769-4534. [Online]. Available: https: //ieeexplore.ieee.org/document/10521927/ 

- [7] J. H. Park and K. H. Lee, “Computational design of modular robots based on genetic algorithm and reinforcement learning,” _Symmetry_ , vol. 13, no. 3, 2021. [Online]. Available: https://www.mdpi.com/ 2073-8994/13/3/471 

- [8] R. Julian, B. Swanson, G. S. Sukhatme, S. Levine, C. Finn, and K. Hausman, “Never Stop Learning: The Effectiveness of Fine-Tuning in Robotic Reinforcement Learning,” _4th Conference on Robot Learning (CoRL 2020), Cambridge MA, USA_ . 

- [9] I. Uchendu, T. Xiao, Y. Lu, B. Zhu, M. Yan, J. Simon, M. Bennice, C. Fu, C. Ma, J. Jiao, S. Levine, and K. Hausman, “Jump-Start Reinforcement Learning,” _Proceedings of the 40 th International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023_ , 2023. 

- [10] J. Eschmann, D. Albani, and G. Loianno, “Raptor: A foundation policy for quadrotor control,” 2025. [Online]. Available: https: //arxiv.org/abs/2509.11481 

- [11] L. Chen, K. Dharmarajan, K. Hari, C. Xu, Q. Vuong, and K. Goldberg, “MIRAGE: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting,” in _Robotics: Science and Systems XX_ . Robotics: Science and Systems Foundation, Jul. 2024. [Online]. Available: http://www.roboticsproceedings.org/rss20/p069.pdf 

- [12] T. Wang, D. Bhatt, X. Wang, and N. Atanasov, “Cross-embodiment robot manipulation skill transfer using latent space alignment,” 2024. [Online]. Available: https://arxiv.org/abs/2406.01968 

- [13] K. S. Luck, H. B. Amor, and R. Calandra, “Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning,” _3rd Conference on Robot Learning (CoRL 2019), Osaka, Japan_ . 

- [14] C. Chen, J. Yu, H. Lu, H. Gao, R. Xiong, and Y. Wang, “Pretraining-finetuning Framework for Efficient Co-design: A Case Study on Quadruped Robot Parkour,” Sep. 2024, arXiv:2407.06770 [cs]. [Online]. Available: http://arxiv.org/abs/2407.06770 

- [15] X. Liu, D. Pathak, and K. M. Kitani, “REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer,” _Proceedings of the 39th International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022_ . 

- [16] L. K. Le Goff, E. Buchanan, E. Hart, A. E. Eiben, W. Li, M. De Carlo, A. F. Winfield, M. F. Hale, R. Woolley, M. Angus, J. Timmis, and A. M. Tyrrell, “Morpho Evolution With Learning Using a Controller 

Archive as an Inheritance Mechanism,” _IEEE Transactions on Cognitive and Developmental Systems_ , vol. 15, no. 2, pp. 507–517, Jun. 2023. [Online]. Available: https://ieeexplore.ieee.org/document/9701596/ 

- [17] D. Zhang, A. Loquercio, J. Tang, T.-H. Wang, J. Malik, and M. W. Mueller, “A learning-based quadcopter controller with extreme adaptation,” _IEEE Transactions on Robotics_ , vol. 41, p. 3948–3964, 2025. [Online]. Available: http://dx.doi.org/10.1109/TRO.2025.3577037 

- [18] J. Eschmann, D. Albani, and G. Loianno, “Learning to Fly in Seconds,” _IEEE Robotics and Automation Letters_ , vol. 9, no. 7, pp. 6336–6343, Jul. 2024. [Online]. Available: https: //ieeexplore.ieee.org/document/10517383/ 

- [19] J. Hwangbo, I. Sa, R. Siegwart, and M. Hutter, “Control of a Quadrotor with Reinforcement Learning,” _IEEE Robotics and Automation Letters_ , vol. 2, no. 4, pp. 2096–2103, Oct. 2017, arXiv:1707.05110 [cs]. [Online]. Available: http://arxiv.org/abs/1707.05110 

- [20] R. Penicka, Y. Song, E. Kaufmann, and D. Scaramuzza, “Learning Minimum-Time Flight in Cluttered Environments,” _IEEE Robotics and Automation Letters_ , vol. 7, no. 3, pp. 7209–7216, Jul. 2022. [Online]. Available: https://ieeexplore.ieee.org/document/9794627/ 

- [21] M. Tognon and A. Franchi, “Omnidirectional Aerial Vehicles With Unidirectional Thrusters: Theory, Optimal Design, and Control,” _IEEE Robotics and Automation Letters_ , vol. 3, no. 3, pp. 2277–2282, Jul. 2018. [Online]. Available: http://ieeexplore.ieee.org/document/8281444/ 

- [22] S. P. Boyd and L. Vandenberghe, _Convex optimization_ , version 29 ed. Cambridge New York Melbourne New Delhi Singapore: Cambridge University Press, 2023. 

- [23] T. A. Johansen and T. I. Fossen, “Control allocation—A survey,” _Automatica_ , vol. 49, no. 5, pp. 1087–1103, May 2013. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S0005109813000368 

- [24] B. Amos and J. Z. Kolter, “Optnet: Differentiable optimization as a layer in neural networks,” in _International conference on machine learning_ . PMLR, 2017, pp. 136–145. 

- [25] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the Continuity of Rotation Representations in Neural Networks,” Jun. 2020, arXiv:1812.07035 [cs, stat]. [Online]. Available: http://arxiv.org/ abs/1812.07035 

- [26] S. Mysore, B. Mabsout, R. Mancuso, and K. Saenko, “Regularizing Action Policies for Smooth Control with Reinforcement Learning,” in _2021 IEEE International Conference on Robotics and Automation (ICRA)_ . Xi’an, China: IEEE, May 2021, pp. 1810–1816. [Online]. Available: https://ieeexplore.ieee.org/document/9561138/ 

- [27] V. M. Panaretos and Y. Zemel, “Statistical Aspects of Wasserstein Distances,” _Annual Review of Statistics and Its Application_ , vol. 6, no. 1, pp. 405–431, Mar. 2019, arXiv:1806.05500 [stat]. [Online]. Available: http://arxiv.org/abs/1806.05500 

- [28] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” Aug. 2017. [Online]. Available: http://arxiv.org/abs/1707.06347 

- [29] A. Petrenko, Z. Huang, T. Kumar, G. Sukhatme, and V. Koltun, “Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning,” _Proceedings of the 37 th International Conference on Machine Learning, Online, PMLR 119, 2020_ . 

- [30] M. Kulkarni, W. Rehberg, and K. Alexis, “Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots,” _IEEE Robotics and Automation Letters_ , vol. 10, no. 4, pp. 4093–4100, Apr. 2025. [Online]. Available: https://ieeexplore.ieee.org/document/ 10910148/ 

