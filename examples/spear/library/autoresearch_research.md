# Research Memo — Ariel Drone MTRL (2026-08-03)
Baseline: 32.374 (exp_080). hover=64.5, fig8=24.0, slalom=26.8, shuttle=28.1, circle=34.5.
Target: 35+. Gap: ~2.6 metric points.

---

## Section 1: Paper Impl Re-assessment at 100M Steps

**P1 — Song 2021 (progress reward), lateral dev -68% at 500k steps**
Worth re-running at full scale: YES, high priority.
The 500k result on a gate-racing env already shows -68% lateral deviation, which is
a meaningful signal even before the MTRL context. At 100M steps (200x more), the
trunk would have time to learn finer trajectory shaping beyond raw deviation reduction.
The current fig8=24.0 is the weakest task and its failure mode is drift ~0.14m from
the gate center — exactly what progress-projection fixes. The 500k result likely
understates the gain because (a) the trunk hasn't converged on slalom/circle yet, and
(b) the lateral penalty term takes many iterations to propagate through value
bootstrapping. Risk: k_lateral tuning — too high and it penalises the intentional
lateral motion in slalom/circle gate approaches.

**P2 — Kaufmann 2023 (Swift track curriculum), 4% at 80k steps**
Worth re-running: MAYBE, low priority for MTRL metric.
4% at 80k is not informative — Swift needs 10M+ steps to show curriculum benefits
(the Nature paper trained >1 billion simulator steps). The correct comparison is
against a fixed-track baseline at 5M steps. However, P2 tests a gate-racing single
policy, not the 5-task MTRL setting; gains there do not transfer directly to the
hover/slalom/shuttle tasks that dominate the metric. Re-run only if fig8 per-task
reward is confirmed as the binding constraint after P1 changes.

**P3 — Molchanov 2019 (morph randomization), metric -5.57 at 400k**
Worth re-running: NO at this stage.
The -5.57 at 400k is damning: morph randomization adds variance to the trunk at the
worst possible time (before the MTRL policy has learned stable baselines). At 100M
steps the trunk may eventually recover, but exp_080 already generalises across 90
morphologies without randomization. Unless azimuth tolerance is the failure mode
(it is not — hover=64.5 is healthy), morph randomization is lower priority than
task-interference fixes.

**P4 — Johannink 2019 (alpha ablation), validates alpha=0.40 fig8 / 0.10 hover**
Not worth re-running: SKIP.
The ablation already converged. The autoresearch_strategy.md notes the remaining
alpha lever is hover 0.10→0.03 and circle prior restoration — those are direct
extensions of P4's finding and should be coded as hyperparameter sweeps in
37_train_residual_mtrl.py, not as a repeat of the P4 harness.

**P5 — PCGrad, worse than baseline at 200k, conflict fraction 0.44-0.50**
Worth re-running: MAYBE, but first try CAGrad instead.
200k is far too short for PCGrad to help: the trunk has not stabilised, so the
gradient projections are noisy and the normalisation by num_tasks cuts the effective
step size. The 0.44-0.50 conflict fraction is high but not unusual for early PPO
training; it would need re-measurement at 50M+ to be diagnostic. At 100M, PCGrad
might be neutral-to-small-positive, but the literature (see "Fantastic Multi-Task
Gradient Updates" arXiv 2502.00217) suggests CAGrad outperforms PCGrad on RL tasks
with similar conflict rates. CAGrad's convergence guarantee on average loss is also
better suited to our metric formula.

**P6 — CARE context attention, worse at 200k, attention near-uniform**
Worth re-running: NO.
Near-uniform attention at 200k is the expected initialisation outcome — the context
MLP hasn't learned task-discriminative features yet. But 100M steps would give it
time. The concern is that the CARE architecture adds ~5% parameter overhead and a
second backward pass, and our shared-trunk interference is driven by per-task
gradient scale mismatches (slalom/circle are sparser rewards), not by the hard
one-hot gating. CARE addresses gating flexibility, not gradient scale. FiLM
conditioning (see GEAR paper, arXiv 2602.10997) is a lighter-weight alternative
that may outperform CARE here.

---

## Section 2: New Research Proposals

### P7 — CAGrad Actor Gradient Surgery (replaces PCGrad re-run)
**Mechanism.** CAGrad (Liu et al. NeurIPS 2021) finds the gradient update that
minimises average task loss while maximising the worst-task improvement. Unlike
PCGrad which projects pairwise, CAGrad solves a small QP per update and has a
convergence guarantee. With conflict fraction 0.44-0.50 and fig8 as the consistent
casualty, CAGrad's focus on worst-task improvement is directly applicable.
Implementation is ~60 lines: subclass PCGradPPO and replace `_pcgrad_project` with
the CAGrad QP (frank-wolfe iteration or scipy.optimize.minimize on 5 vars). Reuse
the existing per-task gradient computation from p5 exactly.
**Expected per-task effect.** fig8: +1-3 (worst-task protection); slalom/circle:
neutral-to-+1; hover: neutral (prior-stabilised, less gradient noise).
**Implementation sketch.** In p7_liu2021_cagrad.py, import p5's PCGradPPO,
override `_pcgrad_project` with CAGrad QP (5-task, c=0.5). Flag `--cagrad-c`.
~80 lines new code.
**Risk.** QP solver adds wall-clock overhead per minibatch (~5ms on CPU with 5 tasks).
If the 5-task QP diverges, fall back to PCGrad. c hyperparameter needs a screen.

### P8 — FiLM Task Conditioning in the Shared Trunk
**Mechanism.** Feature-wise Linear Modulation (Perez et al. 2018, used in GEAR
arXiv 2602.10997) injects task conditioning into the trunk via per-layer affine
transforms: `h_l = gamma_l(task_embed) * h_l + beta_l(task_embed)`. This is lighter
than CARE's mixture-of-encoders and does not change the forward pass shape. The
key insight: trunk features for slalom and fig8 are currently forced to co-exist in
the same activation space; FiLM gives each task its own "normalization lane" in each
layer without per-task actor heads (which catastrophically failed in exp_068).
Task embed = existing one-hot (4 dims per task), gamma/beta MLPs are 2-layer, 32
hidden. Add to MTRLActorCriticPolicy.shared_trunk forward pass.
**Expected per-task effect.** fig8: +2-4 (task-specific trunk activations break the
slalom/circle interference found in vel-coef sweeps exp_072/074); slalom: +1-2;
hover: neutral (prior-dominant).
**Implementation sketch.** In p8_film_trunk.py, subclass MTRLActorCriticPolicy,
wrap each trunk layer in a FiLMLayer(nn.Module). ~100 lines. Flag `--film {on,off}`.
**Risk.** Adds 2*num_tasks*trunk_width parameters per layer. If trunk_width is large,
gamma/beta may overfit morph-specific features and hurt generalisation. Check trunk
width in 37_train before implementing.

### P9 — Circle Prior Restoration + Hover Alpha Sweep (direct strategy extension)
**Mechanism.** autoresearch_strategy.md already identifies these as the top two
levers. This proposal formalises them as a single paired experiment:
(a) extend the CMA-ES prior to circle with alpha=0.45, k_tilt*0.3 (same recipe as
exp_080 fig8 fix); (b) simultaneously sweep hover alpha: 0.10→0.05→0.03 in a
3-cell screen. These are not paper impls but are the highest-confidence path to +2-3
metric points based on the trajectory of exp_044→080.
**Expected per-task effect.** circle: +3-5 (prior stabilises trunk, analogous to
exp_080's +2.3 from fig8-prior); hover: +0.5-1.5 from alpha cut; slalom: +0.5
co-benefit via shared trunk stabilisation.
**Implementation.** Pure hyperparameter change in 37_train_residual_mtrl.py.
No new file needed. Screen at 20M, confirm at 100M if screen passes.
**Risk.** exp_039/040/047 show that alpha > 0.40 on slalom/circle destabilises trunk.
Circle alpha must stay ≤ 0.45. If circle prior at 0.45 collapses slalom (as higher
circle vel coefs did in exp_074), back off to alpha=0.30.

### P10 — Asymmetric Entropy Annealing Per Task Group
**Mechanism.** Current ent_start=0.02 is applied uniformly. Hover is prior-stabilised
and has low entropy need; slalom/circle still need exploration to find gate-passing
trajectories. Warmup hover entropy faster (anneal 0.02→0 by 20M) while keeping
trajectory tasks at 0.02 until 60M. Implemented as a custom EntCoefAnneal callback
that reads task_id from the env and scales ent_coef per minibatch task mask.
**Expected per-task effect.** slalom: +1-2 (more exploration budget retained later);
hover: neutral-to-+0.5 (faster convergence to prior-matched action); fig8: +0.5.
**Implementation sketch.** ~50 lines in 37_train_residual_mtrl.py or a new callback
p10_task_entropy.py. Requires access to per-task minibatch masks (already computed
in PCGradPPO's minibatch splitting).
**Risk.** Per-task ent_coef requires hooking into SB3's PPO train() loop. The 0.02
sweet spot was found on the aggregate; per-task annealing may hit a different regime.
Screen at 20M is essential before committing.

---

## Section 3: What Short Budget Runs Missed

**200k-500k results systematically understate trajectory task performance.**
The cosine LR schedule only reaches ~2% of its range at 200k steps (LR still near
3e-4). fig8 and circle both show late-training gains (exp_069 found fig8 jumped from
14.5 to 22.1 in the final 30M of a 100M run). Short runs measure early-phase
exploration, not converged behaviour.

**PCGrad (P5) conflict fraction 0.44-0.50 is a red herring at 200k.**
At 200k the trunk is still in a high-entropy exploration phase; gradient conflicts are
universal, not diagnostic of task interference. The relevant measurement is conflict
fraction at 50M+ when each task has a stable value function. Dismissing PCGrad as
"worse than baseline" based on 200k is likely wrong in the long run.

**CARE (P6) near-uniform attention is expected, not a failure.**
The context MLP needs ~5-10M steps to develop task-discriminative representations.
The 200k result tells us nothing about CARE's ceiling, only that it doesn't give a
free lunch in the first 200k steps.

**The screen→confirm gap is real and asymmetric (see strategy doc).**
The 20M screen systematically underpredicts circle and fig8 gains (confirmed by
exp_080: screen said circle=-3.8, confirm said circle=+0.3). Any paper impl that
shows circle/fig8 regression at 200k should not be dismissed without a longer run.

**Short budget runs cannot detect trunk-stabilisation effects.**
exp_080's +2.3 gain came from restoring the fig8 prior, which stabilises the shared
trunk for ALL tasks. This effect takes 30-50M steps to propagate. P1's lateral
deviation reduction at 500k likely understates its eventual metric effect for the
same reason — the slalom and circle co-benefit from a better-shaped fig8 trajectory
signal would only appear at 20M+.

**Practical implication.** Do not schedule any paper impl run at less than 20M steps
on the MTRL 5-task setting (10 envs, 90 morphs). The 200k budget is only valid for
smoke tests (does it crash? does loss decrease?). Use 20M as the screen threshold
and 100M as the confirm threshold, consistent with the strategy doc's rules.

---

## Section 4: New Literature Proposals (2026-08-04) — Post-gamma-failure bottleneck

Bottleneck as of exp_082 confirm: gamma=0.999 killed slalom (-24.8); gamma path is
permanently closed. fig8=24.0 and slalom=26.8 are the weakest tasks. Cross-task
interference via shared trunk remains the primary mechanistic gap (vel-coef sweeps
exp_072/074 confirmed: any single-task increase destroys another). The following
methods were identified as the highest-priority unimplemented options.

### PopArt Reward Normalisation (Hessel et al., AAAI 2019, arXiv 1809.04474)
- Mechanism: Maintains a running mean/std of each task's episodic return and
  rescales both the critic target and the output layer weights so that the value
  network always operates in a normalised (-1,+1) range per task. This means
  hover (returns ~65) and slalom (returns ~27) present equal-magnitude gradients
  to the shared critic, removing reward-scale-induced dominance.
- Why it fits our bottleneck: hover returns are ~2.4× higher than slalom/fig8.
  This biases the shared critic toward hover-shaped value landscapes, which may
  suppress gradient flow for trajectory tasks. exp_073 showed vf_coef reduction
  kills slalom — the correct fix is not scaling the coefficient but normalising
  the targets so the critic doesn't over-fit hover scale.
- Implementation sketch: In `37_train_residual_mtrl.py` or a custom SB3 callback,
  track per-task running mean/std of returns (5 exponential moving averages).
  In the critic loss, normalise each task's TD target: `y_norm = (y - mu_t) / sigma_t`.
  Preserve unnormalised policy gradient (actor side unchanged). ~40 lines in a
  `PopArtValueNorm` callback hooked into `on_rollout_end`. No architecture change.
- Risk: SB3 PPO does not expose the per-task critic target directly; requires
  hooking into `train()` or subclassing `PPO`. If the EMA sigma collapses to near-
  zero during early hover convergence, normalisation diverges — add a sigma floor.

### TOPPO Critic Balancing (Li et al., arXiv 2605.11473, 2026)
- Mechanism: Identifies "critic-side gradient ill-conditioning" in multi-task PPO
  where easy tasks (hover, here) dominate the shared value function's gradient,
  causing tail tasks to stall. Introduces per-task gradient rescaling on the critic
  loss via a balancing module that equalises gradient norms across tasks before the
  shared critic backward pass. On-policy; no off-policy replay needed.
- Why it fits our bottleneck: exp_079 showed clip_range_vf=0.2 hurt fig8 and
  shuttle by over-constraining their value updates. TOPPO addresses the same root
  cause (critic imbalance) without clipping — it rescales rather than clips, so
  trajectory tasks retain full gradient magnitude while hover's dominance is
  suppressed. The paper shows PPO can match SAC on MTRL, which is directly
  relevant to our SB3 PPO setup.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass SB3's PPO and
  override `train()`. After computing per-task critic losses, compute per-task
  gradient norms, form a balancing weight vector (softmax of inverse norms), and
  re-weight critic losses before `.backward()`. ~50 lines. Flag `--toppo-balance`.
- Risk: The balancing weight is computed per-minibatch; if a task's minibatch is
  small (e.g., hover gets fewer episodes in a rollout), the norm estimate is noisy.
  Use a 10-step EMA on per-task norms for stability.

### EPPO Entropy-Paced Clipping (Hu et al., arXiv 2607.07178, 2026)
- Mechanism: Replaces PPO's single global clip_range with per-task adaptive bounds
  based on each task's current entropy. Tasks with low entropy (over-confident,
  e.g., hover after prior stabilisation) get a tighter clip bound; tasks still
  exploring (slalom, circle early in training) get a relaxed bound. Uses EMA-
  tracked per-task entropy to set bounds each update.
- Why it fits our bottleneck: hover is prior-stabilised and converges fast
  (low entropy), while slalom/circle need more exploration budget. A uniform
  clip_range=0.2 over-restricts trajectory tasks that are still learning gate
  geometry. EPPO's adaptive bound lets hover settle without constraining the tasks
  that are still in the learning phase — complementary to PopArt (which fixes the
  critic) by also fixing the actor clipping.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass PPO and override
  `train()`. Before each minibatch update, compute per-task entropy from the current
  policy (already available as `log_prob` in SB3's rollout buffer). Set per-task
  `clip_range_t = base_clip * (H_t / H_max)` clipped to [0.05, 0.35]. ~45 lines.
  Flag `--eppo-clip`. Shares per-task minibatch infrastructure with PCGradPPO.
- Risk: P10 (asymmetric entropy annealing) targets the same per-task entropy lever
  via ent_coef; running both simultaneously may interact. Screen EPPO alone first
  before combining with P10. Also note clip_range=0.15 failed (exp_023/076 aborted),
  so the lower bound of 0.05 must never be reached in practice — add a hard floor at
  0.12 per task.

---

## Section 5: Post-exp_085 Bottleneck Analysis (2026-08-04)
Current confirmed baseline: 32.374. Active blockers: (a) PopArt failed in exp_085
because return normalisation corrupted GAE bootstrap values; (b) gamma can't exceed
0.99 (slalom catastrophe exp_082); (c) Tanh locked in (SiLU unbounded, exp_083);
(d) n_steps=2048 near-miss at screen (-0.261, exp_084). Root cause: hover returns
(~64) dominate shared critic; slalom/fig8 interference on shared trunk still not
resolved. Methods below address these specific failure modes.

### PopArt — Loss-Layer-Only Fix (Hessel et al., AAAI 2019, arXiv:1809.04474)
- Mechanism: Normalise each task's critic *target* at loss-computation time (not in
  rollout_buffer). Keep raw values in rollout_buffer.returns for GAE bootstrap;
  apply `y_norm = (y - mu_t) / sigma_t` only when computing the MSE loss inside
  the minibatch loop. Output-layer weights rescaled in-place when mu/sigma update
  so `V_raw(s) = sigma_t * V_norm(s) + mu_t` remains consistent across updates.
- Why it fits our bottleneck: exp_085 identified the exact failure mode — normalising
  rollout_buffer.returns corrupts the bootstrap. The fix is surgical: move
  normalisation to the loss site, leave GAE untouched. Hover (mu~64, sigma~8) and
  slalom (mu~27, sigma~6) will present equal-scale gradients to the shared critic.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass SB3 PPO and
  override `train()`. Add a `PopArtNorm` module (5 EMA trackers, one per task);
  inside the minibatch loop, after selecting task mask, normalise that slice's
  returns before computing `values_pred` loss. Rescale output-layer weights after
  each mu/sigma update. ~50 lines; no change to `compute_returns_and_advantage`.
- Risk: If the EMA window is too short, sigma can spike on sparse gate-pass events
  and de-normalise too aggressively. Use EMA decay=0.999 and a sigma floor of 1.0.
  Test with `--popart-decay 0.999 --popart-floor 1.0`.

### CAGrad — Conflict-Averse Gradient Descent (Liu et al., NeurIPS 2021, arXiv:2110.14048)
- Mechanism: At each actor update step, computes per-task gradients then finds the
  descent direction that minimises average task loss while maximally improving the
  worst task. Solves a small QP (Frank-Wolfe, 5 tasks, ~10 iterations) to find task
  weights w*, then returns gradient `g* = sum_i w*_i * g_i`. Unlike PCGrad (which
  projects pairwise), CAGrad has a convergence guarantee on average loss and
  explicitly protects the worst-performing task — directly relevant given fig8 is
  the consistent casualty in our gradient interference experiments (exp_072, 074).
- Why it fits our bottleneck: Conflict fraction 0.44-0.50 was measured at 200k
  (exp_P5). CAGrad's c hyperparameter (c=0.5 recommended) sets how aggressively
  it defends the worst task; with fig8 as the tail task this is more targeted than
  PCGrad's pairwise projection. Replaces P7 proposal (which used the same logic but
  is now validated by exp_085 failure narrowing the search space to gradient methods).
- Implementation sketch: In `37_train_residual_mtrl.py` or a subclass of PCGradPPO
  from P5, replace `_pcgrad_project` with `_cagrad_project`. The QP is 5-variable;
  use `scipy.optimize.minimize` with SLSQP or implement the 3-line Frank-Wolfe from
  the official repo (github.com/Cranial-XIX/CAGrad). ~60 lines. Flag `--cagrad-c`.
- Risk: QP adds ~5ms per minibatch on CPU (5 tasks, 10 FW iterations). With
  n_minibatches=32 and n_epochs=10, that's ~1.6s extra per rollout — acceptable.
  If FW fails to converge (rare with 5 tasks), fall back to equal weighting.

### Per-Task Advantage Whitening (TOPPO, Li et al., arXiv:2605.11473, 2026)
- Mechanism: Normalise advantages independently within each task's slice of the
  minibatch (zero mean, unit variance per task) before the actor loss. This is
  distinct from the global advantage normalisation SB3 applies by default.
  TOPPO shows this removes most actor-side reward-scale spread and is a prerequisite
  for their FairGrad critic balancing to work cleanly — but the whitening step alone
  is already worth testing as a zero-overhead 5-line change.
- Why it fits our bottleneck: Hover advantages (anchored by prior, tightly
  distributed) and slalom advantages (sparse gate-pass spikes, high variance)
  currently compete in the same normalisation pool. Per-task whitening ensures
  slalom's rare large-advantage events are not washed out by hover's bulk.
  Directly addresses the exp_073 finding that vf_coef cannot be safely reduced
  because trajectory tasks need their full advantage signal.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass SB3 PPO. In the
  minibatch loop, before computing the policy loss, split `advantages` by task_id
  mask and apply `(adv - adv.mean()) / (adv.std() + 1e-8)` per slice. Re-concatenate
  in original order. ~15 lines. Zero new hyperparameters.
- Risk: If a task has very few samples in a minibatch (e.g., hover draws only 8
  episodes in a short rollout), per-task std is unreliable. Add a minimum-sample
  guard: only whiten task slices with n >= 32 samples; fall back to global
  normalisation otherwise.

---

## Section 6: Post-exp_087 Literature Proposals (2026-08-05)

**Context.** exp_087 confirmed that per-task advantage whitening (TOPPO) is a false
positive at 20M that degrades to |res|→0.999 at 100M: rescaling advantages to N(0,1)
each rollout removes the signal about absolute reward magnitude, so the policy learns
to saturate Tanh instead of improving returns. PopArt (exp_085) failed earlier by
corrupting the GAE bootstrap. Both failures share a root cause: any normalisation
that changes the *scale* of the advantage signal relative to what the residual policy
sees at the Tanh output will eventually cause the network to compensate via saturation.
The methods below either (a) operate on task loss weights, not advantage values
(FAMO, CAGrad — already in Section 5 but strengthened here), or (b) normalise rewards
at collection time before GAE so bootstrap is untouched (GDPO-style per-reward std),
or (c) normalise only the critic target without touching actor advantages (PopArt
loss-layer fix, already in Section 5). New entries focus on (b) and the O(1) loss
reweighting alternative.

### GDPO Per-Reward Std-Only Normalisation (Liu et al., arXiv:2601.05242, NVIDIA 2026)
- Mechanism: Normalises each reward component independently at rollout-collection
  time by dividing by a running per-task std (no mean subtraction). Returns are then
  summed before GAE. This is structurally different from per-task advantage whitening:
  it scales the reward stream before discount accumulation, so GAE bootstrap values
  remain internally consistent. The key is std-only (no mean removal), preserving
  the absolute performance level signal that exp_087 destroyed.
- Why it fits our bottleneck: exp_087's saturation came from rescaling advantages to
  unit variance *after* GAE, which erased the magnitude difference between hover (~64
  return) and slalom (~27 return). GDPO-style normalisation at reward time (before
  GAE) keeps the relative ordering intact and does not inject a rollout-varying
  rescaling factor into the Tanh gradient. The per-task std is a slow EMA (decay
  0.999), so it cannot produce the per-rollout whitening pattern that drove saturation.
- Implementation sketch: In `37_train_residual_mtrl.py`, add a `PerTaskRewardNorm`
  wrapper around the env step. Maintain 5 EMA std trackers (one per task, decay
  0.999, floor=1.0). On each `step()`, divide reward by `sigma_t` before storing.
  No change to `compute_returns_and_advantage`, no change to actor loss. ~30 lines in
  `torch_drone_gate_env.py` or a VecEnv wrapper. Flag `--per-task-reward-std`.
- Risk: The EMA std is computed over the full training run; early in training when
  hover returns are noisy, sigma_hover may be overestimated, depressing hover reward
  signal. Use a warmup period (first 2M steps: no normalisation) before activating.
  Do NOT apply mean subtraction — that would shift the reward baseline and corrupt
  the prior's calibration for hover.

### FAMO Loss Reweighting (Liu et al., NeurIPS 2023, arXiv:2306.03792)
- Mechanism: Maintains per-task loss weights that are updated online to balance the
  rate of loss decrease across tasks. Unlike CAGrad/PCGrad, FAMO reweights task
  losses before the backward pass (O(1) space/time vs O(K) for gradient methods) by
  solving a closed-form weight update using only the scalar task losses from the
  previous and current step. No per-task gradient computation or QP solver needed.
- Why it fits our bottleneck: CAGrad (P7/Section 5) requires per-task gradient
  decomposition at the actor loss level, which is expensive with 5 tasks × 90 morphs.
  FAMO achieves similar worst-task protection via loss reweighting that runs in O(1).
  With hover loss consistently lower than slalom/fig8 loss, FAMO will up-weight
  slalom/fig8 automatically each rollout — the same interference fix as CAGrad but
  with ~5ms per rollout instead of ~1.6s (no QP). Directly compatible with SB3's
  existing multi-task actor loss structure.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass SB3 PPO and
  override `train()`. Track per-task actor losses from the previous update step.
  Compute FAMO weights: `w_t = softmax(-gamma * (L_t - L_t_prev) / L_t_prev)`.
  Multiply each task's actor loss slice by `w_t` before `.backward()`. ~40 lines.
  Flag `--famo-gamma` (recommended 0.025 from paper). No QP, no scipy dependency.
- Risk: FAMO's weight update assumes losses decrease monotonically; in RL where
  rewards are non-stationary, L_t_prev may be stale after a rollout that changed
  the reward distribution. Clip weights to [0.1, 3.0] per task to prevent any single
  task from dominating if loss estimates are noisy. Screen at 20M with |res| monitor:
  if |res| drifts above 0.85 before 20M, revert (same early-warning as exp_087).

---

## Section 7: Post-exp_088 Literature Proposals (2026-08-05)

### Progress Reward — Dense Gate Shaping (Song et al., IROS 2021, arXiv:2103.08624)
- Mechanism: Per-step reward `r_p = vel_world · d̂` where `d̂` is the unit vector to
  the next gate. Dense signal every timestep, not only on gate crossing. Directly
  addresses lateral drift in fig8 by rewarding approach direction not just speed.
- Why it fits our bottleneck: fig8=24.0 is the consistent floor. Velocity-toward-gate
  (0.40 coef) is saturated (exp_071). Progress reward is denser and directional —
  penalises lateral drift implicitly. Strategy memo ranks this #1 unexplored change.
  exp_021 tested distance-difference (not velocity-dot) pre-fixed-start; a clean retest
  at 20M is warranted.
- Implementation sketch: `torch_drone_gate_env.py`, add `vel_world · d̂ * progress_coef`
  to step reward for figure8 task only. ~15 lines. Flag `--progress-coef 0.10`.
- Risk: exp_021 failed at 250k (different formulation, random-init era). If
  `progress_coef > 0.15` raises |res| above 0.85 at screen, back off immediately.

### GradNorm Adaptive Loss Balancing (Chen et al., ICML 2018, arXiv:1711.02257)
- Mechanism: Learns per-task actor-loss weights by minimising a gradient-norm loss:
  tasks whose gradient norm is below average get up-weighted; over-represented tasks
  get down-weighted. Single asymmetry hyperparameter α. ~5% wall-clock overhead.
- Why it fits our bottleneck: hover's analytical prior generates large low-variance
  gradients that dominate the shared trunk. GradNorm directly equalises per-task
  gradient magnitudes at the final shared layer — same goal as CAGrad but via a
  differentiable weight update (no QP, no scipy). Lighter than CAGrad for 5 tasks.
- Implementation sketch: In `37_train_residual_mtrl.py`, after computing per-task actor
  losses, hook `.weight.grad` at trunk's final layer per task, compute GradNorm loss
  `Σ||G_t/Ḡ − r_t^α||²`, backprop through `{w_t}` only. ~50 lines.
  Flag `--gradnorm-alpha` (try 0.5 and 1.5). Do not combine with CAGrad/FAMO.
- Risk: Gradient hooks may interfere with SB3's `max_grad_norm` clipping — apply
  GradNorm weight update before the global norm clip, not after.

### Asymmetric Per-Task Discount Factor (motivated by AMAGO multi-gamma, arXiv:2310.09971, ICLR 2024)
- Mechanism: Set γ=0.995 for hover/fig8/circle (arc tasks, effective horizon ~200
  steps) and γ=0.99 for slalom/shuttle-run (zig-zag tasks). Override SB3's
  `compute_returns_and_advantage` to select γ per episode from `task_id` in `infos`.
  AMAGO (2024) showed parallel multi-γ critic losses improve representation learning;
  this applies the insight selectively to protect slalom from variance catastrophe.
- Why it fits our bottleneck: exp_082 confirmed γ=0.999 collapses slalom (−24.8 at
  100M) but helped circle (+2.2) and hover (+2.7). Asymmetric γ captures arc-task
  benefit without the slalom failure. Strategy memo lists this as priority #2.
- Implementation sketch: Subclass SB3 PPO, store `task_id` per transition in rollout,
  pass `gamma_dict` to patched `compute_returns_and_advantage`. ~40 lines.
  Flag `--gamma-arc 0.995`. Hard screen-abort: slalom 20M delta < −3.
- Risk: SB3 rollout buffer assumes single γ; episode-boundary handling requires care.
  Higher γ for arc tasks increases their return scale, partially re-introducing
  hover-dominance. Monitor |res| at screen; reject if slalom screen-delta < −3.

---

## Section 8: Post-exp_089 Literature Proposals (2026-08-05)

**Context.** 9 consecutive screen failures; all in PPO hparam / architecture /
reward-shaping / entropy / advantage space. |res|=0.77–0.88 at 20M on recent
failures (exp_083 SiLU, exp_087 per-task whitening, exp_089 deeper trunk).
Root cause: residual Tanh saturates when the network receives signals that
increase the *scale* of what it needs to output. Methods below avoid any change
to advantage scaling and focus instead on (a) data allocation across tasks and
(b) orthogonal representation inductive bias in the trunk.

### DRATS — Distributionally Robust Adaptive Task Sampling (Corrado et al., arXiv:2605.14350, 2026)
- Mechanism: Formalises MTRL as a feasibility problem and derives a minimax
  objective for minimising the worst-case return gap. Maintains a softmax
  distribution q over tasks updated via mirror ascent on return gaps: tasks
  furthest from a target return receive proportionally more rollout episodes.
  Modifies only *which* task episodes are collected per rollout; losses and
  advantage computation are untouched. Demonstrated with PPO on MetaWorld-MT10/50.
- Why it fits our bottleneck: Current rollout allocates episodes uniformly across
  5 tasks. Hover converges fast (return~64) and wastes episode budget; slalom/fig8
  are data-starved. DRATS would automatically shift budget toward slalom/fig8 as
  hover saturates, without touching advantage values or loss scaling — safe w.r.t.
  the Tanh saturation failure mode identified in exp_083/087/089.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass SB3 PPO and
  override `collect_rollouts()`. Maintain 5 running-average return estimates and
  target returns (set from exp_080 best: hover=64.5, fig8=24.0, etc.). Compute
  gaps g_i = max(0, target_i − return_i) and set q_i = softmax(η*g_i) (η~0.5).
  Sample task proportions from q before assigning envs. ~35 lines.
  Flag `--drats-eta 0.5`. No change to train(), compute_returns_and_advantage.
- Risk: If target returns are set too high (above the policy's reachable ceiling),
  hover will permanently receive near-zero episodes and the prior loses its
  stabilising role. Set target returns at 110% of current best per task, not at
  theoretical max. Hard abort: hover screen-delta < −4 (prior destabilised).

### MOORE — Mixture of Orthogonal Experts (Hendawy et al., ICLR 2024, arXiv:2311.11385)
- Mechanism: Replaces the flat shared trunk with k expert networks whose output
  representations are orthogonalised via Gram-Schmidt at each forward pass. Each
  task interpolates its trunk representation as a weighted sum of the k orthonormal
  expert outputs. This forces task-specific features into orthogonal directions,
  eliminating the subspace collapse that causes slalom/fig8 to interfere in the
  shared trunk. Architecture-agnostic; the paper validates with both PPO (MiniGrid)
  and SAC (MetaWorld), achieving SOTA on MetaWorld.
- Why it fits our bottleneck: exp_068 showed per-task actor heads catastrophically
  fail; exp_089 showed deeper trunk also fails — the trunk is not capacity-limited
  but *interference*-limited. MOORE's orthogonal subspace gives slalom and fig8
  their own representation directions without adding depth or per-task heads.
  Does not touch advantage values, critic targets, or reward scaling.
- Implementation sketch: In `37_train_residual_mtrl.py`, subclass
  `MTRLActorCriticPolicy`. Replace `shared_trunk` with a `MOOREBlock(k=4,
  d=256)` module: k small MLPs (64→256) + Gram-Schmidt orthogonalisation +
  per-task interpolation weights (5×k learned params). ~70 lines in a new
  `p_moore.py`. Flag `--moore-k 4`. The orthogonalisation runs on the forward
  pass with no extra backward overhead. Keep Tanh activations unchanged.
- Risk: Gram-Schmidt with k=4 experts and d=256 adds O(k²d) FLOPs per forward
  pass (~0.26M FLOPs, <1% of a 256→256 linear layer). Main risk: interpolation
  weights are per-task (not per-morph), so 90 morphologies must share the same
  5×k basis. If morph diversity requires more dimensions than k=4 provides,
  increase to k=8 — but screen k=4 first. Do not combine with FiLM (P8) in
  the same experiment.
