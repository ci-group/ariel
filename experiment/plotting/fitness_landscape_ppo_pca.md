# PPO Fitness Landscape — Mathematical Reference

Companion document to `experiment/plotting/fitness_landscape_ppo_pca.py`.
Explains what the landscape plots actually compute, why the construction is
useful, and where its interpretation breaks down.

Notation uses Unicode symbols so the formulas render in any markdown viewer
(no LaTeX engine required). The pipeline follows Li, Xu, Taylor, Studer &
Goldstein (2018), *Visualizing the Loss Landscape of Neural Nets*, adapted
from supervised CNN loss to PPO reward / displacement over MJX rollouts.

---

## 1. Weight space, formally

Let the policy network have parameters θ ∈ Θ (biases + weights of all
layers). The flattening operator

```
φ : Θ → ℝᵈ,     d = Σₗ (nₗⁱⁿ · nₗᵒᵘᵗ + nₗᵒᵘᵗ)
```

is a bijection; we identify Θ with ℝᵈ. For the (32, 32) policy on the
insect observation / action dims, d ≈ 1400.

Training produces a sequence

```
{θ₁, θ₂, …, θ_K} ⊂ ℝᵈ,     K = 10
```

where θₖ is the policy state at eval boundary k. Stack them into the data
matrix

```
W ∈ ℝ^(K×d),     W[k, :] = θₖᵀ
```

The normalizer (μ_obs, σ_obs) and the value head ψ are also in the saved
tuple, but the landscape fixes them at (μ_obs⁽ᴷ⁾, σ_obs⁽ᴷ⁾, ψ⁽ᴷ⁾) — only θ
moves. The slice center θ_c is either θ_K (``--center final``, default) or
θ_b where b = argmaxₖ R̂(θₖ) (``--center best``).

### Why useful

This framing lets us ask geometric questions ("how curved is the loss near
the solution?", "is PPO's path straight?") using the full apparatus of
linear algebra. We treat training as a curve in ℝᵈ and inspect its shape.

### Limitations

- Fixing the normalizer biases evaluations far from θ_K: observations under
  a very different policy fall outside the distribution the normalizer was
  fit to, so the policy effectively receives mis-whitened inputs. The
  farther (α, β) is from (0, 0), the less the plotted value reflects "what
  this policy would do if trained from scratch."
- d is tiny for a neural net (~1400) but still vastly larger than K, so the
  trajectory samples a mote of the space.

---

## 2. PCA on the snapshot trajectory

Center the snapshots:

```
μ = (1/K) · Σ_{k=1..K} θₖ   ∈ ℝᵈ
W̃ = W − 𝟙_K · μᵀ
```

Take the thin SVD:

```
W̃ = U Σ Vᵀ

U ∈ ℝ^(K×r)
Σ = diag(σ₁ ≥ … ≥ σᵣ) ∈ ℝ^(r×r)
V ∈ ℝ^(d×r)
```

The rank is bounded by

```
r ≤ min(K − 1, d) = min(9, 1400) = 9
```

The −1 is because centering removes one direction (the mean). All K
snapshots lie in an affine subspace of dimension ≤ K − 1 passing through μ;
PCA finds the orthonormal basis of its linear part.

- **Principal directions** are the columns of V: {v₁, …, v₉} ⊂ ℝᵈ,
  orthonormal unit vectors.
- **Explained variance** of PC i: λᵢ = σᵢ² / (K − 1); the ratio is
  λᵢ / Σⱼ λⱼ = σᵢ² / Σⱼ σⱼ².

Interpreting PC1 ≈ 65%:

```
σ₁² / Σ_{j=1..9} σⱼ²  ≈  0.65
```

means 65% of the squared Frobenius distance ‖W̃‖_F² = Σⱼ σⱼ² is captured by
projecting onto a single line. This happens when gradient steps are
strongly auto-correlated — the training path is nearly straight. In PPO
the clipped surrogate loss changes slowly between updates, so this is
expected.

### Why useful

- **A single number quantifies "how 1-dimensional is training?"** The
  entropy of {σᵢ² / Σⱼ σⱼ²} or just λ₁ / Σᵢ λᵢ summarizes how straight the
  trajectory is. Straight trajectories (high PC1 ratio) suggest PPO is
  following a well-defined descent direction; curvy ones suggest it's
  exploring / correcting.
- **Reduces 1400 dimensions to a handful you can plot.** No model-free RL
  analysis technique scales to ℝ¹⁴⁰⁰ visually; PCA is the standard
  dimensionality-reducing lens.
- **Comparable across runs.** Running the same procedure for n1–n4 lets you
  compare *shapes* of landscapes without worrying about absolute weight
  coordinates — provided the perturbation units are also comparable (§3.2).

### Limitations

- **K = 10 samples is a very small statistical base.** Sample covariance
  Σ̂ = W̃ᵀ W̃ / (K − 1) is rank-deficient and its eigenvalues are badly
  biased: the top ones are overestimated and the small ones are zero. By
  the Marchenko–Pastur law for d / K → ∞, almost all signal is pushed into
  the leading components regardless of the underlying geometry. In short,
  the high "PC1 variance ratio" is partly a sampling artifact.
- **PCA is linear.** The trajectory could curve smoothly through weight
  space; PCA would still fit a line through it and report high PC1 ratio.
- **No reward information.** PCA sees only {θₖ}, not {R(θₖ)}. The chosen
  directions are "where PPO walked," not "where the reward surface
  varies." These are correlated but not identical — PPO walks downhill in
  gradient, which depends on local curvature, but need not align with the
  globally-steepest directions of the test-time reward function.

---

## 3. The direction basis

The landscape slice is spanned by two direction vectors d₁, d₂ ∈ ℝᵈ. The
script supports three choices (``--basis {pca, random, both}``) with an
optional per-filter rescaling (``--filter-normalize``, on by default).

### 3.1 Raw PCA basis (``--basis pca --no-filter-normalize``)

Take d₁ = v₁, d₂ = v₂. The grid axes point along the top two singular
directions of W̃; one unit of α is one unit of weight-vector displacement
in weight-space ℓ² norm. This is the pre-paper configuration — simplest
to interpret, but has three failure modes:

1. **Weight-magnitude confounds.** Different runs end at θ_c of
   different norms. For an MLP policy with ReLU/tanh activations and a
   Gaussian action head with fixed σ, scaling a layer's output weights up
   by k and the next layer's input weights down by k produces an
   identical deterministic policy with wildly different ‖θ_c‖, hence
   incomparable sharpness scales.
2. **Per-layer imbalance.** PC1 may put most of its mass on the output
   layer because that layer varies most in weight-ℓ² terms; the
   landscape then shows "output-layer sweep" with early layers barely
   moving.
3. **Scale drifts across runs.** σ₁ depends on how far PPO walked, which
   depends on learning rate, num_timesteps, and morphology complexity.

### 3.2 Filter-wise normalization (``--filter-normalize``, default)

Following Li et al. 2018 §4, for each weight matrix W_l ∈ ℝ^(nₗⁱⁿ × nₗᵒᵘᵗ)
in the policy MLP, column j is the fan-in vector feeding output neuron j
— the direct analogue of a conv filter. Rescale each filter of the raw
direction d_l to match the corresponding filter norm in the center
weights W_{c,l}:

```
d_l[:, j]  ←  (d_l[:, j] / ‖d_l[:, j]‖₂)  ·  ‖W_{c,l}[:, j]‖₂
```

Biases are zeroed (they carry no scale invariance to normalize against).
Denote the resulting direction d̂.

### Why useful

- **Removes the weight-magnitude confound.** One unit of α along d̂ perturbs
  each layer by a consistent *fraction of that layer's own weight norm*,
  so sharpness comparisons across checkpoints become meaningful. The
  paper demonstrates (Fig. 1–4) that this reverses the apparent
  flat/sharp verdict of several standard benchmarks.
- **Per-layer balance.** Early-layer columns get a non-trivial share of
  the perturbation even when their raw norms are small, so the plotted
  slice reflects the whole network's curvature, not just the biggest
  layer.

### Limitations / caveats

- **Oblique basis.** Normalizing each PC independently generally breaks
  orthogonality: cos(d̂₁, d̂₂) ≠ 0. The script logs this cosine so you
  can tell when the slice is near-orthogonal (OK) vs. severely oblique.
  The grid is still a well-defined 2-D affine subspace, just not an
  orthonormal frame; and (α, β) ↦ θ_c + α d̂₁ + β d̂₂ is still injective
  as long as d̂₁ and d̂₂ remain linearly independent.
- **MLP policy only.** The implementation decides by shape: 2-D array ⇒
  kernel ⇒ column-normalize; 1-D / 0-D ⇒ bias ⇒ zero. A policy whose
  params include other 2-D leaves (e.g. a learned log-σ matrix) would be
  silently mishandled. This script targets brax PPO MLPs with
  kernel / bias leaves only.
- **Biases ignored.** Zeroing the bias direction means the slice is
  parallel to the "bias = 0" affine subspace through θ_c. The paper
  argues this is the right choice; it also means bias-driven asymmetries
  in the loss are invisible to the plot.

### 3.3 Random Gaussian basis (``--basis random``)

Sample two directions d₁, d₂ with the same pytree shape as θ_c, each
leaf ~ N(0, 1). Filter-normalize (as above). This is the paper's
default basis for 2-D visualization — direction-independent of training
dynamics.

### Why useful (relative to PCA)

- **Unbiased by the training trajectory.** PCA directions correlate with
  where PPO walked, so the surface there reflects "landscape along the
  descent path" — which is biased toward easier terrain (gradient
  descent avoids cliffs). Random directions sample the local
  neighborhood independently of training dynamics; a sharp drop at
  random-α ≈ 0.3 but a smooth descent along PC1-α ≈ 0.3 reveals that
  PCA found a lucky-easy path through a locally rougher region.
- **Direct cross-run comparability.** No dependence on trajectory
  length; ``--random-scale`` controls the half-width in units of
  ‖θ_c‖, so n1–n4 landscapes land on the same visual axis.

### Limitations

- **α has different semantics.** For PCA, α = s₁ means "the policy
  drifted s₁ std-devs along the actual PC1 direction." For random,
  α = s means "the policy moved ≈ s · ‖θ_c‖ in an arbitrary direction."
  Both are legitimate but not interchangeable.
- **High-dim random directions nearly orthogonal to the trajectory.**
  E[cos(d, θ_K − θ_1)] ≈ √(2/(πd)) ≈ 0.02 for d ≈ 1400, so the random
  slice never contains the training path; the migration animation is
  disabled in this mode.

### 3.4 The 2D grid

Let θ_c be the center and (d₁, d₂) be the chosen (and possibly
filter-normalized) basis. The script **standardizes both directions to
unit ℓ² length** after any filter normalization: filter-norm's job is to
set the per-layer *ratio* of the perturbation vector (each layer's share
proportional to that layer's own weight norm); its *overall* magnitude
is then normalized away so that one unit of α equals one unit of
weight-space ℓ² move. Without this standardization, filter-normalized
directions have ‖d‖ ≈ ‖θ_c‖, which combined with the trajectory-std
scaling below inflates the sweep by ‖θ_c‖² and collapses the whole grid
onto an isolated spike at the center.

With ‖d₁‖ = ‖d₂‖ = 1, the half-widths are:

```
PCA basis:    sᵢ = pc_range · std({ ⟨dᵢ, θₖ − θ_c⟩ : k = 1..K })
random basis: sᵢ = random_scale · ‖θ_c‖
```

Build a 25 × 25 lattice:

```
G = { (αᵢ, βⱼ) :
        αᵢ ∈ linspace(−s₁, +s₁, 25),
        βⱼ ∈ linspace(−s₂, +s₂, 25) }
```

Each grid point corresponds to a perturbed policy

```
θ(α, β) = θ_c + α · d₁ + β · d₂
```

defining a 2D affine slice

```
S_{θ_c} = θ_c + span(d₁, d₂)
```

### Why useful

- Defines a **meaningful "unit" for weight perturbations** per basis.
- **Reproducible across runs.** Any snapshot trajectory (PCA) or any
  rng seed (random) yields a canonical slice.

### Limitations

- The slice is ℝ² inside ℝᵈ ≈ ℝ¹⁴⁰⁰; the orthogonal complement has
  dimension 1398 and any curvature there is invisible. A policy that
  looks fine on the slice could be surrounded by cliffs orthogonal to
  (d₁, d₂).
- The slice is linear, not geodesic. Permutation / rescaling
  symmetries of ReLU/tanh networks create flat directions and non-
  linear structure in weight space; a straight line between two
  "equivalent" networks can pass through non-equivalent, worse policies.
  Filter normalization mitigates the sharpness-comparison artefact but
  does not change the slice from Euclidean to geodesic.

---

## 4. The Z value — scalar fields over the slice

The script supports two scalar fields — *displacement* (behavioural) and
*reward* (optimisation-objective) — optionally averaged over multiple
reset seeds for an uncertainty band. Each grid cell at (α, β) is
evaluated by one or more deterministic rollouts of the policy θ(α, β).

### 4.1 Displacement

```
D : ℝᵈ → ℝ≥₀,     D(θ) = ‖ x_T(θ)[0:2] ‖₂
```

where x_T(θ) ∈ ℝ^(nq) is the generalized position after a T-step
deterministic rollout starting from fixed initial state x₀ and fixed
scan RNG. The rollout composes

```
x_{t+1} = f_env( x_t , argmax_a π_θ(a ∣ o_t) ),     t = 0, …, T−1
```

where π_θ uses the frozen normalizer in computing o_t. D is deterministic
in θ but **piecewise-smooth with discontinuities** from:

- action clipping   a ← clip(a, −π/2, +π/2)   at the actuator;
- episode termination   𝟙[ x_t^up-proj < −0.3 ]   (abrupt freeze of the
  trajectory);
- contact events (impact discontinuities in the physics).

The landscape restricts D to the slice:

```
f_c^D : ℝ² → ℝ,     f_c^D(α, β) = D( θ_c + α · d₁ + β · d₂ )
```

### 4.2 Mean per-step reward (``--reward``)

```
R : ℝᵈ → ℝ,     R(θ) = (1/T) · Σ_{t=1..T} r_t
```

where r_t is the env's per-step scalar reward (the quantity PPO's
advantage targets, after sum-and-scale but before clipping). Averaging
rather than summing keeps R interpretable regardless of T — summing
would mix episode length into the comparison for policies that
terminate early via the upright-termination guard.

```
f_c^R : ℝ² → ℝ,     f_c^R(α, β) = R( θ_c + α · d₁ + β · d₂ )
```

### Why both

- **D is behavioural.** Locomotion research cares that the robot
  travels; D plots directly in meters, no reward-engineering layer in
  the way.
- **R is what PPO sees.** The gradient at θ_c points along ∇R(θ_c), so
  curvature of R near θ_c is what actually governed the final updates.
  Cells where f_c^D and f_c^R disagree (high R, zero D or vice-versa)
  expose reward-hacking modes directly as mismatched contours.

### Limitations

- Both are piecewise-smooth. The finite-difference Hessian below (§5.1)
  is still well-defined, but an autograd-through-rollout Hessian is
  not.
- R inherits the frozen-normalizer caveat from §1; edge-of-slice r_t
  values are untrustworthy.

### 4.3 Multi-seed averaging (``--seeds-per-cell N``)

For N ≥ 2 the inner rollout is vmapped over N reset keys:

```
f̄_c(α, β) = (1/N) · Σ_{s=1..N} f_c(α, β; seed = s)
σ_c(α, β) = sample std of { f_c(α, β; seed = s) }_{s=1..N}
```

The main landscape plot shows f̄_c; a companion heatmap shows σ_c; the
1-D cross-sections (§5.2) draw ±1σ bands.

### Why useful

- **Error bars where there were none.** Single-seed Z is a point
  estimate with no uncertainty; the shape of the landscape could be
  dominated by one unlucky initial condition. Multi-seed quantifies
  this.
- **Separates landscape structure from eval noise.** A cell with large
  σ_c is one where the policy is on a knife's edge — small env
  perturbations flip the behaviour. Large σ_c is itself informative.

### Limitations

- **Linear cost in N.** 25² grid × 5 seeds × 500 ctrl-steps = 3.125 M
  env steps, vs. 625 K for N = 1.
- **Correlated seeds.** Reset randomness is only joint-angle noise
  ±0.05 rad; physics / policy are otherwise identical. σ_c
  underestimates the variance under the broader initial-state
  distribution a redeployed policy would see.

---

## 5. Diagnostics

### 5.1 Hessian at center (sharpness)

Second-order central differences of f_c at (α, β) = (0, 0):

```
H = [ f_αα   f_αβ ]
    [ f_αβ   f_ββ ]

f_αα = ( f(+Δα, 0) − 2 f(0, 0) + f(−Δα, 0) ) / Δα²
f_ββ = ( f(0, +Δβ) − 2 f(0, 0) + f(0, −Δβ) ) / Δβ²
f_αβ = ( f(+Δα, +Δβ) − f(+Δα, −Δβ)
      − f(−Δα, +Δβ) + f(−Δα, −Δβ) ) / (4 Δα Δβ)
```

with Δα = 2 s₁ / (grid_n − 1), Δβ = 2 s₂ / (grid_n − 1) — i.e. the actual
grid spacing. Eigendecompose: H = Q diag(λ_max, λ_min) Qᵀ. The script
reports

- **λ_max, λ_min** — slice-curvature magnitudes along H's principal axes.
- **Shape classification** of H's eigenvalue signs:
  - ``basin``  both λ > 0 (local min of Z, convex)
  - ``peak``   both λ < 0 (local max of Z, concave)
  - ``saddle`` mixed signs
  - ``flat``   both ≈ 0
- **Convexity index** c = −|λ_min_neg| / |λ_max_pos| ∈ [−1, 0], defined
  only for saddles (where both signs are present) following Li et al.
  2018 §6:
  ```
  c ≈  0      cylindrical valley (one direction nearly flat)
  c ≈ −1     perfectly symmetric saddle
  ```
  For pure basins or pure peaks the index is undefined and reported as
  ``n/a`` — a positive-only or negative-only Hessian has no "convex vs
  non-convex" tension to measure. (A naive algebraic ratio λ_min/λ_max
  would print `+3.97` for an anisotropic peak, which looks like a
  "very convex basin" number but means the opposite — hence the honest
  n/a.)

Printed on each basis's log line and overlaid on the landscape title.

### Why useful

- Reduces "the surface looks sharp" to a number.
- Compares across runs/morphologies on a bounded ratio.
- Signals saddle-point minima vs. true basins without staring at a 3-D
  render.
- Combined with filter normalization (§3.2), c becomes a scale-
  invariant sharpness statistic.

### Limitations

- The 2 × 2 Hessian lives in the slice; the full ambient Hessian in ℝᵈ
  has 1400 eigenvalues, mostly tiny or negative in overparameterised
  nets. Slice-Hessian c ≈ +1 does not imply ambient convexity.
- Finite-difference accuracy depends on Δα, Δβ. With grid=25 and
  pc_range=2 (PCA) the step is ≈ 0.17 × std along the PC — fine for
  smooth f, but a lot of smoothing if the surface oscillates on the
  scale of a few cells. Doubling grid_n halves Δα.
- D and R are piecewise-smooth; H can be dominated by a nearby
  termination event and report a huge spurious λ_min.

### 5.2 1D cross-sections

Two 1-D plots on a shared y-axis:

```
f_c(α, 0)   for α ∈ [−s₁, +s₁]
f_c(0, β)   for β ∈ [−s₂, +s₂]
```

If N ≥ 2 the sample std σ_c is drawn as a ±1σ shaded band around the
mean curve.

### Why useful

- Flat-vs-sharp is **directly readable** on a 1-D plot; 3-D surfaces
  conceal it under perspective.
- Overlaying 1-D slices from n1–n4 on a single figure is the cleanest
  cross-morphology sharpness comparison.
- The ±1σ band shows whether a shape is a landscape feature or an
  eval-noise feature.

### 5.3 Trajectory diagnostics

Two numbers logged alongside the landscape and overlaid on titles:

- **Cumulative explained variance** at 2 PCs,
  cumvar(2) = (σ₁² + σ₂²) / Σⱼ σⱼ². The paper claims (§7) that
  "optimization trajectories lie in extremely low dimensional space,
  40–90% in 2D." The landscape shows the top-2-PC slice; cumvar(2) is
  the fraction of trajectory variance the plot can see.

- **Final-step alignment**,

  ```
  cos(Δθ_K, v₁) = ⟨θ_K − θ_{K-1}, v₁⟩ / ‖θ_K − θ_{K-1}‖₂
  cos(Δθ_K, v₂) = ⟨θ_K − θ_{K-1}, v₂⟩ / ‖θ_K − θ_{K-1}‖₂
  span alignment² = cos²(Δθ_K, v₁) + cos²(Δθ_K, v₂)
  ```

  High span² (close to 1) means the plane the plot shows is where PPO
  was pointing on the last update, so curvature at the center reflects
  the final gradient direction. Low span² means the last step mostly
  lives *outside* the plotted plane and the slice near the center is a
  side-view of training's final push.

### Why useful

- cumvar(2) tells you *how much* of training is in the picture.
- Alignment tells you *whether the picture includes the final descent*.
  Together they bound the interpretability of the slice.

### Limitations

- Both numbers inherit K = 10's sampling bias (see §2 Limitations).
- "Final step" is one eval-boundary gap, not one gradient step — noisy
  and over-smoothed. Averaging over the last few gaps would be more
  stable, but K = 10 is small.

### 5.4 Migration animation (PCA basis only)

Snapshots θₖ projected onto the (possibly oblique, possibly filter-
normalized) basis:

```
xₖ = ⟨d₁, θₖ − θ_c⟩ / ‖d₁‖₂²
yₖ = ⟨d₂, θₖ − θ_c⟩ / ‖d₂‖₂²
```

With an orthonormal basis these are the standard orthogonal projections;
with a filter-normalized (hence oblique) basis they are the oblique-
projection coefficients that satisfy α·d₁ + β·d₂ ≈ projection into
span(d₁, d₂). Random basis is disabled for migration (see §3.3) because
its slice never contains the trajectory.

---

## 6. What the whole construction gives you, in one sentence

For a basis B = (d₁, d₂) and functional F ∈ {D, R}, the plot is the
restriction of F to a 2-D affine slice of weight space through θ_c:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   f_c(α, β) = 𝔼_seed [ F( θ_c + α · d₁ + β · d₂ ; seed ) ]            │
│                                                                      │
│   B = PCA-plane of W̃         (--basis pca)                           │
│       random Gaussian plane  (--basis random)                        │
│   d₁, d₂ filter-normalized   (--filter-normalize, default)           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Together with the Hessian at the center, the 1-D cross-sections, the
scree curve, the migration animation (PCA basis), and the multi-seed
std heatmap, this is a visualisation battery for the local geometry of
F near a trained policy.

---

## 7. Why this method, globally

- **Diagnoses convergence.** Center at a peak → converged; on a slope →
  under-trained; on a saddle → ambiguous. The convexity index (§5.1)
  turns that visual read into a scalar.
- **Distinguishes "stuck" from "untrained."** A sharp isolated bump =
  local-optimum trap; a broad featureless slope = optimiser still has
  signal.
- **Comparative.** Running it per-morphology (n1–n4) exposes qualitative
  differences in training dynamics — mesa (easy, forgiving landscape)
  vs. isolated bump (hard, narrow basin). With filter normalisation
  those comparisons are no longer confounded by weight magnitude.
- **Separates behaviour from objective.** Comparing f_c^D vs. f_c^R
  reveals reward-hacking modes as mismatched contours.
- **Quantifies eval noise.** Multi-seed σ_c tells you whether a feature
  is landscape structure or physics variance.
- **Cheap.** ~1 min of GPU time per run per basis, plus linearly more
  for N seeds.

## 8. Why this method is *not* enough

- Only probes **two directions out of ~1400**. The paper that introduced
  this technique (Li et al. 2018) emphasises that these slices are
  illustrative, not proof of anything about the actual loss geometry.
- For PCA basis: the 2 directions are **chosen by the optimiser**, not
  by the landscape. The reward surface's intrinsic directions (Hessian
  eigenvectors, Fisher eigenvectors) are what actually matter; PCA
  eigenvectors of the trajectory are a proxy that correlates with them
  when gradient descent is well-behaved.
- For random basis: the 2 directions are **arbitrary** and have the
  opposite bias — they usually miss the direction PPO cares about.
- Multi-seed averaging adds behavioural uncertainty but **not** policy
  uncertainty. Repeating from scratch with different *training* seeds
  would do that, at vastly higher cost.
- The slice Hessian is in ℝ², not the ambient Hessian in ℝᵈ. Ambient
  convexity, Fisher information, and policy-gradient variance are
  separate measurements the slice cannot substitute for.
- **No causal story.** A mesa plot does not prove PPO *will* generalise
  to new seeds, or that the policy is at a reward maximum — it only
  shows that the plane you sliced happens to be flat and high.

In short: this is a **visualisation tool for generating hypotheses about
training dynamics**, not a measurement tool for proving properties of
the loss surface. For the latter you want Hessian or Fisher spectra on
the full ambient space, which measure intrinsic (not trajectory-
dependent) local geometry.

---

## Symbol glossary

| Symbol         | Meaning                                                  |
|----------------|----------------------------------------------------------|
| θ              | flat policy-parameter vector in ℝᵈ                       |
| d              | number of policy parameters (~1400 for insect)           |
| K              | number of training snapshots (10)                        |
| θₖ             | k-th snapshot                                            |
| W              | K × d data matrix of snapshots                           |
| μ              | mean snapshot                                            |
| W̃              | centered data matrix, W − 𝟙_K · μᵀ                       |
| U, Σ, V        | thin-SVD factors of W̃                                   |
| r              | SVD rank (≤ K − 1 = 9)                                   |
| σᵢ             | i-th singular value                                      |
| λᵢ (PCA)       | i-th eigenvalue of sample covariance, σᵢ² / (K − 1)      |
| v₁, v₂         | PC1 and PC2 unit vectors in ℝᵈ (raw PCA components)      |
| d₁, d₂         | basis directions actually used in the plot               |
| d̂ᵢ            | filter-normalized direction (equals dᵢ when --filter-normalize)|
| W_{c,l}        | center weights of layer l                                |
| s₁, s₂         | half-width of the α / β sweep (meters along dᵢ)          |
| θ_c            | center of the slice (final or best-reward snapshot)      |
| α, β           | offset coordinates along d₁, d₂                          |
| G              | 25 × 25 grid of (α, β) pairs                             |
| S_{θ_c}        | 2D affine slice through θ_c                              |
| D              | displacement functional ℝᵈ → ℝ≥₀                         |
| R              | mean-per-step reward functional ℝᵈ → ℝ                   |
| f_c^D, f_c^R   | D or R restricted to the slice                           |
| f̄_c, σ_c       | multi-seed mean and std of f_c                           |
| H              | 2 × 2 slice Hessian of f_c at (0, 0)                     |
| λ_max, λ_min (H)| eigenvalues of the slice Hessian                        |
| c              | convexity index λ_min / λ_max                            |
| cumvar(k)      | cumulative explained variance ratio at k PCs             |
| T              | episode length in control steps (500)                    |
| N              | reset seeds per cell (--seeds-per-cell)                  |
| r_t            | env per-step scalar reward                               |
| x_t            | MuJoCo state at step t                                   |
| f_env          | one-step physics transition                              |
| π_θ            | policy with parameters θ                                 |
| ‖·‖_F          | Frobenius norm                                           |
| ‖·‖₂           | Euclidean norm                                           |
| ⟨·, ·⟩         | Euclidean inner product                                  |
| St(d, 2)       | Stiefel manifold of orthonormal 2-frames in ℝᵈ           |

---

## References

- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018).
  *Visualizing the Loss Landscape of Neural Nets.* NeurIPS.
- Goodfellow, I., Vinyals, O., & Saxe, A. (2015). *Qualitatively
  characterizing neural network optimization problems.* ICLR.
- Marchenko, V. A., & Pastur, L. A. (1967). *Distribution of eigenvalues
  for some sets of random matrices.* Math. USSR-Sb.
- Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio,
  Y. (2014). *Identifying and attacking the saddle point problem in
  high-dimensional non-convex optimization.* NeurIPS.
