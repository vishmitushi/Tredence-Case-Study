# DESIGN.md — Technical Design Decisions

## 1. Importance Criterion: Derivation and Justification

### Taylor-based importance: `|w · g|`

We want to estimate, for each weight w_{ij}, the change in loss if that weight were removed (set to zero). The first-order Taylor expansion of the loss around the current weights gives:

```
L(w + Δw) ≈ L(w) + ∇L^T · Δw
```

Removing weight w_{ij} means setting Δw_{ij} = -w_{ij} (all other Δw = 0):

```
ΔL = L(w + Δw) - L(w) ≈ (∂L/∂w_{ij}) · (-w_{ij}) = -g_{ij} · w_{ij}
```

Therefore:

```
|ΔL| ≈ |g_{ij} · w_{ij}|
```

This is our importance score. Weights with small `|w · g|` can be removed with minimal impact on the loss.

### Why this is better than magnitude alone

Magnitude pruning (`|w|`) ignores gradient information entirely. Consider two cases:

1. **Large weight, near-zero gradient**: This weight has converged — it contributes stably to the network's function. Magnitude says "important" (correct), and Taylor also says "important" because `|w|` dominates.

2. **Small weight, large gradient**: This weight is actively being adjusted by training. Magnitude says "unimportant" (and would prune it), but Taylor says "potentially important" because the network is trying to grow this connection. Pruning it would fight the optimizer.

3. **Small weight, near-zero gradient**: This weight has converged to a small value — the network decided it doesn't need it. Both criteria agree: safe to prune.

Taylor importance captures case 2 correctly; magnitude does not.

### Relationship to Optimal Brain Surgeon (OBS)

The full OBS approach (LeCun et al., 1989; Hassibi & Stork, 1993) uses the second-order expansion:

```
ΔL ≈ g^T Δw + (1/2) Δw^T H Δw
```

where H is the Hessian. Our first-order criterion is the diagonal approximation of OBS — it ignores the second-order term but is computationally free (we already have w and g from the training step). The second-order term matters when weights are strongly correlated, but for unstructured pruning of individual connections, the first-order term is usually sufficient.

## 2. Gradient of a Masked Weight

### What our engine computes

In our implementation, the mask is applied inside the computation graph:

```python
# In Linear.forward():
masked_weight = W * mask  # element-wise; mask is a constant binary tensor
output = x @ masked_weight + b
```

The autodiff chain rule gives:

```
∂L/∂W = ∂L/∂(W * mask) · ∂(W * mask)/∂W = ∂L/∂(W * mask) · mask
```

For a pruned weight (mask=0):

```
∂L/∂w_{ij} = ∂L/∂(w_{ij} · 0) · 0 = 0
```

**The gradient of a masked weight is exactly zero.**

### Why this is the right choice

This is mathematically correct: the pruned weight does not participate in the forward computation, so it has no causal influence on the loss, so its gradient is zero by the chain rule.

An alternative would be to compute the "counterfactual gradient" — what the gradient *would be* if the weight were active. This is useful for **regrowth decisions** (estimating whether a pruned connection should come back), and we do compute it separately in the importance criterion. But the **optimizer should never see this counterfactual gradient**, because:

1. It would cause the optimizer to update a weight that the mask will immediately zero out — a wasted computation.
2. More critically, it would cause Adam's moment estimates (`m` and `v`) to track gradients for a weight that isn't active, creating a mismatch between the moment statistics and the actual training dynamics.

### Optimizer state on revival

The most subtle issue: when a weight is pruned (mask→0), its gradient becomes zero, so Adam's moments decay:
- `m ← β₁·m + (1-β₁)·0 = β₁·m` (decays toward 0)
- `v ← β₂·v + (1-β₂)·0 = β₂·v` (decays toward 0)

After T steps of being pruned:
- `m ≈ β₁^T · m_old ≈ 0`
- `v ≈ β₂^T · v_old ≈ 0`

If the weight is revived, the bias-corrected estimates use the global step counter t:
- `m̂ = m / (1 - β₁^t)`  — m is near 0, denominator is near 1 → m̂ ≈ 0 ✓
- `v̂ = v / (1 - β₂^t)`  — v is near 0, denominator is near 1 → v̂ ≈ 0 ✓

The update becomes: `Δw = lr · m̂ / (√v̂ + ε) ≈ lr · 0 / (0 + ε) = 0`

That seems fine — but on the *first step after revival*, the new gradient g is non-zero:
- `m = β₁ · 0 + (1-β₁) · g = (1-β₁) · g`
- `v = β₂ · 0 + (1-β₂) · g² = (1-β₂) · g²`

With global step counter t (e.g. 500):
- `m̂ = (1-β₁)·g / (1 - β₁^500) ≈ (1-β₁)·g / 1 = (1-β₁)·g`
- `v̂ = (1-β₂)·g² / (1 - β₂^500) ≈ (1-β₂)·g²`

The update: `Δw ≈ lr · (1-β₁)·g / (√((1-β₂)·g²) + ε)`
         `= lr · (1-β₁)·g / ((1-β₂)^0.5 · |g| + ε)`
         `= lr · (1-β₁) / (1-β₂)^0.5 · sign(g)`

With β₁=0.9, β₂=0.999: `(1-0.9)/(1-0.999)^0.5 = 0.1/0.0316 ≈ 3.16`

So the first update is `≈ 3.16 · lr · sign(g)` — about 3× larger than intended. This is already problematic, and in edge cases with more extreme hyperparameters, it can be much worse.

**Our solution**: we track a **per-element step counter**. When a weight is revived, its step count resets to 0, so the bias correction for that element uses `t=1`:
- `m̂ = m / (1 - β₁^1) = (1-β₁)·g / (1-β₁) = g` ✓
- `v̂ = v / (1 - β₂^1) = (1-β₂)·g² / (1-β₂) = g²` ✓
- Update: `lr · g / (|g| + ε) ≈ lr · sign(g)` — exactly one learning rate step.

This is the correct warm-up behaviour that Adam was designed to have for fresh parameters.

## 3. Autodiff Engine Bottleneck

### Where it's slow

The primary bottleneck is the **Python interpreter overhead** in the backward pass. Each operation in the computation graph stores a Python closure (`_backward_fn`), and walking the topological sort invokes these closures one at a time. For a network with K operations, this is O(K) Python function calls with no vectorisation across them.

Secondary bottlenecks:
- **Memory allocation**: each forward operation creates a new `Tensor` with `np.ndarray` allocation. A fused kernel would avoid this.
- **No in-place operations**: everything is out-of-place, doubling memory usage.
- **Topological sort**: the DFS is cheap for small graphs but uses Python `set` and recursion.

### How to optimize

1. **Trace and compile**: record the computation graph once, then compile it to a single fused kernel (this is what JAX and PyTorch's `torch.compile` do). Eliminate per-op Python overhead.

2. **Operator fusion**: combine elementwise sequences (e.g. linear → ReLU → linear) into single BLAS calls or custom CUDA kernels.

3. **Memory pooling**: pre-allocate a memory arena and reuse buffers instead of allocating per operation. This also improves cache locality.

4. **Sparse BLAS**: for pruned networks, use CSR/CSC sparse matrix representations and call sparse GEMM routines (e.g. Intel MKL's `mkl_sparse_d_mm`). This turns theoretical FLOP savings into real wall-clock savings.

5. **C extension**: rewrite the hot path (backward closures, topological sort) in C/Cython/Rust with Python bindings.

## 4. Serving a Self-Pruned Model at Scale (AQUA)

### Architecture

A multi-tenant inference service at thousands of requests per second requires:

1. **Model format**: export the pruned model as sparse weight matrices in CSR format with the binary mask baked in. Store only non-zero values and their indices.

2. **Inference runtime**: use a compiled sparse inference engine (e.g. ONNX Runtime with sparse tensor support, or a custom Triton kernel). The sparse representation must yield real speedups, not just theoretical FLOP counts.

3. **Batching**: dynamic batching (e.g. NVIDIA Triton Inference Server) to amortise per-request overhead. Group requests up to a max-latency budget (e.g. 50ms).

4. **Multi-tenancy**: each tenant may have a different pruned variant of the same base model (different sparsity levels, different fine-tuning). Use:
   - A model registry mapping (tenant, model_version) → sparse weights
   - Shared base architecture code, tenant-specific weight files
   - GPU memory management: keep hot models in GPU memory, evict cold ones

5. **Scaling**: horizontal scaling with a load balancer. Each inference worker loads one or more models. Use Kubernetes with GPU node pools and autoscaling based on request queue depth.

### Sparsity-specific considerations

- **Different sparsity levels per tenant**: a tenant willing to accept 95% sparsity gets ~20× fewer FLOPs per inference → we can serve more requests on the same hardware → lower cost per request. This is the business case for self-pruning.

- **Structured vs. unstructured sparsity**: unstructured sparsity (what we implemented) is flexible but hard to accelerate on GPUs (irregular memory access). For production, consider **semi-structured sparsity** (e.g. NVIDIA's 2:4 pattern) which has hardware support on Ampere+ GPUs, or **structured pruning** (removing entire neurons/filters) which reduces dimensions directly.

- **Accuracy SLOs**: monitor per-tenant accuracy metrics. If a heavily pruned model drops below the tenant's accuracy SLO, automatically fall back to a denser variant.

- **A/B testing**: roll out pruned models gradually, comparing accuracy/latency against the dense baseline.

### Cost model

For AQUA's cost optimisation:
```
cost_per_request ∝ FLOPs × GPU_hours_per_FLOP
```

A model at 90% sparsity (with efficient sparse inference) reduces FLOPs by ~10×, meaning we can serve ~10× more requests per GPU-hour. At thousands of RPS, this translates directly to fewer GPU instances and lower cloud spend.
