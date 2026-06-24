from __future__ import annotations

import math
import os
import sys

_LOCAL_GPYTORCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GPytorch"))
if os.path.isdir(_LOCAL_GPYTORCH) and _LOCAL_GPYTORCH not in sys.path:
    sys.path.insert(0, _LOCAL_GPYTORCH)
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:  # Keep this module importable enough to raise a useful runtime error.
    import torch
except Exception as exc:  # pragma: no cover - depends on local environment
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _TORCH_IMPORT_ERROR = None

try:
    import linear_operator  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on local environment
    _LINEAR_OPERATOR_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _LINEAR_OPERATOR_IMPORT_ERROR = None

try:
    import gpytorch
    if not all(hasattr(gpytorch, name) for name in ("Module", "kernels", "means", "functions")):
        raise ImportError("imported gpytorch does not expose the expected package API")
except Exception as exc:  # pragma: no cover - depends on local environment
    gpytorch = None
    _GPYTORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _GPYTORCH_IMPORT_ERROR = None


TensorPair = Tuple[int, int]
Vector = Tuple[float, ...]


class BayesianOptimizationDependencyError(RuntimeError):
    """Raised when the GPyTorch-backed BO dependencies are not available."""


def ensure_bo_dependencies() -> None:
    """Fail with a direct setup hint when BO dependencies are missing."""
    missing = []
    if _TORCH_IMPORT_ERROR is not None:
        missing.append(f"torch ({_TORCH_IMPORT_ERROR})")
    if _LINEAR_OPERATOR_IMPORT_ERROR is not None:
        missing.append(f"linear_operator ({_LINEAR_OPERATOR_IMPORT_ERROR})")
    if _GPYTORCH_IMPORT_ERROR is not None:
        missing.append(f"gpytorch ({_GPYTORCH_IMPORT_ERROR})")
    if missing:
        local_gpytorch = _LOCAL_GPYTORCH
        raise BayesianOptimizationDependencyError(
            "Bayesian SPS needs PyTorch and the local GPyTorch submodule installed. "
            "Missing imports: "
            + "; ".join(missing)
            + ". Install torch>=2.0 and run an editable install for "
            + local_gpytorch
            + "."
        )


@dataclass(frozen=True)
class PreferenceDataset:
    """Unique normalized vectors plus pairwise preference comparisons."""

    points: Tuple[Vector, ...]
    comparisons: Tuple[TensorPair, ...]

    @classmethod
    def from_records(cls, records: Sequence[Tuple[Sequence[float], Sequence[Sequence[float]]]]) -> "PreferenceDataset":
        index: Dict[Vector, int] = {}
        points: List[Vector] = []
        comparisons: List[TensorPair] = []

        def point_index(vector: Sequence[float]) -> int:
            key = tuple(float(value) for value in vector)
            if key not in index:
                index[key] = len(points)
                points.append(key)
            return index[key]

        for chosen, rejected_vectors in records:
            chosen_idx = point_index(chosen)
            for rejected in rejected_vectors:
                rejected_idx = point_index(rejected)
                if chosen_idx != rejected_idx:
                    comparisons.append((chosen_idx, rejected_idx))

        return cls(points=tuple(points), comparisons=tuple(comparisons))


class _PreferenceKernel(gpytorch.Module if gpytorch is not None and hasattr(gpytorch, "Module") else object):
    """Small GPyTorch kernel wrapper for finite preferential GP inference."""

    def __init__(self, dimensions: int):
        ensure_bo_dependencies()
        super().__init__()
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=dimensions,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1.0e-4),
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_constraint=gpytorch.constraints.Positive(),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.5),
        )
        # Normalized design coordinates are in [0, 1], so this is a useful smooth default.
        self.covar_module.base_kernel.lengthscale = 0.28
        self.covar_module.outputscale = 1.0
        self.mean_module.constant = 0.0
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def mean(self, x):
        return self.mean_module(x)

    def covar(self, x1, x2=None):
        return self.covar_module(x1, x2).evaluate()


class PreferentialBayesianOptimizer:
    """Finite preferential GP with Laplace approximation and EI acquisition."""

    def __init__(
        self,
        dimensions: int,
        dtype=None,
        device: Optional[str] = None,
        preference_noise: float = 0.12,
        jitter: float = 1.0e-5,
        map_steps: int = 80,
    ):
        ensure_bo_dependencies()
        if dimensions < 1:
            raise ValueError("PreferentialBayesianOptimizer needs at least one dimension")
        self.dimensions = dimensions
        self.dtype = dtype or torch.double
        self.device = torch.device(device or "cpu")
        self.preference_noise = float(preference_noise)
        self.jitter = float(jitter)
        self.map_steps = int(map_steps)
        self.kernel = _PreferenceKernel(dimensions).to(device=self.device, dtype=self.dtype)
        self.dataset = PreferenceDataset(points=(), comparisons=())
        self._train_x = None
        self._k_inv = None
        self._posterior_cov = None
        self._latent_map = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit_from_records(self, records: Sequence[Tuple[Sequence[float], Sequence[Sequence[float]]]]) -> None:
        dataset = PreferenceDataset.from_records(records)
        self.fit(dataset)

    def fit(self, dataset: PreferenceDataset) -> None:
        self.dataset = dataset
        self._fitted = False
        if len(dataset.points) < 2 or not dataset.comparisons:
            self._clear_fit()
            return

        train_x = torch.tensor(dataset.points, dtype=self.dtype, device=self.device)
        comparisons = torch.tensor(dataset.comparisons, dtype=torch.long, device=self.device)
        eye = torch.eye(train_x.size(0), dtype=self.dtype, device=self.device)

        # The Laplace MAP solve optimizes only latent utilities. Treat kernel
        # values as fixed here so repeated LBFGS closure calls do not reuse a
        # freed GPyTorch autograd graph.
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(self.jitter):
            covar = self.kernel.covar(train_x) + eye * self.jitter
            k_inv = torch.linalg.solve(covar, eye).detach()

        latent = torch.zeros(train_x.size(0), dtype=self.dtype, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([latent], max_iter=self.map_steps, line_search_fn="strong_wolfe")

        def negative_log_posterior(values):
            prior = 0.5 * values @ k_inv @ values
            log_likelihood = self._preference_log_likelihood(values, comparisons)
            return prior - log_likelihood

        def closure():
            optimizer.zero_grad()
            loss = negative_log_posterior(latent)
            loss.backward()
            return loss

        optimizer.step(closure)
        latent_map = latent.detach()
        hessian = torch.autograd.functional.hessian(negative_log_posterior, latent_map)
        posterior_precision = hessian + eye * self.jitter
        posterior_cov = torch.linalg.solve(posterior_precision, eye)

        self._train_x = train_x.detach()
        self._k_inv = k_inv.detach()
        self._posterior_cov = posterior_cov.detach()
        self._latent_map = latent_map.detach()
        self._fitted = True

    def posterior(self, candidates: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        if not candidates:
            return [], []
        with torch.no_grad():
            test_x = torch.tensor(candidates, dtype=self.dtype, device=self.device)
            if not self._fitted:
                mean = self.kernel.mean(test_x)
                variance = self.kernel.covar(test_x).diagonal().clamp_min(self.jitter)
                return self._to_float_list(mean), self._to_float_list(variance)

            k_star = self.kernel.covar(test_x, self._train_x)
            k_ss_diag = self.kernel.covar(test_x).diagonal().clamp_min(self.jitter)
            centered_latent = self._latent_map - self.kernel.mean(self._train_x)
            mean = self.kernel.mean(test_x) + k_star @ self._k_inv @ centered_latent
            correction = self._k_inv - self._k_inv @ self._posterior_cov @ self._k_inv
            variance = k_ss_diag - (k_star @ correction * k_star).sum(dim=1)
            variance = variance.clamp_min(self.jitter)
            return self._to_float_list(mean), self._to_float_list(variance)

    def expected_improvement(self, candidates: Sequence[Sequence[float]], incumbent: Sequence[float]) -> List[float]:
        if not candidates:
            return []
        candidate_mean, candidate_var = self.posterior(candidates)
        incumbent_mean, _ = self.posterior([incumbent])
        best_mean = incumbent_mean[0] if incumbent_mean else 0.0
        out = []
        for mean, variance in zip(candidate_mean, candidate_var):
            sigma = math.sqrt(max(variance, self.jitter))
            improvement = mean - best_mean
            gamma = improvement / sigma if sigma > 0.0 else 0.0
            normal = torch.distributions.Normal(
                torch.tensor(0.0, dtype=self.dtype, device=self.device),
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
            )
            gamma_tensor = torch.tensor(gamma, dtype=self.dtype, device=self.device)
            cdf = float(normal.cdf(gamma_tensor).detach().cpu())
            pdf = math.exp(-0.5 * gamma * gamma) / math.sqrt(2.0 * math.pi)
            out.append(max(0.0, improvement * cdf + sigma * pdf))
        return out

    def best_expected_improvement(
        self,
        candidates: Sequence[Sequence[float]],
        incumbent: Sequence[float],
    ) -> Tuple[Vector, float]:
        if not candidates:
            return tuple(float(value) for value in incumbent), 0.0
        scores = self.expected_improvement(candidates, incumbent)
        best_idx = max(range(len(candidates)), key=lambda idx: scores[idx])
        return tuple(float(value) for value in candidates[best_idx]), float(scores[best_idx])

    def _preference_log_likelihood(self, latent, comparisons):
        if comparisons.numel() == 0:
            return latent.new_tensor(0.0)
        chosen = latent[comparisons[:, 0]]
        rejected = latent[comparisons[:, 1]]
        scale = math.sqrt(2.0) * max(self.preference_noise, 1.0e-6)
        logits = (chosen - rejected) / scale
        return gpytorch.functions.log_normal_cdf(logits).sum()

    def _clear_fit(self) -> None:
        self._train_x = None
        self._k_inv = None
        self._posterior_cov = None
        self._latent_map = None

    def _to_float_list(self, tensor) -> List[float]:
        return [float(value) for value in tensor.detach().cpu().reshape(-1)]



