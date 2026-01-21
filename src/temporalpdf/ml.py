"""
Machine Learning module for temporalpdf.

Provides DistributionalRegressor - a distribution-agnostic learner that trains
with proper scoring rules (CRPS/log score) instead of MSE on parameters.

The key insight: predicting distribution parameters with MSE doesn't optimize
for probabilistic accuracy. CRPS directly measures how well the predicted
distribution captures the true outcome.

Example:
    >>> from temporalpdf.ml import DistributionalRegressor
    >>>
    >>> model = DistributionalRegressor(
    ...     distribution="student_t",
    ...     loss="crps",
    ... )
    >>> model.fit(X_train, y_train)  # y is raw values, not params
    >>> params = model.predict(X_test)  # Returns fitted parameters

Requires PyTorch: pip install torch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

import numpy as np

# Check torch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# For type checking only (doesn't run at runtime)
if TYPE_CHECKING:
    import torch
    import torch.nn as nn


def _check_torch() -> None:
    """Raise helpful error if torch not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the ml module. "
            "Install with: pip install torch"
        )


class DistributionalRegressor:
    """
    Distribution-agnostic regressor with proper scoring rule training.

    Instead of predicting parameters with MSE (which doesn't optimize for
    probabilistic accuracy), this trains with CRPS or log score to directly
    optimize how well the predicted distribution captures outcomes.

    Key difference from standard regression:
    - Standard: Features -> Model -> Point prediction
    - This: Features -> Model -> Distribution parameters -> Full distribution

    Example:
        >>> model = DistributionalRegressor(
        ...     distribution="student_t",
        ...     loss="crps",
        ... )
        >>> model.fit(X_train, y_train)  # y is raw target, not params
        >>> params = model.predict(X_test)
        >>> # params[:, 0] = mu, params[:, 1] = sigma, params[:, 2] = nu

    Args:
        distribution: "normal", "student_t", or "nig"
        loss: "crps" (recommended) or "log_score"
        hidden_dims: MLP hidden layer sizes
        learning_rate: Adam learning rate
        n_epochs: Training epochs
        batch_size: Mini-batch size
        n_samples: Samples for CRPS estimation (more = lower variance)
        device: "cpu" or "cuda"
        verbose: Print training progress
    """

    def __init__(
        self,
        distribution: Literal["normal", "student_t", "nig"] = "student_t",
        loss: Literal["crps", "log_score"] = "crps",
        hidden_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 32,
        n_samples: int = 100,
        device: str = "cpu",
        verbose: bool = True,
    ):
        _check_torch()

        self.distribution = distribution
        self.loss_type = loss
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32]
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.device = device
        self.verbose = verbose

        self._model: Any = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DistributionalRegressor":
        """
        Train the model on features X and target values y.

        NOTE: y should be raw target values (e.g., returns), NOT pre-fitted
        distribution parameters. The model learns to predict distribution
        parameters that best explain y according to the scoring rule.

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) target values

        Returns:
            self (for chaining)
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        n_features = X_t.shape[1]
        self._model = _DistributionalMLP(
            n_features,
            self.distribution,
            self.hidden_dims,
        )
        self._model.to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                params = self._model(X_batch)

                if self.loss_type == "crps":
                    loss = _crps_loss_via_sampling(
                        params, y_batch, self.distribution, self.n_samples
                    )
                else:
                    loss = _negative_log_likelihood(params, y_batch, self.distribution)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch+1}/{self.n_epochs}, {self.loss_type.upper()}: {avg_loss:.4f}")

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict distribution parameters for new data.

        Args:
            X: (n_samples, n_features) feature matrix

        Returns:
            (n_samples, n_params) array of distribution parameters.
            For student_t: [mu, sigma, nu]
            For normal: [mu, sigma]
            For nig: [mu, delta, alpha, beta]
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            params = self._model(X_t)

        return params.cpu().numpy()

    def predict_distribution(self, X: np.ndarray) -> list:
        """
        Predict and return temporalpdf parameter objects.

        Args:
            X: (n_samples, n_features) feature matrix

        Returns:
            List of parameter objects (StudentTParameters, NormalParameters, etc.)
        """
        # Import here to avoid circular imports
        from . import StudentTParameters, NormalParameters, NIGParameters

        params_array = self.predict(X)

        if self.distribution == "student_t":
            return [
                StudentTParameters(mu_0=p[0], sigma_0=p[1], nu=p[2])
                for p in params_array
            ]
        elif self.distribution == "normal":
            return [
                NormalParameters(mu=p[0], sigma=p[1])
                for p in params_array
            ]
        elif self.distribution == "nig":
            return [
                NIGParameters(mu=p[0], delta=p[1], alpha=p[2], beta=p[3])
                for p in params_array
            ]

        raise ValueError(f"Unknown distribution: {self.distribution}")

    def sample(
        self,
        X: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate samples from the predicted distributions.

        Args:
            X: (n_obs, n_features) feature matrix
            n_samples: Number of samples per observation

        Returns:
            (n_obs, n_samples) array of samples
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            params = self._model(X_t)
            samples = _sample_distribution(params, self.distribution, n_samples)

        # samples is (n_samples, batch), transpose to (batch, n_samples)
        return samples.cpu().numpy().T


# ============================================================================
# Internal implementation (torch-dependent)
# ============================================================================

def _DistributionalMLP(
    n_features: int,
    distribution: str,
    hidden_dims: list[int],
) -> "nn.Module":
    """Factory function to create the MLP model."""
    _check_torch()

    # Determine number of output parameters
    if distribution == "student_t":
        n_params = 3  # mu, sigma, nu
    elif distribution == "normal":
        n_params = 2  # mu, sigma
    elif distribution == "nig":
        n_params = 4  # mu, delta, alpha, beta
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Build shared layers
    layers: list[nn.Module] = []
    prev_dim = n_features
    for dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        ])
        prev_dim = dim

    # Create model class dynamically to capture distribution
    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(*layers)
            self.output = nn.Linear(hidden_dims[-1], n_params)
            self._distribution = distribution

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            shared = self.shared(x)
            raw = self.output(shared)
            return self._apply_constraints(raw)

        def _apply_constraints(self, raw: "torch.Tensor") -> "torch.Tensor":
            if self._distribution == "student_t":
                mu = raw[:, 0]
                sigma = nn.functional.softplus(raw[:, 1]) + 1e-6
                nu = nn.functional.softplus(raw[:, 2]) + 2.0
                return torch.stack([mu, sigma, nu], dim=1)

            elif self._distribution == "normal":
                mu = raw[:, 0]
                sigma = nn.functional.softplus(raw[:, 1]) + 1e-6
                return torch.stack([mu, sigma], dim=1)

            elif self._distribution == "nig":
                mu = raw[:, 0]
                delta = nn.functional.softplus(raw[:, 1]) + 1e-6
                alpha = nn.functional.softplus(raw[:, 2]) + 1e-6
                beta = raw[:, 3]
                beta = torch.clamp(beta, -alpha + 1e-6, alpha - 1e-6)
                return torch.stack([mu, delta, alpha, beta], dim=1)

            raise ValueError(f"Unknown distribution: {self._distribution}")

    return MLP()


def _sample_distribution(
    params: "torch.Tensor",
    distribution: str,
    n_samples: int,
) -> "torch.Tensor":
    """
    Reparameterized sampling - gradients flow through samples.

    This is the key to training with CRPS: we need to differentiate through
    the sampling process. The reparameterization trick expresses samples as
    deterministic functions of parameters plus independent noise.
    """
    _check_torch()

    batch_size = params.shape[0]
    device = params.device

    if distribution == "normal":
        mu, sigma = params[:, 0], params[:, 1]
        z = torch.randn(n_samples, batch_size, device=device)
        return mu + sigma * z

    elif distribution == "student_t":
        mu, sigma, nu = params[:, 0], params[:, 1], params[:, 2]
        # Student-t: X = mu + sigma * Z * sqrt(nu/V)
        z = torch.randn(n_samples, batch_size, device=device)
        chi2 = torch.distributions.Chi2(nu)
        v = chi2.rsample((n_samples,))
        return mu + sigma * z * torch.sqrt(nu / v)

    elif distribution == "nig":
        mu, delta, alpha, beta = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        gamma = torch.sqrt(alpha**2 - beta**2 + 1e-8)

        # Inverse Gaussian sampling
        u = torch.randn(n_samples, batch_size, device=device)
        y = u ** 2
        mu_ig = delta / gamma
        x = mu_ig + (mu_ig**2 * y) / (2 * gamma) - \
            (mu_ig / (2 * gamma)) * torch.sqrt(4 * mu_ig * gamma * y + mu_ig**2 * y**2)

        uniform = torch.rand(n_samples, batch_size, device=device)
        z_ig = torch.where(uniform <= mu_ig / (mu_ig + x), x, mu_ig**2 / x)

        w = torch.randn(n_samples, batch_size, device=device)
        return mu + beta * z_ig + torch.sqrt(z_ig + 1e-8) * w

    raise ValueError(f"Unknown distribution: {distribution}")


def _crps_loss_via_sampling(
    params: "torch.Tensor",
    y_true: "torch.Tensor",
    distribution: str,
    n_samples: int = 100,
) -> "torch.Tensor":
    """
    Compute CRPS loss via sampling.

    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    First term: how far are samples from the true value?
    Second term: how spread out is the distribution? (rewards sharpness)
    """
    _check_torch()

    samples = _sample_distribution(params, distribution, n_samples)

    # Term 1: E|X - y|
    term1 = torch.abs(samples - y_true.unsqueeze(0)).mean(dim=0)

    # Term 2: E|X - X'|
    half = n_samples // 2
    term2 = torch.abs(samples[:half] - samples[half:2*half]).mean(dim=0)

    return (term1 - 0.5 * term2).mean()


def _negative_log_likelihood(
    params: "torch.Tensor",
    y_true: "torch.Tensor",
    distribution: str,
) -> "torch.Tensor":
    """Negative log-likelihood loss (log score)."""
    _check_torch()

    if distribution == "normal":
        mu, sigma = params[:, 0], params[:, 1]
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(y_true).mean()

    elif distribution == "student_t":
        mu, sigma, nu = params[:, 0], params[:, 1], params[:, 2]
        dist = torch.distributions.StudentT(nu, mu, sigma)
        return -dist.log_prob(y_true).mean()

    elif distribution == "nig":
        raise NotImplementedError(
            "Log score for NIG not implemented. Use loss='crps' instead."
        )

    raise ValueError(f"Unknown distribution: {distribution}")


__all__ = ["DistributionalRegressor"]
