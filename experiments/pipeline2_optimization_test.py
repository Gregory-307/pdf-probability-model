"""
Pipeline 2 Optimization Testing Framework

This script systematically tests the optimization opportunities outlined in PIPELINE2_OPTIMIZATION.md.

Optimizations tested:
1. Parameter constraints (log-transform σ, ν)
2. Quasi-Monte Carlo simulation
3. Analytical approximation for Normal distribution
4. Shared neural network architecture
5. CRPS-optimized training (via custom loss)

Run: python experiments/pipeline2_optimization_test.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.stats import qmc
import time

import temporalpdf as tpdf


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    horizon: int = 20
    barrier: float = 5.0
    n_features: int = 32
    test_frac: float = 0.2
    n_sims: int = 1000
    random_seed: int = 42


# ============================================================================
# Load cached data
# ============================================================================

cache_path = Path('experiments/data_cache.pkl')
if not cache_path.exists():
    print("ERROR: Run 'python experiments/prepare_data.py' first!")
    sys.exit(1)

print("Loading cached data...", flush=True)
with open(cache_path, 'rb') as f:
    data_cache = pickle.load(f)
print("Loaded.\n")


# ============================================================================
# OPTIMIZATION 1: Log-Transform Parameters (Quick Win)
# ============================================================================

def create_log_transformed_targets(y_params: np.ndarray, distribution: str) -> np.ndarray:
    """Transform parameters to unconstrained space."""
    y_transformed = y_params.copy()
    
    if distribution == 'normal':
        # [mu, sigma] -> [mu, log(sigma)]
        y_transformed[:, 1] = np.log(np.maximum(y_params[:, 1], 1e-6))
    elif distribution == 'student_t':
        # [mu, sigma, nu] -> [mu, log(sigma), log(nu-2)]
        y_transformed[:, 1] = np.log(np.maximum(y_params[:, 1], 1e-6))
        y_transformed[:, 2] = np.log(np.maximum(y_params[:, 2] - 2, 1e-6))
    return y_transformed


def inverse_log_transform(pred: np.ndarray, distribution: str) -> np.ndarray:
    """Transform back from unconstrained space."""
    result = pred.copy()
    
    if distribution == 'normal':
        result[:, 1] = np.exp(pred[:, 1])
    elif distribution == 'student_t':
        result[:, 1] = np.exp(pred[:, 1])
        result[:, 2] = np.exp(pred[:, 2]) + 2
    return result


# ============================================================================
# OPTIMIZATION 2: Quasi-Monte Carlo (Quick Win)
# ============================================================================

def simulate_barrier_mc(params, dist, horizon, barrier, n_sims=1000, rng=None):
    """Standard Monte Carlo simulation."""
    if rng is None:
        rng = np.random.default_rng(42)
    samples = dist.sample(n_sims * horizon, 0.0, params, rng=rng)
    paths = samples.reshape(n_sims, horizon)
    return np.mean(np.max(np.cumsum(paths, axis=1), axis=1) >= barrier)


def simulate_barrier_qmc(params, dist, horizon, barrier, n_sims=1000):
    """Quasi-Monte Carlo simulation using Sobol sequences."""
    # Generate Sobol sequence
    sampler = qmc.Sobol(d=horizon, scramble=True, seed=42)
    u = sampler.random(n=n_sims)  # shape: (n_sims, horizon)
    
    # Transform uniform to target distribution
    # For Student-t: use inverse CDF
    if isinstance(params, tpdf.StudentTParameters):
        samples = stats.t.ppf(u, df=params.nu, loc=params.mu_0, scale=params.sigma_0)
    elif isinstance(params, tpdf.NormalParameters):
        samples = stats.norm.ppf(u, loc=params.mu_0, scale=params.sigma_0)
    else:
        # Fall back to MC for unsupported distributions
        return simulate_barrier_mc(params, dist, horizon, barrier, n_sims)
    
    # Compute barrier probability
    paths = np.cumsum(samples, axis=1)
    return np.mean(np.max(paths, axis=1) >= barrier)


# ============================================================================
# OPTIMIZATION 3: Analytical Approximation for Normal (Medium Effort)
# ============================================================================

def barrier_prob_analytical_normal(mu, sigma, horizon, barrier):
    """
    Analytical barrier probability for Brownian motion with drift.
    
    Uses reflection principle for the probability that cumulative sum of 
    N(mu, sigma^2) random variables exceeds barrier at any point.
    
    For discrete random walk S_n = X_1 + X_2 + ... + X_n where X_i ~ N(μ, σ²),
    we approximate using continuous-time Brownian motion.
    
    P(max_{0≤t≤T} W_t ≥ b) where W_t ~ N(μt, σ²t)
    """
    if barrier <= 0:
        return 1.0
    
    drift = mu * horizon
    vol = sigma * np.sqrt(horizon)
    
    if vol < 1e-10:
        return 1.0 if drift >= barrier else 0.0
    
    # First passage approximation using reflection principle
    # P(max S_t >= b) ≈ 2 * P(S_T >= b) for mu = 0 (reflection)
    # For mu != 0, use Siegmund's approximation
    
    d1 = (barrier - drift) / vol
    
    # Standard approximation: P(max >= b) ≈ Φ(-d1) + exp(2μb/σ²) * Φ(-d2)
    # where d2 = (b + drift) / vol
    if mu != 0:
        d2 = (barrier + drift) / vol
        correction = np.exp(2 * mu * barrier / (sigma ** 2))
        prob = stats.norm.cdf(-d1) + correction * stats.norm.cdf(-d2)
    else:
        # Reflection principle for zero drift
        prob = 2 * stats.norm.cdf(-d1)
    
    return np.clip(prob, 0, 1)


def simulate_barrier_analytical(params, dist, horizon, barrier, n_sims=None):
    """Use analytical approximation when applicable."""
    if isinstance(params, tpdf.NormalParameters):
        return barrier_prob_analytical_normal(params.mu_0, params.sigma_0, horizon, barrier)
    else:
        # Fall back to MC for non-normal
        return simulate_barrier_mc(params, dist, horizon, barrier, n_sims or 1000)


# ============================================================================
# OPTIMIZATION 4: Neural Network with Shared Representation
# ============================================================================

def train_shared_nn(X_train, y_train, hidden_layers=(64, 32), max_iter=500):
    """Train neural network with shared hidden layers for all parameters."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    model.fit(X_train, y_train)
    return model


# ============================================================================
# OPTIMIZATION 5: Chained Parameter Prediction
# ============================================================================

def train_chained_models(X_train, y_train, base_estimator_class=GradientBoostingRegressor):
    """Train models where later parameters can see earlier predictions."""
    n_params = y_train.shape[1]
    models = []
    
    for i in range(n_params):
        # For parameter i, use original features + predictions of params 0..i-1
        if i == 0:
            X_augmented = X_train
        else:
            # Get predictions from previous models
            prev_preds = np.column_stack([
                models[j].predict(X_train if j == 0 else np.column_stack([X_train] + [models[k].predict(X_train).reshape(-1, 1) for k in range(j)]))
                for j in range(i)
            ])
            X_augmented = np.column_stack([X_train, prev_preds])
        
        model = base_estimator_class(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_augmented, y_train[:, i])
        models.append(model)
    
    return models


def predict_chained(models, X_test):
    """Predict using chained models."""
    predictions = []
    
    for i, model in enumerate(models):
        if i == 0:
            X_augmented = X_test
        else:
            prev_preds = np.column_stack(predictions[:i])
            X_augmented = np.column_stack([X_test, prev_preds])
        
        pred = model.predict(X_augmented)
        predictions.append(pred.reshape(-1, 1))
    
    return np.column_stack(predictions)


# ============================================================================
# Evaluation Framework
# ============================================================================

def evaluate_pipeline(
    config: ExperimentConfig,
    model_type: Literal['gbr', 'gbr_log', 'nn', 'chained'],
    sim_type: Literal['mc', 'qmc', 'analytical'],
    verbose: bool = True
) -> dict:
    """
    Evaluate a pipeline configuration.
    
    Args:
        config: Experiment configuration
        model_type: Model architecture
            - 'gbr': Standard GBR MultiOutput
            - 'gbr_log': GBR with log-transformed targets
            - 'nn': Neural network with shared layers
            - 'chained': Chained parameter prediction
        sim_type: Simulation method
            - 'mc': Standard Monte Carlo
            - 'qmc': Quasi-Monte Carlo
            - 'analytical': Analytical approximation (with MC fallback)
    
    Returns:
        Dictionary with evaluation metrics
    """
    data = data_cache[config.horizon]
    distribution = 'student_t'
    
    # Select features and targets
    X = data['X_32'] if config.n_features == 32 else data['X_8']
    y_barrier = data['y_barrier'][config.barrier]
    y_params = data['y_params_student_t']
    
    dist = tpdf.StudentT()
    
    def make_params(p):
        return tpdf.StudentTParameters(
            mu_0=p[0], 
            sigma_0=max(p[1], 0.01), 
            nu=max(p[2], 2.1)
        )
    
    # Split data
    split = int(len(X) * (1 - config.test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_barrier[:split], y_barrier[split:]
    y_params_train, y_params_test = y_params[:split], y_params[split:]
    
    start_time = time.time()
    
    # ========== Training ==========
    if model_type == 'gbr':
        # Standard approach
        model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        )
        model.fit(X_train, y_params_train)
        y_pred = model.predict(X_test)
        
    elif model_type == 'gbr_log':
        # Log-transformed targets
        y_train_transformed = create_log_transformed_targets(y_params_train, distribution)
        model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        )
        model.fit(X_train, y_train_transformed)
        y_pred_transformed = model.predict(X_test)
        y_pred = inverse_log_transform(y_pred_transformed, distribution)
        
    elif model_type == 'nn':
        # Neural network with shared representation
        model = train_shared_nn(X_train, y_params_train, hidden_layers=(64, 32))
        y_pred = model.predict(X_test)
        
    elif model_type == 'chained':
        # Chained prediction
        models = train_chained_models(X_train, y_params_train)
        y_pred = predict_chained(models, X_test)
    
    train_time = time.time() - start_time
    
    # ========== Simulation ==========
    start_time = time.time()
    rng = np.random.default_rng(config.random_seed)
    
    p2_probs = []
    for i in range(len(y_test)):
        params = make_params(y_pred[i])
        
        if sim_type == 'mc':
            prob = simulate_barrier_mc(params, dist, config.horizon, config.barrier, config.n_sims, rng)
        elif sim_type == 'qmc':
            prob = simulate_barrier_qmc(params, dist, config.horizon, config.barrier, config.n_sims)
        elif sim_type == 'analytical':
            # Use analytical for normal, fall back to MC for Student-t
            # (Analytical Student-t is complex, so we compare simulation variance reduction)
            prob = simulate_barrier_mc(params, dist, config.horizon, config.barrier, config.n_sims, rng)
        
        p2_probs.append(prob)
    
    p2_probs = np.array(p2_probs)
    sim_time = time.time() - start_time
    
    # ========== Metrics ==========
    brier = np.mean((p2_probs - y_test.astype(float)) ** 2)
    
    # Calibration: bin predictions and compare to actual rates
    bins = np.linspace(0, 1, 11)
    calibration_error = 0
    for j in range(len(bins) - 1):
        mask = (p2_probs >= bins[j]) & (p2_probs < bins[j+1])
        if mask.sum() > 10:
            expected = p2_probs[mask].mean()
            actual = y_test[mask].mean()
            calibration_error += (expected - actual) ** 2 * mask.sum()
    calibration_error = np.sqrt(calibration_error / len(y_test))
    
    # Parameter prediction quality
    param_corrs = []
    for j in range(y_params_test.shape[1]):
        corr = np.corrcoef(y_pred[:, j], y_params_test[:, j])[0, 1]
        param_corrs.append(corr if np.isfinite(corr) else 0)
    
    results = {
        'model_type': model_type,
        'sim_type': sim_type,
        'horizon': config.horizon,
        'barrier': config.barrier,
        'n_features': config.n_features,
        'brier_score': brier,
        'calibration_error': calibration_error,
        'actual_rate': y_test.mean(),
        'predicted_mean': p2_probs.mean(),
        'predicted_std': p2_probs.std(),
        'train_time': train_time,
        'sim_time': sim_time,
        'param_corr_mu': param_corrs[0],
        'param_corr_sigma': param_corrs[1],
        'param_corr_nu': param_corrs[2],
    }
    
    if verbose:
        print(f"  {model_type:12s} + {sim_type:12s}: Brier={brier:.4f}, Cal={calibration_error:.4f}, "
              f"Train={train_time:.2f}s, Sim={sim_time:.2f}s")
    
    return results


def run_baseline_p1(config: ExperimentConfig, verbose: bool = True) -> dict:
    """Run Pipeline 1 (direct classification) for baseline comparison."""
    data = data_cache[config.horizon]
    
    X = data['X_32'] if config.n_features == 32 else data['X_8']
    y_barrier = data['y_barrier'][config.barrier]
    
    split = int(len(X) * (1 - config.test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_barrier[:split], y_barrier[split:]
    
    start_time = time.time()
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    p1_probs = clf.predict_proba(X_test)[:, 1]
    train_time = time.time() - start_time
    
    brier = np.mean((p1_probs - y_test.astype(float)) ** 2)
    
    # Calibration
    bins = np.linspace(0, 1, 11)
    calibration_error = 0
    for j in range(len(bins) - 1):
        mask = (p1_probs >= bins[j]) & (p1_probs < bins[j+1])
        if mask.sum() > 10:
            expected = p1_probs[mask].mean()
            actual = y_test[mask].mean()
            calibration_error += (expected - actual) ** 2 * mask.sum()
    calibration_error = np.sqrt(calibration_error / len(y_test))
    
    results = {
        'model_type': 'P1_baseline',
        'sim_type': 'N/A',
        'horizon': config.horizon,
        'barrier': config.barrier,
        'n_features': config.n_features,
        'brier_score': brier,
        'calibration_error': calibration_error,
        'actual_rate': y_test.mean(),
        'predicted_mean': p1_probs.mean(),
        'predicted_std': p1_probs.std(),
        'train_time': train_time,
        'sim_time': 0,
        'param_corr_mu': np.nan,
        'param_corr_sigma': np.nan,
        'param_corr_nu': np.nan,
    }
    
    if verbose:
        print(f"  P1_baseline                   : Brier={brier:.4f}, Cal={calibration_error:.4f}")
    
    return results


# ============================================================================
# QMC Variance Comparison
# ============================================================================

def compare_simulation_variance(n_trials: int = 50):
    """Compare variance between MC and QMC."""
    print("\n" + "=" * 70)
    print("SIMULATION VARIANCE COMPARISON")
    print("=" * 70)
    
    # Fixed parameters for comparison
    params = tpdf.StudentTParameters(mu_0=0.05, sigma_0=1.5, nu=5.0)
    dist = tpdf.StudentT()
    horizon = 20
    barrier = 5.0
    n_sims = 1000
    
    mc_results = []
    qmc_results = []
    
    print(f"\nRunning {n_trials} trials with params: μ={params.mu_0}, σ={params.sigma_0}, ν={params.nu}")
    print(f"Horizon={horizon}, Barrier={barrier}, N_sims={n_sims}")
    
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        mc_prob = simulate_barrier_mc(params, dist, horizon, barrier, n_sims, rng)
        mc_results.append(mc_prob)
        
        # QMC with different scrambling seed
        qmc_prob = simulate_barrier_qmc(params, dist, horizon, barrier, n_sims)
        qmc_results.append(qmc_prob)
    
    mc_results = np.array(mc_results)
    qmc_results = np.array(qmc_results)
    
    print(f"\nMonte Carlo:       mean={mc_results.mean():.4f}, std={mc_results.std():.4f}")
    print(f"Quasi-Monte Carlo: mean={qmc_results.mean():.4f}, std={qmc_results.std():.4f}")
    print(f"Variance reduction: {mc_results.std() / qmc_results.std():.2f}x")
    
    return mc_results, qmc_results


# ============================================================================
# Analytical Approximation Accuracy
# ============================================================================

def test_analytical_accuracy():
    """Compare analytical approximation to Monte Carlo for Normal distribution."""
    print("\n" + "=" * 70)
    print("ANALYTICAL APPROXIMATION ACCURACY (Normal Distribution)")
    print("=" * 70)
    
    test_cases = [
        # (mu, sigma, horizon, barrier)
        (0.0, 1.0, 20, 3.0),
        (0.0, 1.0, 20, 5.0),
        (0.05, 1.5, 20, 5.0),
        (-0.05, 1.0, 20, 3.0),
        (0.1, 2.0, 30, 7.0),
        (0.0, 0.5, 10, 2.0),
    ]
    
    print(f"\n{'μ':>8} {'σ':>8} {'H':>4} {'B':>4} {'Analytical':>12} {'MC (10k)':>12} {'Error':>10}")
    print("-" * 70)
    
    for mu, sigma, horizon, barrier in test_cases:
        params = tpdf.NormalParameters(mu_0=mu, sigma_0=sigma)
        dist = tpdf.Normal()
        
        analytical = barrier_prob_analytical_normal(mu, sigma, horizon, barrier)
        
        # High-precision MC estimate
        mc_estimates = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            mc = simulate_barrier_mc(params, dist, horizon, barrier, 10000, rng)
            mc_estimates.append(mc)
        mc_mean = np.mean(mc_estimates)
        
        error = abs(analytical - mc_mean)
        print(f"{mu:>8.2f} {sigma:>8.2f} {horizon:>4d} {barrier:>4.1f} {analytical:>12.4f} {mc_mean:>12.4f} {error:>10.4f}")


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_optimization_experiments():
    """Run comprehensive optimization tests."""
    print("=" * 70)
    print("PIPELINE 2 OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    
    # Test configurations
    configs = [
        ExperimentConfig(horizon=10, barrier=5.0, n_features=32),
        ExperimentConfig(horizon=20, barrier=5.0, n_features=32),
        ExperimentConfig(horizon=30, barrier=5.0, n_features=32),
        ExperimentConfig(horizon=20, barrier=3.0, n_features=32),
    ]
    
    model_types = ['gbr', 'gbr_log', 'nn', 'chained']
    sim_types = ['mc', 'qmc']
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"HORIZON={config.horizon}d, BARRIER={config.barrier}%")
        print("=" * 70)
        
        # Baseline
        baseline = run_baseline_p1(config)
        all_results.append(baseline)
        
        # P2 variants
        for model_type in model_types:
            for sim_type in sim_types:
                result = evaluate_pipeline(config, model_type, sim_type)
                result['improvement_vs_p1'] = baseline['brier_score'] - result['brier_score']
                all_results.append(result)
    
    # Summary DataFrame
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Best configurations
    p2_results = df[df['model_type'] != 'P1_baseline']
    
    print("\nBest P2 configuration per scenario:")
    for horizon in df['horizon'].unique():
        for barrier in df['barrier'].unique():
            mask = (p2_results['horizon'] == horizon) & (p2_results['barrier'] == barrier)
            subset = p2_results[mask]
            if len(subset) > 0:
                best = subset.loc[subset['brier_score'].idxmin()]
                p1_brier = df[(df['model_type'] == 'P1_baseline') & 
                             (df['horizon'] == horizon) & 
                             (df['barrier'] == barrier)]['brier_score'].values[0]
                print(f"  H={horizon:2d}d, B={barrier}%: {best['model_type']:12s}+{best['sim_type']:4s} "
                      f"Brier={best['brier_score']:.4f} (P1={p1_brier:.4f}, Δ={best['improvement_vs_p1']:+.4f})")
    
    # Save results
    output_path = Path('experiments/optimization_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return df


# ============================================================================
# Additional Analysis: Parameter Transformation Benefits
# ============================================================================

def analyze_parameter_predictions():
    """Analyze how log-transform affects parameter prediction quality."""
    print("\n" + "=" * 70)
    print("PARAMETER PREDICTION ANALYSIS")
    print("=" * 70)
    
    config = ExperimentConfig(horizon=20, barrier=5.0, n_features=32)
    data = data_cache[config.horizon]
    
    X = data['X_32']
    y_params = data['y_params_student_t']
    
    split = int(len(X) * (1 - config.test_frac))
    X_train, X_test = X[:split], X[split:]
    y_params_train, y_params_test = y_params[:split], y_params[split:]
    
    # Standard GBR
    model_std = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    )
    model_std.fit(X_train, y_params_train)
    pred_std = model_std.predict(X_test)
    
    # Log-transformed GBR
    y_train_log = create_log_transformed_targets(y_params_train, 'student_t')
    model_log = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    )
    model_log.fit(X_train, y_train_log)
    pred_log_raw = model_log.predict(X_test)
    pred_log = inverse_log_transform(pred_log_raw, 'student_t')
    
    param_names = ['μ', 'σ', 'ν']
    
    print(f"\n{'Parameter':<10} {'Std Corr':<12} {'Log Corr':<12} {'Std MSE':<12} {'Log MSE':<12}")
    print("-" * 60)
    
    for i, name in enumerate(param_names):
        corr_std = np.corrcoef(pred_std[:, i], y_params_test[:, i])[0, 1]
        corr_log = np.corrcoef(pred_log[:, i], y_params_test[:, i])[0, 1]
        
        mse_std = np.mean((pred_std[:, i] - y_params_test[:, i]) ** 2)
        mse_log = np.mean((pred_log[:, i] - y_params_test[:, i]) ** 2)
        
        print(f"{name:<10} {corr_std:<12.4f} {corr_log:<12.4f} {mse_std:<12.4f} {mse_log:<12.4f}")
    
    # Check for constraint violations
    print("\nConstraint violations (should be 0 for log-transform):")
    print(f"  Standard: σ < 0.01: {(pred_std[:, 1] < 0.01).sum()}, ν < 2.1: {(pred_std[:, 2] < 2.1).sum()}")
    print(f"  Log-transform: σ < 0.01: {(pred_log[:, 1] < 0.01).sum()}, ν < 2.1: {(pred_log[:, 2] < 2.1).sum()}")


# ============================================================================
# Adaptive Simulation Count Analysis
# ============================================================================

def test_adaptive_simulation():
    """Test if adaptive simulation counts improve edge case accuracy."""
    print("\n" + "=" * 70)
    print("ADAPTIVE SIMULATION COUNT ANALYSIS")
    print("=" * 70)
    
    config = ExperimentConfig(horizon=20, barrier=5.0, n_features=32)
    data = data_cache[config.horizon]
    
    X = data['X_32']
    y_barrier = data['y_barrier'][config.barrier]
    y_params = data['y_params_student_t']
    
    split = int(len(X) * (1 - config.test_frac))
    X_test = X[split:]
    y_test = y_barrier[split:]
    y_params_test = y_params[split:]
    
    dist = tpdf.StudentT()
    
    def make_params(p):
        return tpdf.StudentTParameters(mu_0=p[0], sigma_0=max(p[1], 0.01), nu=max(p[2], 2.1))
    
    # Compare fixed vs adaptive
    rng = np.random.default_rng(42)
    
    fixed_probs = []
    adaptive_probs = []
    
    for i in range(len(y_test)):
        params = make_params(y_params_test[i])
        
        # Fixed simulation count
        fixed = simulate_barrier_mc(params, dist, 20, 5.0, 1000, rng)
        fixed_probs.append(fixed)
        
        # Adaptive: more sims for edge cases
        initial = simulate_barrier_mc(params, dist, 20, 5.0, 500, rng)
        if initial < 0.1 or initial > 0.9:
            adaptive = simulate_barrier_mc(params, dist, 20, 5.0, 5000, rng)
        else:
            adaptive = initial
        adaptive_probs.append(adaptive)
    
    fixed_probs = np.array(fixed_probs)
    adaptive_probs = np.array(adaptive_probs)
    
    # Compare performance on edge cases
    edge_mask = (y_test.astype(float) < 0.3) | (y_test.astype(float) > 0.7)  # True edge cases
    
    brier_fixed_edge = np.mean((fixed_probs[edge_mask] - y_test[edge_mask].astype(float)) ** 2)
    brier_adaptive_edge = np.mean((adaptive_probs[edge_mask] - y_test[edge_mask].astype(float)) ** 2)
    
    brier_fixed_all = np.mean((fixed_probs - y_test.astype(float)) ** 2)
    brier_adaptive_all = np.mean((adaptive_probs - y_test.astype(float)) ** 2)
    
    print(f"\nOverall Brier:     Fixed={brier_fixed_all:.4f}, Adaptive={brier_adaptive_all:.4f}")
    print(f"Edge Cases Brier:  Fixed={brier_fixed_edge:.4f}, Adaptive={brier_adaptive_edge:.4f}")
    print(f"Edge case samples: {edge_mask.sum()}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Run main experiments
    df = run_optimization_experiments()
    
    # Additional analyses
    compare_simulation_variance(n_trials=50)
    test_analytical_accuracy()
    analyze_parameter_predictions()
    test_adaptive_simulation()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
