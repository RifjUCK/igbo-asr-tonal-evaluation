"""
Evaluation metrics for ASR tonal fidelity analysis.

This module implements:
- Diacritic Error Rate (DER)
- Raw Diacritic Drop Rate (RDD)
- Bootstrap confidence intervals
- Character Error Rate (CER)
"""

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from typing import Tuple, Dict


def count_diacritics(text: str) -> int:
    """
    Count diacritic characters in Igbo text.
    
    Args:
        text: Input string
        
    Returns:
        Number of diacritic characters
    """
    diacritics = set('ụọịàèìòùáéíóúẹṣ')
    return sum(1 for c in text.lower() if c in diacritics)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate character-level error rate.
    
    Args:
        reference: Ground truth text
        hypothesis: Model prediction
        
    Returns:
        CER as float (0-1)
    """
    return 1 - SequenceMatcher(None, reference.lower(), hypothesis.lower()).ratio()


def calculate_der(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Diacritic Error Rate (DER).
    
    DER = (dropped + hallucinated) / expected
    
    Args:
        df: DataFrame with columns 'diacritics_expected', 'diacritics_produced'
        
    Returns:
        Dictionary with DER statistics
    """
    E = df['diacritics_expected'].sum()
    P = df['diacritics_produced'].sum()
    D = max(0, E - P)  # dropped
    H = max(0, P - E)  # hallucinated
    
    return {
        'expected': E,
        'produced': P,
        'dropped': D,
        'hallucinated': H,
        'DER': (D + H) / E if E > 0 else 0,
        'drop_rate': D / E if E > 0 else 0,
        'hallucination_rate': H / P if P > 0 else 0
    }


def bootstrap_ci(
    data: pd.DataFrame,
    stat_fn,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: DataFrame to resample
        stat_fn: Function to compute statistic on resampled data
        n_boot: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95)
        seed: Random seed for reproducibility
        
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    if n == 0:
        return (np.nan, np.nan, np.nan)
    
    # Point estimate
    point = float(stat_fn(data))
    
    # Bootstrap resampling
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(stat_fn(data.iloc[idx]))
    
    # Percentile CI
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    
    return (point, lo, hi)


# Statistic functions for bootstrap
def stat_loss_rate(df: pd.DataFrame) -> float:
    """Diacritic loss rate (drops only)."""
    exp = df['diacritics_expected'].sum()
    loss = df['diacritic_loss'].clip(lower=0).sum()
    return loss / exp if exp > 0 else np.nan


def stat_halluc_rate(df: pd.DataFrame) -> float:
    """Hallucination rate."""
    prod = df['diacritics_produced'].sum()
    halluc = (-df['diacritic_loss']).clip(lower=0).sum()
    return halluc / prod if prod > 0 else np.nan


def stat_avg_cer(df: pd.DataFrame) -> float:
    """Average character error rate."""
    return df['character_error_rate'].mean()


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all evaluation metrics.
    
    Args:
        df: DataFrame with transcription results
        
    Returns:
        DataFrame with added metric columns
    """
    # Add diacritic counts
    df['ground_truth_diacritics'] = df['ground_truth'].apply(count_diacritics)
    df['model_output_diacritics'] = df['model_output'].apply(count_diacritics)
    df['diacritic_loss'] = df['ground_truth_diacritics'] - df['model_output_diacritics']
    
    # Add CER
    df['character_error_rate'] = df.apply(
        lambda row: character_error_rate(row['ground_truth'], row['model_output']),
        axis=1
    )
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <metadata.csv>")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(sys.argv[1])
    
    # Compute metrics
    df = compute_all_metrics(df)
    
    # Overall DER
    overall = calculate_der(df)
    print(f"\nOverall DER: {overall['DER']:.3f}")
    print(f"  Dropped: {overall['dropped']}/{overall['expected']}")
    print(f"  Hallucinated: {overall['hallucinated']}")
    
    # By category
    print("\nBy Category:")
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        cat_der = calculate_der(cat_data)
        print(f"  {cat}: {cat_der['DER']:.3f}")
    
    # Bootstrap CIs
    print("\nBootstrap 95% CIs:")
    overall_loss = bootstrap_ci(df, stat_loss_rate)
    print(f"  Overall loss: {overall_loss[0]:.3f} [{overall_loss[1]:.3f}, {overall_loss[2]:.3f}]")
    
    # Save enhanced results
    output_path = sys.argv[1].replace('.csv', '_analyzed.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
