"""
Visualization functions for ASR tonal fidelity analysis.

Generates publication-quality figures for:
- Diacritic loss by category
- CER vs diacritic loss scatter
- Bootstrap confidence intervals
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# Set publication-quality defaults
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def plot_loss_by_category(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Create horizontal bar chart of diacritic loss by category.
    
    Args:
        df: DataFrame with columns 'category', 'diacritic_loss', 'diacritics_expected'
        output_path: Where to save figure (optional)
        figsize: Figure size in inches
    """
    # Calculate loss rate by category
    category_stats = df.groupby('category').agg({
        'diacritic_loss': 'sum',
        'diacritics_expected': 'sum'
    })
    category_stats['loss_rate'] = (
        category_stats['diacritic_loss'] / category_stats['diacritics_expected'] * 100
    )
    category_stats = category_stats.sort_values('loss_rate')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color code by severity
    colors = []
    for rate in category_stats['loss_rate']:
        if rate > 50:
            colors.append('#d62728')  # Red (severe)
        elif rate > 20:
            colors.append('#ff7f0e')  # Orange (moderate)
        elif rate > 0:
            colors.append('#2ca02c')  # Green (low)
        else:
            colors.append('#9467bd')  # Purple (hallucination)
    
    # Plot
    bars = ax.barh(category_stats.index, category_stats['loss_rate'], color=colors)
    
    # Add value labels
    for i, (idx, row) in enumerate(category_stats.iterrows()):
        value = row['loss_rate']
        ax.text(
            value + 2 if value >= 0 else value - 2,
            i,
            f"{value:.1f}%",
            va='center',
            ha='left' if value >= 0 else 'right',
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xlabel('Diacritic Loss Rate (%)', fontweight='bold')
    ax.set_title('Diacritic Loss by Error Category', fontweight='bold', pad=15)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_cer_vs_loss(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Scatter plot of CER vs diacritic loss.
    
    Args:
        df: DataFrame with columns 'character_error_rate', 'diacritic_loss', 'category'
        output_path: Where to save figure (optional)
        figsize: Figure size in inches
    """
    # Calculate per-sample loss rate
    df['loss_rate'] = df['diacritic_loss'] / df['diacritics_expected'] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by category
    categories = df['category'].unique()
    colors = sns.color_palette("husl", len(categories))
    
    for i, cat in enumerate(categories):
        cat_data = df[df['category'] == cat]
        ax.scatter(
            cat_data['character_error_rate'] * 100,
            cat_data['loss_rate'],
            label=cat.replace('_', ' ').title(),
            alpha=0.7,
            s=100,
            color=colors[i]
        )
    
    # Formatting
    ax.set_xlabel('Character Error Rate (%)', fontweight='bold')
    ax.set_ylabel('Diacritic Loss Rate (%)', fontweight='bold')
    ax.set_title('CER vs. Diacritic Loss (by Sample)', fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_bootstrap_ci(
    bootstrap_results: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Forest plot of bootstrap confidence intervals.
    
    Args:
        bootstrap_results: Dict mapping category -> (point, lo, hi)
        output_path: Where to save figure (optional)
        figsize: Figure size in inches
    """
    # Prepare data
    categories = list(bootstrap_results.keys())
    points = [v[0] * 100 for v in bootstrap_results.values()]
    lows = [v[1] * 100 for v in bootstrap_results.values()]
    highs = [v[2] * 100 for v in bootstrap_results.values()]
    errors = [[p - l for p, l in zip(points, lows)],
              [h - p for h, p in zip(highs, points)]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot error bars
    y_pos = np.arange(len(categories))
    ax.errorbar(
        points, y_pos, xerr=errors,
        fmt='o', markersize=8, capsize=5, capthick=2,
        color='#1f77b4', ecolor='#1f77b4', alpha=0.8
    )
    
    # Add 50% threshold line
    ax.axvline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% threshold')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace('_', ' ').title() for c in categories])
    ax.set_xlabel('Diacritic Loss Rate (%)', fontweight='bold')
    ax.set_title('Bootstrap 95% Confidence Intervals', fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def generate_all_figures(
    df: pd.DataFrame,
    bootstrap_results: dict,
    output_dir: str = "visualizations"
):
    """
    Generate all figures and save to output directory.
    
    Args:
        df: DataFrame with analysis results
        bootstrap_results: Bootstrap CI results
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Generating figures...")
    
    # Figure 1: Loss by category
    plot_loss_by_category(
        df,
        output_path=output_path / "fig1_loss_by_category.png"
    )
    
    # Figure 2: CER vs loss
    plot_cer_vs_loss(
        df,
        output_path=output_path / "fig2_cer_vs_diacritic_loss.png"
    )
    
    # Figure 3: Bootstrap CIs
    plot_bootstrap_ci(
        bootstrap_results,
        output_path=output_path / "fig3_bootstrap_ci.png"
    )
    
    print(f"\nAll figures saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <metadata_analyzed.csv>")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(sys.argv[1])
    
    # Example bootstrap results (replace with actual computation)
    bootstrap_results = {
        'overall': (0.526, 0.303, 0.697),
        'tonal_diacritics': (0.755, 0.571, 0.897),
        'script_hallucination': (0.360, 0.087, 0.680),
    }
    
    # Generate all figures
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
    generate_all_figures(df, bootstrap_results, output_dir)
