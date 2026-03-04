"""
Utility functions for ASR evaluation.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_metadata(filepath: str) -> pd.DataFrame:
    """
    Load and validate metadata CSV.
    
    Args:
        filepath: Path to metadata.csv
        
    Returns:
        Validated DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required = [
        'file_name', 'ground_truth', 'model_output',
        'category', 'language'
    ]
    
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def print_summary(df: pd.DataFrame):
    """
    Print dataset summary statistics.
    
    Args:
        df: DataFrame with analysis results
    """
    print("="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Languages: {', '.join(df['language'].unique())}")
    
    print("\nSamples by category:")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat}: {count}")
    
    if 'diacritics_expected' in df.columns:
        print(f"\nTotal expected diacritics: {df['diacritics_expected'].sum()}")
        print(f"Total produced diacritics: {df['diacritics_produced'].sum()}")
        print(f"Net loss: {df['diacritic_loss'].sum()}")
    
    if 'character_error_rate' in df.columns:
        print(f"\nAverage CER: {df['character_error_rate'].mean():.3f}")
        print(f"  Min: {df['character_error_rate'].min():.3f}")
        print(f"  Max: {df['character_error_rate'].max():.3f}")


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate category-level statistics.
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        DataFrame with category statistics
    """
    stats = df.groupby('category').agg({
        'file_name': 'count',
        'diacritic_loss': 'sum',
        'diacritics_expected': 'sum',
        'character_error_rate': 'mean'
    }).rename(columns={
        'file_name': 'samples',
        'character_error_rate': 'avg_cer'
    })
    
    stats['loss_rate'] = stats['diacritic_loss'] / stats['diacritics_expected']
    
    return stats


def export_results(
    df: pd.DataFrame,
    category_stats: pd.DataFrame,
    output_dir: str = "results"
):
    """
    Export analysis results to CSV files.
    
    Args:
        df: Full analysis DataFrame
        category_stats: Category-level statistics
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save full results
    df.to_csv(output_path / "full_analysis.csv", index=False)
    
    # Save category summary
    category_stats.to_csv(output_path / "category_summary.csv")
    
    print(f"\nResults exported to: {output_path}")
    print(f"  - full_analysis.csv")
    print(f"  - category_summary.csv")


def validate_audio_files(
    metadata_path: str,
    audio_dir: str = "data/audio"
) -> List[str]:
    """
    Check if all audio files referenced in metadata exist.
    
    Args:
        metadata_path: Path to metadata CSV
        audio_dir: Directory containing audio files
        
    Returns:
        List of missing files (empty if all present)
    """
    df = pd.read_csv(metadata_path)
    audio_path = Path(audio_dir)
    
    missing = []
    for filename in df['file_name']:
        filepath = audio_path / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        print(f"⚠️  Missing {len(missing)} audio files:")
        for f in missing:
            print(f"  - {f}")
    else:
        print(f"✓ All {len(df)} audio files present")
    
    return missing


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python utils.py <metadata.csv>")
        sys.exit(1)
    
    # Load and summarize
    df = load_metadata(sys.argv[1])
    print_summary(df)
    
    # Category breakdown
    print("\n" + "="*70)
    print("CATEGORY BREAKDOWN")
    print("="*70)
    stats = category_breakdown(df)
    print(stats.to_string())
