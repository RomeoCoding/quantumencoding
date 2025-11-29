#!/usr/bin/env python3
"""
True Random Number Generator (TRNG) - Main Entry Point

This program generates true random numbers using physical noise sources:
1. Quantum Shot Noise from webcam CMOS/CCD sensors
2. Thermal (Johnson-Nyquist) Noise from microphone circuits

The entropy is mixed, timestamped, and whitened using SHA-256 to produce
cryptographically suitable random data.

ECE Physics Background:
-----------------------
TRUE vs PSEUDO Random Number Generation:

Pseudo-Random Number Generators (PRNGs):
- Algorithmic (deterministic) - same seed produces same sequence
- Examples: Mersenne Twister, Linear Congruential Generator
- Fast but cryptographically weak if seed is known
- Period is finite (though can be astronomically large)

True Random Number Generators (TRNGs):
- Based on physical phenomena (quantum mechanics, thermodynamics)
- Non-deterministic - cannot be predicted even with full system knowledge
- Slower than PRNGs (limited by physical sampling rate)
- No period - each bit is independent

This implementation uses two fundamental physical noise sources:

1. QUANTUM SHOT NOISE (Webcam):
   - Photons are quantized (E = hν where h is Planck's constant)
   - Photon arrival follows Poisson statistics: P(n) = (λ^n × e^-λ) / n!
   - Variance σ² = mean λ (fundamental quantum limit)
   - This randomness is irreducible - it's the nature of quantum mechanics

2. THERMAL NOISE (Microphone):
   - Electrons in conductors undergo Brownian motion due to temperature
   - Described by Nyquist formula: V²_rms = 4kTRΔf
   - At room temperature (300K), 1kΩ resistor has ~4 μV/√Hz noise density
   - This noise is unavoidable at any temperature above absolute zero

Output Validation:
------------------
The program generates validation plots:
1. Histogram - Should show uniform distribution across all byte values (0-255)
2. Bitmap - Should show no visible patterns (salt-and-pepper appearance)

A good random source should:
- Pass NIST SP 800-22 statistical tests
- Have entropy close to 8 bits per byte
- Show no correlations between successive values

Usage:
    python main.py [--output OUTPUT_FILE] [--size SIZE_KB]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Conditional matplotlib import for headless environments
try:
    import matplotlib
    # Use non-interactive backend for headless environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization disabled.")

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from engine import create_entropy_engine


# Constants
DEFAULT_SIZE_KB = 100  # 100 KB as specified
DEFAULT_OUTPUT_FILE = 'random_output.bin'
HISTOGRAM_OUTPUT = 'validation_histogram.png'
BITMAP_OUTPUT = 'validation_bitmap.png'

import gc # Add this at the top with other imports

def generate_random_data(size_bytes: int) -> bytes:
    """
    Generate random data in memory-safe chunks to prevent MemoryError.
    """
    print(f"Generating {size_bytes:,} bytes of random data...")
    
    # Chunk size: 1 MB (Safe for almost any laptop)
    CHUNK_SIZE = 1024 * 1024 
    
    final_buffer = bytearray()
    generated_so_far = 0
    
    with create_entropy_engine() as engine:
        while generated_so_far < size_bytes:
            # Calculate how much to generate this round
            remaining = size_bytes - generated_so_far
            current_chunk_size = min(CHUNK_SIZE, remaining)
            
            # Generate the small chunk
            chunk = engine.generate_random_bytes(current_chunk_size)
            final_buffer.extend(chunk)
            
            generated_so_far += len(chunk)
            
            # Progress Report
            percent = (generated_so_far / size_bytes) * 100
            print(f"   Progress: {percent:.1f}% ({generated_so_far:,} bytes) - RAM Cleanup...")
            
            # CRITICAL: Force Python to clean up the discarded video frames immediately
            gc.collect()

    print(f"Generated {len(final_buffer):,} bytes successfully.")
    return bytes(final_buffer)

def generate_random_data(size_bytes: int) -> bytes:
    """
    Generate random data using the entropy engine.
    
    ECE Note: The entropy rate depends on:
    - Webcam frame rate (~30 fps) × frame size (~230KB) = ~7 MB/s raw entropy
    - Audio sample rate (44.1kHz) × sample size (2 bytes) = ~88 KB/s raw entropy
    - SHA-256 compression ratio (input -> 32 bytes output)
    
    Actual throughput depends on the slowest source and processing overhead.
    
    Args:
        size_bytes: Number of bytes to generate.
        
    Returns:
        Random bytes of the specified length.
    """
    print(f"Generating {size_bytes:,} bytes of random data...")
    
    # Create entropy engine (hardware or fallback)
    with create_entropy_engine() as engine:
        # Generate random bytes
        random_data = engine.generate_random_bytes(size_bytes)
    
    print(f"Generated {len(random_data):,} bytes successfully.")
    return random_data


def save_to_binary_file(data: bytes, filepath: str) -> None:
    """
    Save random data to a binary file.
    
    Args:
        data: Random bytes to save.
        filepath: Path to output file.
    """
    with open(filepath, 'wb') as f:
        f.write(data)
    print(f"Saved random data to: {filepath}")


def create_histogram_plot(data: bytes, output_path: str) -> None:
    """
    Create a histogram showing the distribution of byte values.
    
    ECE Validation: For a good random source, the histogram should be
    approximately uniform across all 256 possible byte values (0-255).
    
    Statistical expectation for N bytes:
    - Mean count per bin: N / 256
    - Standard deviation: √(N × (1/256) × (255/256)) ≈ √(N/256)
    
    Chi-squared test can quantify deviation from uniform distribution.
    
    Args:
        data: Random bytes to analyze.
        output_path: Path for output PNG file.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping histogram (matplotlib not available)")
        return
    
    # Convert bytes to numpy array for analysis
    byte_array = np.frombuffer(data, dtype=np.uint8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram with 256 bins (one per possible byte value)
    counts, bins, patches = ax.hist(
        byte_array, 
        bins=256, 
        range=(0, 256),
        density=False,
        color='steelblue',
        edgecolor='none',
        alpha=0.7
    )
    
    # Calculate expected count for uniform distribution
    expected_count = len(byte_array) / 256
    ax.axhline(y=expected_count, color='red', linestyle='--', 
               label=f'Expected uniform: {expected_count:.1f}')
    
    # Calculate chi-squared statistic
    chi_squared = np.sum((counts - expected_count) ** 2 / expected_count)
    # Degrees of freedom = 256 - 1 = 255
    # For p=0.05, chi-squared critical value ≈ 293
    
    # Add labels and title
    ax.set_xlabel('Byte Value (0-255)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(
        f'TRNG Output Histogram - {len(byte_array):,} bytes\n'
        f'χ² = {chi_squared:.1f} (uniform expectation: ~255)',
        fontsize=14
    )
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # ECE annotation
    textstr = (
        'ECE Validation:\n'
        '• Uniform distribution indicates good randomness\n'
        '• Peaks/valleys suggest bias in entropy source\n'
        '• χ² ≈ 255 expected for ideal uniform distribution'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {output_path}")
    print(f"Chi-squared statistic: {chi_squared:.1f} (expected ~255 for uniform)")


def create_bitmap_plot(data: bytes, output_path: str) -> None:
    """
    Create a bitmap visualization of the random data.
    
    ECE Validation: The bitmap should appear as uniform "salt and pepper"
    noise with no visible patterns, stripes, or clusters.
    
    Visual patterns indicate:
    - Horizontal stripes: Correlation between successive bytes
    - Vertical stripes: Periodic patterns in the data
    - Clusters: Non-uniform distribution
    - Diagonal patterns: Linear correlation structure
    
    A good TRNG should produce a completely featureless bitmap.
    
    Args:
        data: Random bytes to visualize.
        output_path: Path for output PNG file.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping bitmap (matplotlib not available)")
        return
    
    # Convert bytes to numpy array
    byte_array = np.frombuffer(data, dtype=np.uint8)
    
    # Calculate dimensions for roughly square bitmap
    total_pixels = len(byte_array)
    width = int(np.sqrt(total_pixels))
    height = total_pixels // width
    
    # Truncate to fit exact rectangle
    used_pixels = width * height
    image_data = byte_array[:used_pixels].reshape((height, width))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display as grayscale image
    im = ax.imshow(image_data, cmap='gray', interpolation='nearest',
                   vmin=0, vmax=255)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Byte Value (0-255)', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(
        f'TRNG Output Bitmap - {used_pixels:,} bytes ({width}×{height})\n'
        'Visual Test: Should appear as uniform noise with no patterns',
        fontsize=14
    )
    
    # ECE annotation
    textstr = (
        'ECE Visual Validation:\n'
        '• Good: Uniform "salt-and-pepper" noise\n'
        '• Bad: Visible stripes, patterns, or clusters\n'
        '• Patterns indicate correlation or bias'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, color='black')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Bitmap saved to: {output_path}")


def validate_randomness(data: bytes, histogram_path: str, bitmap_path: str) -> None:
    """
    Perform validation of the random data and generate plots.
    
    ECE Note: These are visual validation methods. For production use,
    consider running formal statistical tests such as:
    - NIST SP 800-22 Statistical Test Suite
    - Diehard/Dieharder tests
    - TestU01 (Crush, BigCrush)
    
    Args:
        data: Random bytes to validate.
        histogram_path: Output path for histogram plot.
        bitmap_path: Output path for bitmap plot.
    """
    print("\n=== Randomness Validation ===")
    
    # Basic statistics
    byte_array = np.frombuffer(data, dtype=np.uint8)
    
    # Calculate entropy estimate (bits per byte)
    # Shannon entropy: H = -Σ p(x) log2(p(x))
    unique, counts = np.unique(byte_array, return_counts=True)
    probabilities = counts / len(byte_array)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    print(f"Data size: {len(data):,} bytes")
    print(f"Mean value: {np.mean(byte_array):.2f} (expected: 127.5)")
    print(f"Std deviation: {np.std(byte_array):.2f} (expected: ~73.9)")
    print(f"Shannon entropy: {entropy:.4f} bits/byte (max: 8.0)")
    print(f"Entropy efficiency: {entropy/8*100:.2f}%")
    
    # Generate plots
    create_histogram_plot(data, histogram_path)
    create_bitmap_plot(data, bitmap_path)


def main():
    """
    Main entry point for the TRNG application.
    
    Generates random data, saves to file, and creates validation plots.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='True Random Number Generator using physical noise sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ECE Physics Notes:
  This TRNG harvests entropy from two fundamental physical noise sources:
  1. Quantum shot noise from webcam sensors (photon counting statistics)
  2. Thermal Johnson-Nyquist noise from microphone circuits
  
  The raw entropy is mixed and whitened using SHA-256 to produce
  cryptographically suitable random data.
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output binary file path (default: {DEFAULT_OUTPUT_FILE})'
    )
    
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=DEFAULT_SIZE_KB,
        help=f'Size in kilobytes to generate (default: {DEFAULT_SIZE_KB} KB)'
    )
    
    parser.add_argument(
        '--histogram',
        type=str,
        default=HISTOGRAM_OUTPUT,
        help=f'Histogram output path (default: {HISTOGRAM_OUTPUT})'
    )
    
    parser.add_argument(
        '--bitmap',
        type=str,
        default=BITMAP_OUTPUT,
        help=f'Bitmap output path (default: {BITMAP_OUTPUT})'
    )
    
    args = parser.parse_args()
    
    # Convert KB to bytes
    size_bytes = args.size * 1024
    
    print("=" * 60)
    print("True Random Number Generator (TRNG)")
    print("Using Quantum Shot Noise & Thermal Noise")
    print("=" * 60)
    print()
    
    # Generate random data
    random_data = generate_random_data(size_bytes)
    
    # Save to binary file
    save_to_binary_file(random_data, args.output)
    
    # Validate and create plots
    validate_randomness(random_data, args.histogram, args.bitmap)
    
    print("\n" + "=" * 60)
    print("TRNG generation complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
