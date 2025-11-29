import sys
import struct
import os
import argparse
import math

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

def monte_carlo_pi(filepath, output_image="pi_plot.png"):
    if not os.path.exists(filepath):
        print(f"[-] Error: File '{filepath}' not found.")
        return

    file_size = os.path.getsize(filepath)
    print(f"[*] Analyzing file: {filepath}")
    print(f"[*] File Size: {file_size / 1024:.2f} KB")

    # Each point needs 8 bytes (4 for X, 4 for Y)
    points_count = file_size // 8
    
    if points_count < 1000:
        print("[-] Error: File too small. We need at least 8KB of data.")
        return

    print(f"[*] Simulating with {points_count:,} points...")

    inside_circle = 0
    
    # Lists for plotting (we will only plot a sample to avoid crashing RAM)
    plot_limit = 10000 
    x_inside, y_inside = [], []
    x_outside, y_outside = [], []

    with open(filepath, "rb") as f:
        count = 0
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                break
                
            x_raw, y_raw = struct.unpack("II", chunk)
            x = x_raw / 4294967295.0
            y = y_raw / 4294967295.0
            
            if (x*x + y*y) <= 1.0:
                inside_circle += 1
                if count < plot_limit:
                    x_inside.append(x)
                    y_inside.append(y)
            else:
                if count < plot_limit:
                    x_outside.append(x)
                    y_outside.append(y)
            
            count += 1

    # Calculation
    pi_estimate = 4.0 * (inside_circle / points_count)
    actual_pi = 3.14159265359
    error_percent = abs((pi_estimate - actual_pi) / actual_pi) * 100

    print("\n=== RESULTS ===")
    print(f"Points Inside: {inside_circle:,}")
    print(f"Total Points:  {points_count:,}")
    print(f"Estimated Pi:  {pi_estimate:.5f}")
    print(f"Actual Pi:     {actual_pi:.5f}")
    print(f"Error:         {error_percent:.4f}%")

    # Visualization
    if HAS_PLOT:
        print(f"\n[*] Generating visualization ({min(points_count, plot_limit)} points)...")
        plt.figure(figsize=(8, 8))
        
        # Plot points
        plt.scatter(x_inside, y_inside, color='green', s=1, alpha=0.5, label='Inside')
        plt.scatter(x_outside, y_outside, color='red', s=1, alpha=0.5, label='Outside')
        
        # Draw the circle arc
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        
        plt.title(f"Monte Carlo Pi Estimation\nPi ≈ {pi_estimate:.5f} (Error: {error_percent:.2f}%)")
        plt.xlabel("X (Normalized Random)")
        plt.ylabel("Y (Normalized Random)")
        plt.legend(loc="upper right")
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.savefig(output_image, dpi=150)
        print(f"[+] Plot saved to: {output_image}")
    else:
        print("[-] Matplotlib not found. Skipping plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to binary file")
    args = parser.parse_args()
    
    monte_carlo_pi(args.file)