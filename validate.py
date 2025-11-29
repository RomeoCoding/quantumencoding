import sys
import struct
import os
import argparse

def monte_carlo_pi(filepath):
    """
    Estimates Pi using Monte Carlo simulation.
    
    The Logic:
    1. Imagine a square with side length 1. Area = 1.
    2. Inscribe a quarter-circle inside it. Area = Pi/4.
    3. If we scatter random points (x, y) uniformly:
       Ratio of (Points inside Circle) / (Total Points) should approx Pi/4.
    4. Therefore: Pi ≈ 4 * (Inside / Total)
    """
    
    if not os.path.exists(filepath):
        print(f"[-] Error: File '{filepath}' not found.")
        return

    file_size = os.path.getsize(filepath)
    print(f"[*] Analyzing file: {filepath}")
    print(f"[*] File Size: {file_size / 1024:.2f} KB")

    # We need 8 bytes per point (4 bytes for X coordinate, 4 bytes for Y coordinate)
    # Each coordinate is a 32-bit unsigned integer (0 to 4,294,967,295)
    points_count = file_size // 8
    
    if points_count < 1000:
        print("[-] Error: File too small. We need at least 8KB of data for a decent test.")
        return

    print(f"[*] Simulating {points_count:,} random points...")

    inside_circle = 0
    
    with open(filepath, "rb") as f:
        # Read the file in chunks of 8 bytes
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                break
                
            # Unpack two 32-bit integers ('I' = unsigned int, 'I' = unsigned int)
            x_raw, y_raw = struct.unpack("II", chunk)
            
            # Normalize them to a range of 0.0 to 1.0
            # 2^32 - 1 = 4294967295
            x = x_raw / 4294967295.0
            y = y_raw / 4294967295.0
            
            # Check if point is inside the unit circle (x² + y² <= 1)
            if (x*x + y*y) <= 1.0:
                inside_circle += 1

    # Calculate Pi
    pi_estimate = 4.0 * (inside_circle / points_count)
    actual_pi = 3.14159265359
    error_percent = abs((pi_estimate - actual_pi) / actual_pi) * 100

    print("\n=== RESULTS ===")
    print(f"Points Inside: {inside_circle:,}")
    print(f"Total Points:  {points_count:,}")
    print(f"Estimated Pi:  {pi_estimate:.5f}")
    print(f"Actual Pi:     {actual_pi:.5f}")
    print(f"Error:         {error_percent:.4f}%")
    
    if error_percent < 1.0:
        print("\n[+] PASS: Excellent randomness (Error < 1%)")
    elif error_percent < 5.0:
        print("\n[~] OKAY: Acceptable for small samples (Error < 5%)")
    else:
        print("\n[-] FAIL: High bias detected or sample size too small.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Pi Validator for TRNG")
    parser.add_argument("file", help="Path to the binary file to test")
    args = parser.parse_args()
    
    monte_carlo_pi(args.file)