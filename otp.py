import sys
import os
import argparse
from pathlib import Path

# --- ECE SETUP: Connect to your Entropy Engine ---
# We need to add the 'src' folder to the path so we can import the engine
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from engine import create_entropy_engine
except ImportError:
    print("ERROR: Could not find 'src/engine.py'. Make sure your folder structure is correct!")
    sys.exit(1)

def xor_data(data: bytes, key: bytes) -> bytes:
    """
    Performs the XOR (Exclusive OR) operation byte-by-byte.
    ECE Logic: A ^ B = C  -->  C ^ B = A
    """
    return bytes(a ^ b for a, b in zip(data, key))

def encrypt_file(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"[-] Error: File '{filepath}' not found.")
        return

    print(f"[*] Reading plaintext: {path.name}...")
    with open(path, 'rb') as f:
        plaintext = f.read()

    file_size = len(plaintext)
    print(f"[*] File size: {file_size} bytes")
    print("[*] Harvesting Quantum-Thermal Entropy for Key (this may take a moment)...")

    # 1. Generate a One-Time Pad exactly the size of the file
    with create_entropy_engine() as trng:
        key = trng.generate_random_bytes(file_size)

    # 2. Encrypt (XOR)
    ciphertext = xor_data(plaintext, key)

    # 3. Save Artifacts
    # Save the Encrypted file
    enc_path = path.with_suffix(path.suffix + '.enc')
    with open(enc_path, 'wb') as f:
        f.write(ciphertext)
    
    # Save the Key (The "Pad")
    key_path = path.with_suffix(path.suffix + '.key')
    with open(key_path, 'wb') as f:
        f.write(key)

    print(f"\n[+] SUCCESS!")
    print(f"    Encrypted File: {enc_path} (Send this to anyone)")
    print(f"    Secret Key:     {key_path} (KEEP THIS SAFE & PRIVATE)")
    print("    NEVER reuse this key file for another message.")

def decrypt_file(enc_path, key_path):
    enc_file = Path(enc_path)
    key_file = Path(key_path)

    if not enc_file.exists() or not key_file.exists():
        print("[-] Error: Missing input files.")
        return

    print(f"[*] Reading ciphertext: {enc_file.name}")
    with open(enc_file, 'rb') as f:
        ciphertext = f.read()

    print(f"[*] Reading key: {key_file.name}")
    with open(key_file, 'rb') as f:
        key = f.read()

    if len(ciphertext) != len(key):
        print("[-] ERROR: Key length does not match file length!")
        print("    This is a One-Time Pad; the key must be exactly the size of the message.")
        return

    # Decrypt (XOR again)
    print("[*] Decrypting...")
    plaintext = xor_data(ciphertext, key)

    # Save restored file
    # Remove the .enc extension to guess original format
    original_name = enc_file.stem 
    # Create a name like 'secret.txt.dec' to avoid overwriting original
    out_path = enc_file.with_name("restored_" + original_name)
    
    with open(out_path, 'wb') as f:
        f.write(plaintext)

    print(f"\n[+] DECRYPTION COMPLETE.")
    print(f"    Restored File: {out_path}")

# --- CLI INTERFACE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum-Thermal One-Time Pad Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--encrypt', metavar='FILE', help='Encrypt a file')
    group.add_argument('-d', '--decrypt', nargs=2, metavar=('ENC_FILE', 'KEY_FILE'), help='Decrypt a file (requires .enc and .key)')

    args = parser.parse_args()

    if args.encrypt:
        encrypt_file(args.encrypt)
    elif args.decrypt:
        decrypt_file(args.decrypt[0], args.decrypt[1])