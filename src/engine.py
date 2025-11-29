"""
Entropy Engine Module for True Random Number Generation (TRNG)

This module harvests entropy from physical noise sources:
1. Webcam - Quantum Shot Noise from CMOS/CCD image sensors
2. Microphone - Thermal (Johnson-Nyquist) Noise from audio circuits

ECE Physics Background:
-----------------------
QUANTUM SHOT NOISE (Webcam):
- Shot noise arises from the discrete nature of electric charge (electrons/photons)
- In image sensors, photons arrive as discrete packets following Poisson statistics
- The shot noise variance equals the mean signal: σ² = μ (in photon counts)
- This is a fundamental quantum mechanical effect that cannot be eliminated
- SNR due to shot noise: SNR = √N where N is the number of photons
- At low light levels, shot noise dominates and provides excellent entropy

THERMAL (JOHNSON-NYQUIST) NOISE (Microphone):
- Thermal noise is caused by random thermal motion of charge carriers (electrons)
- First characterized by Johnson (1928) and explained by Nyquist using thermodynamics
- Power spectral density: S_v = 4kTR (V²/Hz) where:
  * k = Boltzmann constant (1.38 × 10⁻²³ J/K)
  * T = Temperature in Kelvin
  * R = Resistance in Ohms
- RMS voltage: V_rms = √(4kTRΔf) over bandwidth Δf
- This noise is fundamental and present in all resistive elements
- Microphone preamp circuits exhibit thermal noise in the audio band

ENTROPY MIXING AND WHITENING:
- Raw sensor data may have biases or correlations
- SHA-256 acts as a cryptographic "whitening" function
- Mixing multiple independent sources increases entropy density
- Timestamp adds temporal uniqueness (not primary entropy source)
"""

import hashlib
import struct
import time
from typing import Optional

import numpy as np

# Conditional imports for hardware interfaces
# These may not be available in all environments (e.g., CI/CD, headless servers)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class EntropyEngine:
    """
    True Random Number Generator using physical noise sources.
    
    Harvests entropy from:
    1. Webcam sensor (quantum shot noise from photon arrival statistics)
    2. Microphone (thermal Johnson-Nyquist noise from electronic circuits)
    
    The entropy is mixed with high-resolution timestamps and whitened
    using SHA-256 to produce cryptographically suitable random bytes.
    
    ECE Note: The entropy quality depends on the noise floor of the
    analog-to-digital converters in the sensors. Consumer-grade devices
    typically have 8-16 bits of resolution, with the LSBs dominated by
    thermal and quantization noise - ideal for entropy harvesting.
    """
    
    # Audio capture parameters
    # ECE Note: Higher sample rates capture more thermal noise bandwidth
    # Nyquist theorem: f_max = sample_rate / 2
    AUDIO_SAMPLE_RATE = 44100  # Hz - standard CD quality
    AUDIO_CHUNK_SIZE = 1024    # samples per buffer
    AUDIO_FORMAT_BITS = 16     # bits per sample (pyaudio.paInt16)
    
    # Video capture parameters  
    # ECE Note: Lower resolution is fine since we want noise, not image quality
    VIDEO_WIDTH = 320
    VIDEO_HEIGHT = 240
    
    def __init__(self):
        """
        Initialize the entropy engine.
        
        Sets up handles for webcam and microphone capture.
        Falls back gracefully if hardware is unavailable.
        """
        self._video_capture: Optional[object] = None
        self._audio_stream: Optional[object] = None
        self._pyaudio_instance: Optional[object] = None
        
        # Track which sources are available
        self._webcam_available = False
        self._mic_available = False
        
        self._init_webcam()
        self._init_microphone()
    
    def _init_webcam(self) -> None:
        """
        Initialize webcam capture for quantum shot noise harvesting.
        
        ECE Physics: The CMOS/CCD sensor converts photons to electrons via
        the photoelectric effect. Each pixel acts as a photon counter,
        and the statistical variation in photon arrival times creates
        shot noise that is truly random at the quantum level.
        """
        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available. Webcam entropy disabled.")
            return
            
        try:
            self._video_capture = cv2.VideoCapture(0)
            if self._video_capture.isOpened():
                # Set capture resolution
                self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.VIDEO_WIDTH)
                self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.VIDEO_HEIGHT)
                # Disable auto-exposure to maximize shot noise visibility
                # ECE Note: Auto-exposure tries to maintain constant brightness,
                # which can reduce the relative shot noise contribution
                self._video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self._webcam_available = True
                print("Webcam initialized for quantum shot noise harvesting.")
            else:
                print("Warning: Could not open webcam. Webcam entropy disabled.")
        except Exception as e:
            print(f"Warning: Webcam initialization failed: {e}")
    
    def _init_microphone(self) -> None:
        """
        Initialize microphone capture for thermal noise harvesting.
        
        ECE Physics: The microphone and its preamp circuit contain resistors
        that exhibit Johnson-Nyquist noise due to thermal electron motion.
        Even in a silent room, the ADC captures this thermal noise floor.
        The noise power is proportional to temperature and bandwidth.
        """
        if not PYAUDIO_AVAILABLE:
            print("Warning: PyAudio not available. Microphone entropy disabled.")
            return
            
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            self._audio_stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,  # 16-bit signed integers
                channels=1,               # Mono capture
                rate=self.AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.AUDIO_CHUNK_SIZE
            )
            self._mic_available = True
            print("Microphone initialized for thermal noise harvesting.")
        except Exception as e:
            print(f"Warning: Microphone initialization failed: {e}")
            self._cleanup_audio()
    
    def _cleanup_audio(self) -> None:
        """Clean up PyAudio resources."""
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        if self._pyaudio_instance is not None:
            try:
                self._pyaudio_instance.terminate()
            except Exception:
                pass
            self._pyaudio_instance = None
    
    def harvest_webcam_entropy(self) -> bytes:
        """
        Capture a frame from the webcam and extract raw pixel data.
        
        ECE Physics: Each pixel value contains:
        1. Signal component (scene illumination)
        2. Shot noise (√N photon statistics) - QUANTUM RANDOMNESS
        3. Read noise (amplifier thermal noise)
        4. Dark current noise (thermally generated electrons)
        
        The least significant bits are dominated by noise, providing
        excellent entropy. A 320x240 RGB frame provides ~230KB of raw data.
        
        Returns:
            Raw frame bytes, or empty bytes if capture fails.
        """
        if not self._webcam_available or self._video_capture is None:
            return b''
        
        try:
            ret, frame = self._video_capture.read()
            if ret and frame is not None:
                # Convert frame to bytes
                # ECE Note: The raw bytes include all color channels (BGR in OpenCV)
                # Each channel independently exhibits shot noise
                return frame.tobytes()
        except Exception as e:
            print(f"Warning: Webcam capture failed: {e}")
        
        return b''
    
    def harvest_microphone_entropy(self) -> bytes:
        """
        Capture an audio buffer from the microphone.
        
        ECE Physics: The audio signal contains:
        1. Acoustic signal (if any sound present)
        2. Thermal (Johnson-Nyquist) noise from resistors - THERMAL RANDOMNESS
        3. 1/f (flicker) noise from semiconductor devices
        4. Quantization noise from the ADC
        
        In a quiet environment, the LSBs are dominated by thermal noise.
        At 44.1kHz, 16-bit, 1024 samples provides 2KB of raw audio data.
        
        Returns:
            Raw audio bytes, or empty bytes if capture fails.
        """
        if not self._mic_available or self._audio_stream is None:
            return b''
        
        try:
            # Read audio chunk
            # ECE Note: exception_on_overflow=False prevents crashes if
            # the buffer overflows (can happen under high CPU load)
            audio_data = self._audio_stream.read(
                self.AUDIO_CHUNK_SIZE,
                exception_on_overflow=False
            )
            return audio_data
        except Exception as e:
            print(f"Warning: Microphone capture failed: {e}")
        
        return b''
    
    def get_timestamp_entropy(self) -> bytes:
        """
        Generate entropy from high-resolution timestamp.
        
        ECE Note: While timestamps are deterministic, the high-resolution
        component (nanoseconds) depends on exact execution timing, which
        varies due to OS scheduling, CPU cache states, and other factors.
        This adds temporal uniqueness but should not be a primary entropy source.
        
        Returns:
            8 bytes representing current timestamp in nanoseconds.
        """
        # time.time_ns() provides nanosecond resolution on modern systems
        timestamp_ns = time.time_ns()
        # Pack as unsigned 64-bit integer (little-endian)
        return struct.pack('<Q', timestamp_ns)
    
    def mix_entropy(self, *sources: bytes) -> bytes:
        """
        Combine multiple entropy sources by concatenation.
        
        ECE Physics: When combining independent noise sources:
        - If sources are truly independent, total entropy is additive
        - Concatenation preserves all entropy from each source
        - Even if one source is compromised, others maintain security
        
        Args:
            *sources: Variable number of byte strings to combine.
            
        Returns:
            Concatenated entropy bytes.
        """
        return b''.join(sources)
    
    def whiten(self, raw_entropy: bytes) -> bytes:
        """
        Apply SHA-256 to whiten/condition the raw entropy.
        
        ECE/Crypto Background:
        - Raw sensor data may have statistical biases (e.g., more 0s than 1s)
        - Raw data may have correlations between adjacent samples
        - SHA-256 is a cryptographic hash that:
          1. Produces uniform output distribution (each bit has 50% probability)
          2. Exhibits avalanche effect (small input change -> ~50% output change)
          3. Is computationally one-way (cannot recover input from output)
        
        This "whitening" process extracts and concentrates entropy.
        Note: You can only extract at most as much entropy as you input.
        
        Args:
            raw_entropy: Raw bytes from entropy sources.
            
        Returns:
            32 bytes (256 bits) of whitened random data.
        """
        # SHA-256 produces exactly 32 bytes of output
        return hashlib.sha256(raw_entropy).digest()
    
    def generate_random_bytes(self, num_bytes: int) -> bytes:
        """
        Generate the specified number of cryptographically random bytes.
        
        Process:
        1. Harvest entropy from webcam (quantum shot noise)
        2. Harvest entropy from microphone (thermal noise)
        3. Add timestamp for temporal uniqueness
        4. Mix all sources together
        5. Whiten using SHA-256
        6. Repeat until we have enough bytes
        
        ECE Note: Each iteration collects fresh physical noise samples,
        ensuring that the random output is not simply stretched from
        a single seed value.
        
        Args:
            num_bytes: Number of random bytes to generate.
            
        Returns:
            Bytes object of specified length containing random data.
        """
        if not self._webcam_available and not self._mic_available:
            raise RuntimeError(
                "No entropy sources available! "
                "Need at least webcam or microphone access."
            )
        
        result = bytearray()
        
        # SHA-256 produces 32 bytes per iteration
        bytes_per_iteration = 32
        
        while len(result) < num_bytes:
            # Harvest entropy from all available sources
            webcam_entropy = self.harvest_webcam_entropy()
            mic_entropy = self.harvest_microphone_entropy()
            timestamp_entropy = self.get_timestamp_entropy()
            
            # Mix all entropy sources
            mixed = self.mix_entropy(
                webcam_entropy,
                mic_entropy,
                timestamp_entropy
            )
            
            # Whiten the mixed entropy
            whitened = self.whiten(mixed)
            
            # Append to result
            result.extend(whitened)
        
        # Truncate to exact requested length
        return bytes(result[:num_bytes])
    
    def close(self) -> None:
        """
        Release all hardware resources.
        
        Should be called when done generating random numbers to free
        webcam and microphone devices for other applications.
        """
        # Release webcam
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                pass
            self._video_capture = None
            self._webcam_available = False
        
        # Release microphone
        self._cleanup_audio()
        self._mic_available = False
        
        print("Entropy engine resources released.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


# Fallback entropy source for environments without hardware access
class FallbackEntropyEngine(EntropyEngine):
    """
    Fallback entropy engine using OS-provided randomness.
    
    This is used when webcam and microphone are not available.
    Uses os.urandom() which typically sources from:
    - Linux: /dev/urandom (kernel entropy pool)
    - Windows: CryptGenRandom (CSP)
    - macOS: /dev/random (Yarrow PRNG)
    
    ECE Note: Modern OS entropy pools collect from multiple sources:
    - Interrupt timing variations
    - Disk I/O timing jitter
    - Network packet arrival times
    - Hardware RNG (RDRAND on modern Intel/AMD CPUs)
    """
    
    def __init__(self):
        """Initialize without hardware access."""
        self._webcam_available = False
        self._mic_available = False
        self._video_capture = None
        self._audio_stream = None
        self._pyaudio_instance = None
        print("Using fallback entropy (OS random source).")
    
    def generate_random_bytes(self, num_bytes: int) -> bytes:
        """
        Generate random bytes using OS entropy source.
        
        Args:
            num_bytes: Number of random bytes to generate.
            
        Returns:
            Cryptographically random bytes from OS.
        """
        import os
        return os.urandom(num_bytes)
    
    def close(self) -> None:
        """No resources to release in fallback mode."""
        pass


def create_entropy_engine() -> EntropyEngine:
    """
    Factory function to create the appropriate entropy engine.
    
    Returns hardware-based engine if available, otherwise fallback.
    
    Returns:
        EntropyEngine instance (hardware-based or fallback).
    """
    engine = EntropyEngine()
    
    # If no hardware sources available, use fallback
    if not engine._webcam_available and not engine._mic_available:
        engine.close()
        return FallbackEntropyEngine()
    
    return engine
