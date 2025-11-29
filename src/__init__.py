"""
TRNG (True Random Number Generator) Source Package

Provides entropy harvesting from physical noise sources:
- Quantum shot noise from webcam sensors
- Thermal (Johnson-Nyquist) noise from microphone circuits
"""

from .engine import EntropyEngine, FallbackEntropyEngine, create_entropy_engine

__all__ = ['EntropyEngine', 'FallbackEntropyEngine', 'create_entropy_engine']
