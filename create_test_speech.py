import wave
import struct
import math
import random

sample_rate = 16000
duration = 2.0

# Speech-like signal with formants
samples = []
for i in range(int(sample_rate * duration)):
    t = i / sample_rate
    
    # Fundamental frequency ~100Hz 
    f0 = 100 + 30 * math.sin(2 * math.pi * 2 * t)
    
    # Formants for vowel-like sounds
    f1 = 500 + 100 * math.sin(2 * math.pi * 0.5 * t)
    f2 = 1500 + 200 * math.sin(2 * math.pi * 0.3 * t)
    
    # Combine with harmonic structure
    sample = (
        0.6 * math.sin(2 * math.pi * f0 * t) +
        0.3 * math.sin(2 * math.pi * f1 * t) +
        0.15 * math.sin(2 * math.pi * f2 * t) +
        0.1 * math.sin(2 * math.pi * 3 * f0 * t)  # Third harmonic
    )
    
    # Envelope
    envelope = 0.3 + 0.7 * abs(math.sin(math.pi * t / duration))
    sample *= envelope
    
    # Noise
    sample += 0.02 * (2 * random.random() - 1)
    
    samples.append(int(32767 * sample))

with wave.open('test_speech.wav', 'w') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)  
    wav.setframerate(sample_rate)
    wav.writeframes(struct.pack('<' + 'h' * len(samples), *samples))

print('Created test_speech.wav')
