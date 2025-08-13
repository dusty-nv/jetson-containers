#!/usr/bin/env python3
"""
Test script for sound-utils package.
Tests the main audio libraries: soundfile, sounddevice, and portaudio functionality.
"""
import os
import sys
import numpy as np
import tempfile
import argparse


def test_imports():
    """Test that all required audio libraries can be imported."""
    print("Testing imports...")
    
    try:
        import soundfile as sf
        print(f"✓ soundfile imported successfully (version: {sf.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import soundfile: {e}")
        return False
    
    try:
        import sounddevice as sd
        print(f"✓ sounddevice imported successfully (version: {sd.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import sounddevice: {e}")
        return False
    
    try:
        # Test torch and torchaudio dependencies
        import torch
        import torchaudio
        print(f"✓ torch imported successfully (version: {torch.__version__})")
        print(f"✓ torchaudio imported successfully (version: {torchaudio.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import torch/torchaudio: {e}")
        return False
    
    return True


def test_audio_devices():
    """Test audio device detection."""
    print("\nTesting audio devices...")
    
    try:
        import sounddevice as sd
        
        # Query devices
        devices = sd.query_devices()
        print(f"✓ Found {len(devices)} audio devices")
        
        # Get default devices
        default_input = sd.default.device[0] if sd.default.device[0] is not None else "None"
        if (
            sd.default.device is not None
            and isinstance(sd.default.device, (tuple, list))
            and len(sd.default.device) >= 2
        ):
            default_input = sd.default.device[0] if sd.default.device[0] is not None else "None"
            default_output = sd.default.device[1] if sd.default.device[1] is not None else "None"
        else:
            default_input = "None"
            default_output = "None"
        
        print(f"  - Default input device: {default_input}")
        print(f"  - Default output device: {default_output}")
        
        return True
    except Exception as e:
        print(f"✗ Audio device test failed: {e}")
        return False


def test_audio_file_operations(test_audio_path=None):
    """Test reading and writing audio files."""
    print("\nTesting audio file operations...")
    
    try:
        import soundfile as sf
        import numpy as np
        
        # Create test audio data
        sample_rate = 44100
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Test writing audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            sf.write(tmp_path, audio_data, sample_rate)
            print(f"✓ Successfully wrote test audio to {tmp_path}")
            
            # Test reading audio file
            read_data, read_sr = sf.read(tmp_path)
            print(f"✓ Successfully read audio file:")
            print(f"  - Sample rate: {read_sr} Hz")
            print(f"  - Duration: {len(read_data) / read_sr:.2f} seconds")
            print(f"  - Shape: {read_data.shape}")
            print(f"  - Data type: {read_data.dtype}")
            
            # Test reading existing audio file if provided
            if test_audio_path and os.path.exists(test_audio_path):
                try:
                    existing_data, existing_sr = sf.read(test_audio_path)
                    print(f"✓ Successfully read existing audio file {test_audio_path}:")
                    print(f"  - Sample rate: {existing_sr} Hz")
                    print(f"  - Duration: {len(existing_data) / existing_sr:.2f} seconds")
                    print(f"  - Shape: {existing_data.shape}")
                except Exception as e:
                    print(f"⚠ Could not read existing audio file {test_audio_path}: {e}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Audio file operations test failed: {e}")
        return False


def test_torchaudio_integration():
    """Test torchaudio functionality."""
    print("\nTesting torchaudio integration...")
    
    try:
        import torch
        import torchaudio
        import numpy as np
        
        # Create test audio tensor
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio_tensor = 0.5 * torch.sin(2 * np.pi * frequency * t).unsqueeze(0)  # Add channel dimension
        
        print(f"✓ Created test audio tensor:")
        print(f"  - Shape: {audio_tensor.shape}")
        print(f"  - Sample rate: {sample_rate} Hz")
        
        # Test torchaudio transforms
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        mel_spec = transform(audio_tensor)
        print(f"✓ Applied MelSpectrogram transform, output shape: {mel_spec.shape}")
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            torchaudio.save(tmp_path, audio_tensor, sample_rate)
            print(f"✓ Saved audio tensor to {tmp_path}")
            
            loaded_tensor, loaded_sr = torchaudio.load(tmp_path)
            print(f"✓ Loaded audio tensor from file:")
            print(f"  - Shape: {loaded_tensor.shape}")
            print(f"  - Sample rate: {loaded_sr} Hz")
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Torchaudio integration test failed: {e}")
        return False


def test_audio_playback(test_duration=0.5):
    """Test audio playback capability (non-blocking)."""
    print(f"\nTesting audio playback (non-blocking, {test_duration}s)...")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        # Create a simple test tone
        sample_rate = 44100
        frequency = 440  # A4 note
        t = np.linspace(0, test_duration, int(sample_rate * test_duration), False)
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)  # Lower volume
        
        # Check if we have output devices
        devices = sd.query_devices()
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        if not output_devices:
            print("⚠ No output devices found, skipping playback test")
            return True
        
        # Attempt non-blocking playback
        print("✓ Starting non-blocking audio playback...")
        sd.play(audio_data, sample_rate, blocking=False)
        
        # Wait for playback to finish
        sd.wait()
        print("✓ Audio playback completed successfully")
        
        return True
        
    except Exception as e:
        print(f"⚠ Audio playback test failed (this is often expected in containers): {e}")
        # Don't fail the entire test suite for playback issues in containers
        return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test sound-utils package')
    parser.add_argument('--audio-file', type=str, help='Path to test audio file')
    parser.add_argument('--skip-playback', action='store_true', 
                        help='Skip audio playback test')
    parser.add_argument('--playback-duration', type=float, default=0.5,
                        help='Duration for playback test in seconds')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sound Utils Package Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Run tests
    test_functions = [
        test_imports,
        test_audio_devices,
        lambda: test_audio_file_operations(args.audio_file),
        test_torchaudio_integration,
    ]
    
    if not args.skip_playback:
        test_functions.append(lambda: test_audio_playback(args.playback_duration))
    
    for test_func in test_functions:
        total_tests += 1
        if test_func():
            tests_passed += 1
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Sound utils package is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
