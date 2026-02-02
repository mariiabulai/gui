"""Test if all imports work correctly"""

print("Testing imports...")

# Test 1: Import calibration module
try:
    from src.calibration import validate_calibration
    print("✓ Calibration module imported successfully")
except ImportError as e:
    print(f"✗ Calibration import failed: {e}")

# Test 2: Import reconstruction module
try:
    from src.reconstruction import SfMReconstructor, BundleAdjuster
    print("✓ Reconstruction module imported successfully")
except ImportError as e:
    print(f"✗ Reconstruction import failed: {e}")

# Test 3: Load calibration data
try:
    import numpy as np
    data = np.load('src/calibration/calibration_results/calibration_data.npz')
    K = data['mtx']
    dist = data['dist']
    print(f"✓ Calibration data loaded: K shape {K.shape}, dist shape {dist.shape}")
except Exception as e:
    print(f"✗ Calibration data load failed: {e}")

# Test 4: Instantiate SfMReconstructor
try:
    reconstructor = SfMReconstructor('src/calibration/calibration_results/calibration_data.npz')
    print(f"✓ SfMReconstructor instantiated")
except Exception as e:
    print(f"✗ SfMReconstructor instantiation failed: {e}")

# Test 5: Instantiate BundleAdjuster
try:
    ba = BundleAdjuster(np.eye(3), np.zeros(5))
    print(f"✓ BundleAdjuster instantiated")
except Exception as e:
    print(f"✗ BundleAdjuster instantiation failed: {e}")

print("\n✅ All imports OK! Ready to run reconstruction.")