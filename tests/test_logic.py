import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import compute_headway_proxy, compute_progress_along_shape, build_shape_cache

def test_circular_logic():
    print("Testing Circular Logic...")
    # Mock data: 3 vehicles on a shape of length 1000m
    # Vehicles at 100m, 500m, 900m
    # Gaps: 400m, 400m, 200m (wrap around constraint)
    # If route_len is correctly used as 1000m:
    # Gap 100->500: 400
    # Gap 500->900: 400
    # Gap 900->100: (1000 - 900) + 100 = 200
    
    df = pd.DataFrame({
        "route_id": ["R1", "R1", "R1"],
        "progress_m": [100.0, 500.0, 900.0],
        "shape_len": [1000.0, 1000.0, 1000.0]  # This column is crucial for the fix
    })
    
    # Run with circular=True
    res = compute_headway_proxy(df, circular=True)
    
    print("Result:")
    print(res[["progress_m", "headway_m"]])
    
    # Expectations
    # 100m: min(gap_behind, gap_ahead) = min(200, 400) = 200
    # 500m: min(400, 400) = 400
    # 900m: min(400, 200) = 200
    
    h100 = res.loc[res["progress_m"] == 100.0, "headway_m"].iloc[0]
    h500 = res.loc[res["progress_m"] == 500.0, "headway_m"].iloc[0]
    h900 = res.loc[res["progress_m"] == 900.0, "headway_m"].iloc[0]
    
    assert h100 == 200.0, f"Expected 200.0 at 100m, got {h100}"
    assert h500 == 400.0, f"Expected 400.0 at 500m, got {h500}"
    assert h900 == 200.0, f"Expected 200.0 at 900m, got {h900}"
    print("✅ Circular logic test passed!")

def test_fallback_logic_simulation():
    print("\nTesting Fallback Logic Simulation...")
    # Simulating the app.py change where we drop direction_id
    
    df = pd.DataFrame({
        "route_id": ["R1"] * 4,
        "direction_id": [0, 0, 1, 1],
        "progress_m": [100, 200, 150, 250],
        "shape_len": [1000] * 4
    })
    
    # Case: Suppose we ignore direction. Sorted progress: 100, 150, 200, 250
    # Gaps: 50, 50, 50. Circular wrap: (1000-250)+100 = 850
    # Headways: 
    # 100: min(850, 50) = 50
    # 150: min(50, 50) = 50
    # 200: min(50, 50) = 50
    # 250: min(50, 850) = 50
    
    # We just want to ensure compute_headway_proxy works without direction_id
    df_no_dir = df.drop(columns=["direction_id"])
    res = compute_headway_proxy(df_no_dir, circular=True)
    
    print("Fallback Result (no direction):")
    print(res[["progress_m", "headway_m"]])
    
    assert not res["headway_m"].isna().any(), "Should have computed headways for all"
    assert (res["headway_m"] == 50.0).all(), "All headways should be 50.0 in this synthetic example"
    print("✅ Fallback simulation passed!")

if __name__ == "__main__":
    try:
        test_circular_logic()
        test_fallback_logic_simulation()
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
