import json
from argus.core.json_repair import extract_json_object

test_str = """
{
  "num_support": 0,
  "num_against": 5,
  "num_partial": 0,
  "total_scanned": 5,
  "summaries": ["5 studies challenge the claim of US invasion of Cuba by 2050",],
  "direction_confidence": 0.9
}
"""

try:
    obj = extract_json_object(test_str)
    print("SUCCESS")
    print(json.dumps(obj, indent=2))
except Exception as e:
    print(f"FAILED: {e}")
