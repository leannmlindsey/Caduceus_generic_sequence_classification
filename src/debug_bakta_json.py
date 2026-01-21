"""
Debug script to inspect Bakta JSON structure.

Usage:
    python -m src.debug_bakta_json /path/to/bakta.json
"""

import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.debug_bakta_json /path/to/bakta.json")
        sys.exit(1)

    json_path = sys.argv[1]
    print(f"Loading: {json_path}\n")

    with open(json_path) as f:
        data = json.load(f)

    print("Top-level keys:", list(data.keys()))
    print()

    # Look at features
    features = data.get('features', [])
    print(f"Total features: {len(features)}")

    # Find CDS features
    cds_features = [f for f in features if f.get('type') == 'cds']
    print(f"CDS features: {len(cds_features)}")
    print()

    # Show first CDS structure
    if cds_features:
        print("First CDS feature keys:", list(cds_features[0].keys()))
        print()
        print("First CDS feature (full):")
        print(json.dumps(cds_features[0], indent=2))
        print()

    # Search for cog_category anywhere
    print("\n" + "="*60)
    print("Searching for 'cog_category' in JSON...")
    print("="*60)

    def find_cog(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'cog_category':
                    print(f"Found at path: {path}.{k} = {v}")
                    return True
                if find_cog(v, f"{path}.{k}"):
                    return True
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:5]):  # Only check first 5
                if find_cog(item, f"{path}[{i}]"):
                    return True
        return False

    find_cog(data)


if __name__ == "__main__":
    main()
