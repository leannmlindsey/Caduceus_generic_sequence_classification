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

    # Find CDS with COG
    features = data.get('features', [])

    cog_count = 0
    cds_with_cog = None

    for feat in features:
        if feat.get('type') == 'cds' and feat.get('cog_category'):
            cog_count += 1
            if cds_with_cog is None:
                cds_with_cog = feat

    print(f"Total CDS: {len([f for f in features if f.get('type') == 'cds'])}")
    print(f"CDS with cog_category: {cog_count}")

    if cds_with_cog:
        print("\nExample CDS WITH cog_category:")
        print(f"  locus: {cds_with_cog.get('locus')}")
        print(f"  product: {cds_with_cog.get('product')}")
        print(f"  cog_id: {cds_with_cog.get('cog_id')}")
        print(f"  cog_category: {cds_with_cog.get('cog_category')}")
        print(f"  keys: {list(cds_with_cog.keys())}")
    else:
        print("\nNo CDS with cog_category found directly on feature.")
        print("Searching entire JSON for cog_category...")

        # Search raw JSON
        raw = json.dumps(data)
        if 'cog_category' in raw:
            print("  cog_category EXISTS in JSON but not on CDS features directly")
        else:
            print("  cog_category NOT FOUND in JSON")


if __name__ == "__main__":
    main()
