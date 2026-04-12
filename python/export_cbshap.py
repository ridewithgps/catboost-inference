"""Export a CBSHAP v1 sidecar from a trained CatBoost .cbm file.

Usage:
    uv run python export_cbshap.py <model.cbm> [output.cbshap]

If output is omitted, writes <model>.shap.cbshap next to the input.
Invoked by ``just cbshap``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cbshap import export_from_cbm


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    if not 1 <= len(args) <= 2:
        print("usage: export_cbshap.py <model.cbm> [output.cbshap]", file=sys.stderr)
        return 1

    model_path = Path(args[0])
    output_path = Path(args[1]) if len(args) == 2 else None

    result = export_from_cbm(model_path, output_path=output_path)
    size_kb = result.stat().st_size / 1024
    print(f"wrote {result} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
