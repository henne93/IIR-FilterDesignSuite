"""Entry point for the IIR Filter Design Suite desktop app (Phase 5).

Run with `python main.py`. For headless/automated smoke testing, set
`QT_QPA_PLATFORM=offscreen` in the environment before launching -- Qt reads
that variable itself, nothing here needs to branch on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT / "src" / "python", _ROOT / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from ui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
