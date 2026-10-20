"""Start the PRAA-FL web app:  python run_app.py  ->  http://127.0.0.1:8000

Run from the project root (the local ``datasets/`` package must shadow HuggingFace ``datasets``).
Options: --host, --port, --reload;  env: PRAA_CHECKPOINT, PRAA_PROTOTYPES, PRAA_DEVICE.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import uvicorn
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload,
                reload_dirs=[str(ROOT / "app")] if args.reload else None)


if __name__ == "__main__":
    main()
