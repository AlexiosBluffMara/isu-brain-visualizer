"""CLI wrapper for the anomaly pipeline — delegates to the src module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fcd_prototype.anomaly_pipeline import main

if __name__ == "__main__":
    main()
