import sys
from pathlib import Path

# Ensure src/ is on the path during tests without requiring an editable install at every step.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
