import sys
from pathlib import Path

# Add src directory to path so deusauditron can be imported as a top-level module
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))