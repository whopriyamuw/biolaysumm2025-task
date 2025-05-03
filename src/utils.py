import os.path
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports")
