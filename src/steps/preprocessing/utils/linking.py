import os
import shutil
from pathlib import Path

def hardlink_or_copy(src: Path, dst: Path, mode: str = "hardlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)