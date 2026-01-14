import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class AdbConfig:
    adb_path: str = "adb"
    serial: Optional[str] = None


class AdbDevice:
    def __init__(self, cfg: Optional[AdbConfig] = None):
        self.cfg = cfg or AdbConfig()

    def _base_cmd(self) -> Sequence[str]:
        if self.cfg.serial:
            return [self.cfg.adb_path, "-s", self.cfg.serial]
        return [self.cfg.adb_path]

    def _run(self, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
        cmd = list(self._base_cmd()) + list(args)
        return subprocess.run(cmd, check=check, capture_output=True)

    def screenshot(self, out_path: str) -> None:
        cmd = list(self._base_cmd()) + ["exec-out", "screencap", "-p"]
        p = subprocess.run(cmd, check=True, capture_output=True)
        data = p.stdout
        with open(out_path, "wb") as f:
            f.write(data)

    def tap(self, x: int, y: int) -> None:
        self._run(["shell", "input", "tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._run(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])

    def back(self) -> None:
        self._run(["shell", "input", "keyevent", "4"]) 
