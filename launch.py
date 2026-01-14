import subprocess
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    ps_script = repo / "scripts" / "launch_agent.ps1"
    subprocess.Popen(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps_script),
        ],
        cwd=str(repo),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


if __name__ == "__main__":
    main()
