"""Constrain private evaluation outputs to ignored project directories."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIVATE_ROOTS = (ROOT / "artifacts", ROOT / "local_data")


def private_output(path: Path, roots=PRIVATE_ROOTS) -> Path:
    target = Path(path).expanduser().resolve()
    if not any(target != root.resolve() and target.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Private evaluation outputs must be under artifacts/ or local_data/")
    return target


def check_immutable(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("Output already exists with different content; choose a new path")


def write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except FileExistsError:
        check_immutable(path, content)
