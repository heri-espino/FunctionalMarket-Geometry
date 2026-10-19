"""Validate correspondence of numbered experiment source, notes and archived runs."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "market_geometry"
PATTERN = re.compile(r"^\d{3}[-_][a-z0-9_]+$")


def validate() -> None:
    """Validate experiment directory coherence, archive hashes and Markdown images."""
    families = ["code", "notes", "figures", "results"]
    groups = {}
    for name in families:
        groups[name] = {
            path.name for path in (AREA / name).iterdir()
            if path.is_dir() and PATTERN.match(path.name)
        }
    assert groups["code"] == groups["notes"] == groups["figures"] == groups["results"], groups
    assert groups["notes"], "No numbered experiments present"
    for identifier in groups["notes"]:
        note = AREA / "notes" / identifier / "README.md"
        content = note.read_text(encoding="utf-8")
        assert "$$" in content and "Figure plan" in content, identifier
        for reference in re.findall(r"!\[[^]]*\]\(([^)]+)\)", content):
            file_path = (note.parent / reference).resolve()
            assert file_path.is_file(), f"Broken note image {file_path}"
        result_runs = {p.name for p in (AREA / "results" / identifier / "runs").iterdir() if p.is_dir()}
        figure_runs = {p.name for p in (AREA / "figures" / identifier / "runs").iterdir() if p.is_dir()}
        assert result_runs == figure_runs, f"Mismatched runs for {identifier}"
        for rid in result_runs:
            result_dir = AREA / "results" / identifier / "runs" / rid
            manifest = json.loads((result_dir / "sha256_manifest.json").read_text(encoding="utf-8"))
            for rel_path, expected_hash in manifest.items():
                actual = ROOT / rel_path
                assert actual.is_file(), rel_path
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected_hash, rel_path
            assert rid in content, f"Archive run not mentioned in note: {rid}"
    print(f"Validated {len(groups['notes'])} numbered experiments and their existing immutable archives.")


if __name__ == "__main__":
    validate()
