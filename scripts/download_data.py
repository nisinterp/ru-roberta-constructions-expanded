"""Download the immutable source snapshots used in this experiment.

No credentials are required. Large source texts and model weights remain local
and ignored by Git. Existing files are retained; verify them with manifest.json.
"""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import requests

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "80234c447326551cc891c7c11174159af17c5158"
CONSTRUCTICON = "7806b7d74c56192b97150b06f755be4d37ad5b3c"
HF = "92d19006cb041c81b4bc21a34ebf3e4e082c5b4d"
MODEL = "5192d064ca6ac67c14c40e017ce41612e010f05f"
SESSION = requests.Session()


def download(url, target):
    target = ROOT / target
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with SESSION.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        temporary = target.with_suffix(target.suffix + ".partial")
        with temporary.open("wb") as out:
            for block in response.iter_content(1024 * 1024):
                out.write(block)
        temporary.replace(target)


def main():
    base = f"https://raw.githubusercontent.com/nisinterp/ru-constructions-revealed/{UPSTREAM}"
    for name in ["affinity", "analyze", "fetch_rnc", "figures", "parse_constructicon", "score"]:
        download(f"{base}/scripts/{name}.py", f"upstream/{name}.py")
    for source, target in [
        ("README.md", "upstream/README.md"),
        ("paper-typst/main.typ", "upstream/paper.typ"),
        ("results/summary.json", "upstream/original_summary.json"),
        ("data/rnc_items.jsonl", "data/raw/rnc_original.jsonl"),
        ("data/selected_constructions.json", "data/raw/original_selected.json"),
    ]:
        download(f"{base}/{source}", target)
    dest = ROOT / "data/raw/constructicon"
    if len(list(dest.glob("*.yml"))) != 4001:
        response = SESSION.get(
            f"https://codeload.github.com/constructicon/russian-data/tar.gz/{CONSTRUCTICON}", timeout=120
        )
        response.raise_for_status()
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if member.isfile() and len(parts) == 3 and parts[1] == "data" and parts[2].endswith(".yml"):
                    (dest / parts[2]).write_bytes(archive.extractfile(member).read())
    download(
        f"https://raw.githubusercontent.com/constructicon/russian-data/{CONSTRUCTICON}/LICENSE",
        "sources/constructicon-LICENSE.txt",
    )
    download(
        f"https://raw.githubusercontent.com/constructicon/russian-data/{CONSTRUCTICON}/README.md",
        "sources/constructicon-README.md",
    )
    for split in ["train", "validation", "test"]:
        download(
            f"https://huggingface.co/datasets/Futyn-Maker/russian-constructicon/resolve/{HF}/data/{split}-00000-of-00001.parquet",
            f"data/raw/hf/{split}.parquet",
        )
    download(
        f"https://huggingface.co/datasets/Futyn-Maker/russian-constructicon/raw/{HF}/README.md",
        "sources/hf-dataset-card.md",
    )
    for name in ["config.json", "merges.txt", "vocab.json", "pytorch_model.bin"]:
        download(f"https://huggingface.co/ai-forever/ruRoberta-large/resolve/{MODEL}/{name}", f"data/raw/model/{name}")
    download(f"https://huggingface.co/ai-forever/ruRoberta-large/raw/{MODEL}/README.md", "sources/model-card.md")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/model_revision.txt").write_text(MODEL + "\n")
    files = []
    for folder in ["upstream", "sources", "data/raw"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".partial"
                and path != ROOT / "sources/manifest.json"
            ):
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest))
    manifest = dict(
        upstream_revision=UPSTREAM,
        constructicon_revision=CONSTRUCTICON,
        hf_revision=HF,
        model_revision=MODEL,
        files=files,
    )
    (ROOT / "sources/manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Recorded {len(files)} source files")


if __name__ == "__main__":
    main()
