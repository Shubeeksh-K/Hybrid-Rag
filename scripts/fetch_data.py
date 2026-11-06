"""Download the QFT source (not redistributed in this repo).

Source: G. Date, "Lectures on Introduction to Quantum Field Theory",
arXiv:2207.02243. Check the license on the arXiv page before redistributing.
"""
import tarfile
import urllib.request
from pathlib import Path

ARXIV_ID = "2207.02243"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "date2207"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tar_path = OUT.parent / "date2207.tar"
    urllib.request.urlretrieve(f"https://arxiv.org/e-print/{ARXIV_ID}", tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(OUT, filter="data")
    print(f"Extracted to {OUT}")


if __name__ == "__main__":
    main()
