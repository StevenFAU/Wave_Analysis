"""Create a self-contained buoy measurement package and verify the source image hashes."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from wave_analysis.config import data_dir


def export(source, dest):
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        stage = Path(temp) / "package"
        stage.mkdir()
        with (stage / "SHA256SUMS").open("w") as manifest:

            def include(path, relative, expected=None):
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                with target.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if expected is not None and expected != digest:
                    raise ValueError(f"image checksum mismatch for {relative}")
                manifest.write(f"{digest}  {relative}\n")

            for name in ["labels.csv", "bandwidth_checks.csv", "provenance.json"]:
                include(source / name, name)
            horizons = source / "horizons.parquet"
            if not horizons.is_file():
                horizons = (
                    data_dir("processed") / "horizon/ndbc_buoycam_sea_state_v0/horizons.parquet"
                )
            include(horizons, "horizons.parquet")
            with (stage / "labels.csv").open() as f:
                for row in csv.DictReader(f):
                    relative = Path(row["image"])
                    if relative.parts[:1] != ("images",) or ".." in relative.parts:
                        raise ValueError(f"invalid package path {relative}")
                    include(source / relative, relative, row["sha256_package"])
        stage.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    export(args.package, args.out)
