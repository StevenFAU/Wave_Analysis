"""Fetch the original VWBT clips and author-supplied leave-one-site-out inventory."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/28814993/versions/5"
FILES = ("loso_split_by_database_filename.csv", "Set_A.zip")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", action="store_true", help="also fetch the 5.07 GB source zip")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=30) as response:
        meta = json.load(response)
    (args.out / "figshare-v5.json").write_text(json.dumps(meta, indent=2))
    for name in FILES if args.clips else FILES[:1]:
        rec = next(f for f in meta["files"] if f["name"] == name)
        dest = args.out / name
        tmp = dest.with_suffix(dest.suffix + ".part")
        if not dest.exists() and (not tmp.exists() or tmp.stat().st_size != rec["size"]):
            subprocess.run(
                [
                    "curl",
                    "-fLsS",
                    "--connect-timeout",
                    "30",
                    "--retry",
                    "3",
                    "-C",
                    "-",
                    "-o",
                    str(tmp),
                    rec["download_url"],
                ],
                check=True,
            )
        path = dest if dest.exists() else tmp
        with path.open("rb") as fh:
            digest = hashlib.file_digest(fh, "md5").hexdigest()
        if digest != rec["computed_md5"] or path.stat().st_size != rec["size"]:
            raise ValueError(f"failed source checksum: {path}")
        if path == tmp:
            tmp.rename(dest)
        print(dest, "verified", flush=True)


if __name__ == "__main__":
    main()
