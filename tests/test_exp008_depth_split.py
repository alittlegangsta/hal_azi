from __future__ import annotations

import pickle
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_SCRIPT = REPO_ROOT / "scripts" / "thesis_make_exp008_depth_split.py"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "thesis_check_exp008_tfrecord_split.py"


def fake_tfrecord_record(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    zero_crc = b"\x00\x00\x00\x00"
    return length + zero_crc + payload + zero_crc


def write_fake_tfrecord(path: Path, count: int) -> None:
    with path.open("wb") as handle:
        for i in range(count):
            handle.write(fake_tfrecord_record(f"record-{i}".encode("ascii")))


class Exp008DepthSplitTests(unittest.TestCase):
    def test_builder_creates_manifest_and_split_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_tfrecord = root / "source.tfrecord"
            idx_pkl = root / "source.tfrecord.idx.pkl"
            waveforms_pkl = root / "processed_waveforms.pkl"
            output_dir = root / "split"

            write_fake_tfrecord(source_tfrecord, 20)
            with idx_pkl.open("wb") as handle:
                pickle.dump({"processed_indices": list(range(20))}, handle)
            with waveforms_pkl.open("wb") as handle:
                pickle.dump({"sonic_depths": [1000.0 + i for i in range(20)]}, handle)

            subprocess.run(
                [
                    sys.executable,
                    str(SPLIT_SCRIPT),
                    "--source-tfrecord",
                    str(source_tfrecord),
                    "--source-idx-pkl",
                    str(idx_pkl),
                    "--processed-waveforms-pkl",
                    str(waveforms_pkl),
                    "--output-dir",
                    str(output_dir),
                    "--gap-ft",
                    "0",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            self.assertTrue((output_dir / "train.tfrecord").exists())
            self.assertTrue((output_dir / "val.tfrecord").exists())
            self.assertTrue((output_dir / "test.tfrecord").exists())
            self.assertTrue((output_dir / "split_manifest.csv").exists())
            self.assertTrue((output_dir / "leakage_audit.json").exists())

            check = subprocess.run(
                [
                    sys.executable,
                    str(CHECK_SCRIPT),
                    "--split-dir",
                    str(output_dir),
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertIn('"status": "pass"', check.stdout)

    def test_dry_run_does_not_write_split_tfrecords(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_tfrecord = root / "source.tfrecord"
            idx_pkl = root / "source.tfrecord.idx.pkl"
            waveforms_pkl = root / "processed_waveforms.pkl"
            output_dir = root / "split"

            write_fake_tfrecord(source_tfrecord, 12)
            with idx_pkl.open("wb") as handle:
                pickle.dump({"processed_indices": list(range(12))}, handle)
            with waveforms_pkl.open("wb") as handle:
                pickle.dump({"sonic_depths": [2000.0 + i * 0.5 for i in range(12)]}, handle)

            subprocess.run(
                [
                    sys.executable,
                    str(SPLIT_SCRIPT),
                    "--source-tfrecord",
                    str(source_tfrecord),
                    "--source-idx-pkl",
                    str(idx_pkl),
                    "--processed-waveforms-pkl",
                    str(waveforms_pkl),
                    "--output-dir",
                    str(output_dir),
                    "--gap-ft",
                    "0",
                    "--dry-run",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            self.assertFalse((output_dir / "train.tfrecord").exists())
            self.assertTrue((output_dir / "split_manifest.csv").exists())
            self.assertTrue((output_dir / "leakage_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
