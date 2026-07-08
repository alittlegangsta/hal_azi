from __future__ import annotations

import pickle
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional local dependency
    h5py = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_SCRIPT = REPO_ROOT / "scripts" / "thesis_make_exp007_depth_split.py"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "thesis_check_exp007_tfrecord_split.py"


def fake_tfrecord_record(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    zero_crc = b"\x00\x00\x00\x00"
    return length + zero_crc + payload + zero_crc


def write_fake_tfrecord(path: Path, count: int) -> None:
    with path.open("wb") as handle:
        for i in range(count):
            handle.write(fake_tfrecord_record(f"record-{i}".encode("ascii")))


def depth_key(depth: float) -> str:
    return str(float(depth)).replace(".", "_")


@unittest.skipIf(h5py is None, "h5py is not available")
class Exp007DepthSplitTests(unittest.TestCase):
    def test_builder_reconstructs_mapping_from_path_data_and_writes_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_tfrecord = root / "profile_regression_data.tfrecord"
            waveforms_pkl = root / "processed_waveforms.pkl"
            ground_truth_h5 = root / "ground_truth_db_array_03.h5"
            output_dir = root / "split"
            depths = [3000.0 + i for i in range(24)]

            write_fake_tfrecord(source_tfrecord, 20)
            with waveforms_pkl.open("wb") as handle:
                pickle.dump({"sonic_depths": depths, "waveforms": None}, handle)
            with h5py.File(ground_truth_h5, "w") as handle:
                path_data = handle.create_group("path_data")
                for depth in depths[:20]:
                    path_data.create_dataset(depth_key(depth), data=[[1.0]])

            subprocess.run(
                [
                    sys.executable,
                    str(SPLIT_SCRIPT),
                    "--source-tfrecord",
                    str(source_tfrecord),
                    "--processed-waveforms-pkl",
                    str(waveforms_pkl),
                    "--ground-truth-h5",
                    str(ground_truth_h5),
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

    def test_count_mismatch_fails_instead_of_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_tfrecord = root / "profile_regression_data.tfrecord"
            waveforms_pkl = root / "processed_waveforms.pkl"
            ground_truth_h5 = root / "ground_truth_db_array_03.h5"
            output_dir = root / "split"
            depths = [3100.0 + i for i in range(10)]

            write_fake_tfrecord(source_tfrecord, 8)
            with waveforms_pkl.open("wb") as handle:
                pickle.dump({"sonic_depths": depths, "waveforms": None}, handle)
            with h5py.File(ground_truth_h5, "w") as handle:
                path_data = handle.create_group("path_data")
                for depth in depths[:7]:
                    path_data.create_dataset(depth_key(depth), data=[[1.0]])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SPLIT_SCRIPT),
                    "--source-tfrecord",
                    str(source_tfrecord),
                    "--processed-waveforms-pkl",
                    str(waveforms_pkl),
                    "--ground-truth-h5",
                    str(ground_truth_h5),
                    "--output-dir",
                    str(output_dir),
                    "--gap-ft",
                    "0",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("do not match", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
