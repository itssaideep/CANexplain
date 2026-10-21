"""Robust chunked downloader for automotive CAN datasets with retries."""

import os
import time
import urllib.request

import click
from tqdm import tqdm

BASE_URL = "https://bitbucket.org/brooke-lampe/can-train-and-test-v1.5/raw/master"

TARGET_FILES = {
    "train_benign_accessory": (
        f"{BASE_URL}/set_01/train_01_attack_free/accessory-1.csv",
        "data/real/set_01/train_01_attack_free/accessory-1.csv",
    ),
    "test_dos_flood": (
        f"{BASE_URL}/set_01/test_01_known_vehicle_known_attack/DoS-3.csv",
        "data/real/set_01/test_01_known_vehicle_known_attack/DoS-3.csv",
    ),
    "test_standstill": (
        f"{BASE_URL}/set_01/test_01_known_vehicle_known_attack/standstill-3.csv",
        "data/real/set_01/test_01_known_vehicle_known_attack/standstill-3.csv",
    ),
}


def download_file_robust(url: str, output_path: str, max_retries: int = 5) -> None:
    """Downloads a file with stream chunks, progress bar, and retries."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TRY {attempt}/{max_retries}] Connecting to {url}...")
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.info().get("Content-Length", -1))
                chunk_size = 64 * 1024  # 64 KB

                with (
                    open(output_path, "wb") as out_file,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=os.path.basename(output_path),
                    ) as pbar,
                ):
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))

            if total_size != -1 and os.path.getsize(output_path) < total_size:
                raise OSError(
                    f"Incomplete download: {os.path.getsize(output_path)} of {total_size} bytes."
                )

            print(f"[OK] Downloaded {output_path} ({os.path.getsize(output_path)} bytes)")
            return

        except Exception as e:
            print(f"[WARNING] Attempt {attempt} failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise


@click.command()
@click.option(
    "--all-targets", is_flag=True, default=False, help="Download all candidate attack traces."
)
def main(all_targets: bool) -> None:
    """Download real CAN traffic datasets for CANExplain-QI."""
    print("=" * 70)
    print("CANExplain-QI Real Dataset Downloader (can-train-and-test-v1.5)")
    print("=" * 70)

    download_file_robust(*TARGET_FILES["train_benign_accessory"])
    download_file_robust(*TARGET_FILES["test_dos_flood"])

    if all_targets:
        download_file_robust(*TARGET_FILES["test_standstill"])

    print("\nDataset acquisition completed successfully.")


if __name__ == "__main__":
    main()
