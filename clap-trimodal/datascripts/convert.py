import os
import ffmpeg
from pathlib import Path
from tqdm import tqdm


def convert_mp4_to_wav_ffmpeg(src_root, dst_root):
    failed_files = []

    for split in ["train", "val", "test"]:
        src_dir = Path(src_root) / split
        dst_dir = Path(dst_root) / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        mp4_files = list(src_dir.glob("*.mp4"))

        print(f"\nProcessing {split} set: {len(mp4_files)} files")
        for mp4_file in tqdm(mp4_files, desc=f"Converting {split}", unit="file"):
            wav_filename = mp4_file.with_suffix(".wav").name
            dst_path = dst_dir / wav_filename

            try:
                (
                    ffmpeg.input(str(mp4_file))
                    .output(str(dst_path), acodec="pcm_s16le", ac=1, ar="16000")  # mono, 16kHz
                    .run(quiet=True, overwrite_output=True)
                )
            except ffmpeg.Error as e:
                failed_files.append(mp4_file)
                tqdm.write(f"Failed: {mp4_file.name} - {e.stderr.decode().strip().splitlines()[-1]}")

    print("\nConversion completed.")
    if failed_files:
        print(f"\n{len(failed_files)} file(s) failed to convert:")
        for f in failed_files:
            print(f"- {f}")

        with open("data/processed/meld_audio/failed_conversions.txt", "w") as log_file:
            for f in failed_files:
                log_file.write(str(f) + "\n")
        print("\nList of failed files saved to 'failed_conversions.txt'")
    else:
        print("All files converted successfully!")


if __name__ == "__main__":
    convert_mp4_to_wav_ffmpeg("data/raw/meld", "data/processed/meld_audio")
