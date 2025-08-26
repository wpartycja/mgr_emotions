import os
import torch
import ffmpeg
import torchaudio
import hydra


from pathlib import Path
from tqdm import tqdm
from datascripts.base_dataset import SAMPLE_RATE

SRC_ROOT = Path("data/raw/iemocap/IEMOCAP_full_release")
DST_ROOT = Path("data/processed/iemocap")

# SRC_ROOT = Path("data/processed/meld_audio") 
# DST_ROOT = Path("data/processed/meld_preprocessed") 

def convert_mp4_to_wav_ffmpeg(SRC_ROOT, DST_ROOT):
    failed_files = []

    for split in ["train", "val", "test"]:
        src_dir = Path(SRC_ROOT) / split
        dst_dir = Path(DST_ROOT) / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        mp4_files = list(src_dir.glob("*.mp4"))

        print(f"\nProcessing {split} set: {len(mp4_files)} files")
        for mp4_file in tqdm(mp4_files, desc=f"Converting {split}", unit="file"):
            wav_filename = mp4_file.with_suffix(".wav").name
            dst_path = dst_dir / wav_filename

            try:
                (
                    ffmpeg.input(str(mp4_file))
                    .output(str(dst_path), acodec="pcm_s16le", ac=1, ar=str(SAMPLE_RATE))  # mono, 16kHz
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


def preprocess_and_save(audio_path: Path, dst_path: Path, max_len_seconds: float):
    target_length = SAMPLE_RATE * max_len_seconds
    
    try:
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Pad or trim
        if waveform.size(1) < target_length:
            pad_size = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:, :target_length]

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(waveform, dst_path)

    except Exception as e:
        print(f"[ERROR] Failed: {audio_path.name} - {e}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_meld(cfg):
    max_len_seconds = cfg.dataset.max_audio_length
    splits = ["train", "val", "test"]
    
    for split in splits:
        src_dir = SRC_ROOT / split
        dst_dir = DST_ROOT / split

        wav_files = list(src_dir.glob("*.wav"))
        print(f"\nProcessing {split}: {len(wav_files)} files")

        for wav_file in tqdm(wav_files, desc=f"Preprocessing {split}"):
            dst_file = dst_dir / wav_file.with_suffix(".pt").name
            preprocess_and_save(wav_file, dst_file, max_len_seconds)

    print("\nDone preprocessing all splits!")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_iemocap(cfg):
    max_len_seconds = cfg.dataset.max_audio_length
    splits = ["Session1", "Session2", "Session3", "Session4", "Session5"]

    for session in splits:
        wav_files = list(Path(SRC_ROOT / session / "wav").rglob("*.wav"))

        print(f"\nProcessing {session}: {len(wav_files)} files")
        for wav_file in tqdm(wav_files, desc=f"Preprocessing {session}"):
            relative = wav_file.relative_to(SRC_ROOT)
            dst_file = DST_ROOT / relative.with_suffix(".pt")
            preprocess_and_save(wav_file, dst_file, max_len_seconds)


if __name__ == "__main__":
    # convert_mp4_to_wav_ffmpeg("data/raw/meld", "data/processed/meld_audio")
    
    # MELD
    # preprocess_meld(SRC_ROOT, DST_ROOT)
    
    # IEMOCAP
    preprocess_iemocap()
