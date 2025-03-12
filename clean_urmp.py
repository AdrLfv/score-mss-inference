"""
Script to clean the URMP dataset. All recordings in URMP have a low frequency noise,
which we wanted to remove because it could affect experiment results.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pandas

URMP_PATH = "data/URMP"

def clean_song(track_path, score_start_end):
    """
    Wiener filter to remove low frequency noise which is present in all URMP recordings. To get the noise reference,
    we take the beginning and ending silences, concatenate them, and take a mean of the magnitude spectrogram of the silent parts.
    It seems that all songs have sufficiently long silences in the start and end that this produces a good noise reference. 
    """
    y, sr = librosa.load(track_path, sr=None)
    y_start_sil, _ = librosa.load(track_path, sr=None, duration=score_start_end[track_path.stem][0])
    y_end_sil, _ = librosa.load(track_path, sr=None, offset=score_start_end[track_path.stem][1])
    y_sil = np.concatenate((y_start_sil, y_end_sil))

    mag, phase = librosa.magphase(librosa.stft(y, center=False, win_length=2048, hop_length=512))
    sil_frames, _ = librosa.magphase(librosa.stft(y_sil, center=False, win_length=2048, hop_length=512))
    
    noise_ref = np.tile(np.mean(sil_frames, axis=1), (mag.shape[1], 1)).T
    filt = np.minimum(mag, noise_ref)
    margin = 10
    mask = librosa.util.softmask(mag - filt, margin * filt, power=2)
    cleaned_mag = mask * mag

    spec = cleaned_mag*(np.cos(phase)+1j*np.sin(phase))
    y = librosa.istft(spec)
    clean_track_path = track_path.with_stem(track_path.stem + "_cleaned")
    sf.write(clean_track_path, y, sr)

def get_score_start_end(song_dir):
    """
    Create a dict where the keys are each track of the song, both the mix and individual instruments are included.
    The value is the start time and end time of the score for that track in a tuple (start, end).
    """
    score_start_end = {}
    mix_start, mix_end = None, None
    for file in song_dir.iterdir():
        if file.stem[:5] == "AuMix" and file.stem[-7:] != "cleaned":
            mix_stem = file.stem
        if file.stem[:5] != "Notes": continue
        with open(file, 'r') as f:
            df = pandas.read_csv(f, delimiter=r"\s+", header=None, names=["onset", "frequency", "duration"])
            start = df["onset"].head(1).values
            end = df["onset"].tail(1).values + df["duration"].tail(1).values
            score_start_end[file.stem.replace("Notes", "AuSep")] = (start, end)
            if mix_start is None or start < mix_start:
                mix_start = start
            if mix_end is None or end > mix_end:
                mix_end = end

    score_start_end[mix_stem] = (mix_start, mix_end)
    return score_start_end

if __name__ == "__main__":
    for song_dir in Path(URMP_PATH).iterdir():
        if song_dir.stem == "Supplementary_Files" or not song_dir.is_dir(): continue
        score_start_end = get_score_start_end(song_dir)
        for track in song_dir.iterdir():
            if "." in track.stem or track.suffix != ".wav" or track.stem[-7:] == "cleaned": continue
            clean_song(track, score_start_end)
