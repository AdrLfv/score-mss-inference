import torch
import numpy as np
import argparse
import soundfile as sf
from pathlib import Path
import librosa

from utils import load_model
from eval import separate

def inference_main(model_path, input_path, outdir, no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_path, device)
    targets = model.sources

    print(f"Loading audio file {input_path}...")
    audio, sr = librosa.load(input_path, sr=model.sample_rate, mono=False)
    audio = audio.T
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=1)

    print("Performing separation...")
    score = None
    if model.architecture == "input_concat":
        # Create a dummy piano roll if the model requires it
        n_frames = int(np.ceil(audio.shape[0] / model.n_hop))
        score = {
            source: np.zeros((n_frames, 128))
            for source in targets
        }

    estimates, _ = separate(audio, [model], targets, device=device, score=score, wiener=False)

    output_path = Path(outdir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving separated tracks to {outdir}...")
    for target, estimate in estimates.items():
        sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, sr)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for a single audio file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file (.wav).")
    parser.add_argument("--outdir", type=str, default="./output", help="Directory to save the separated tracks.")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA inference.")

    args = parser.parse_args()

    inference_main(args.model, args.input, args.outdir, args.no_cuda)
