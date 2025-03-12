""" Script to evaluate the X-UMX trained on SynthSOD or EnsembleSet.
Modified from the MUSDB18 example of the asteroid library.
"""

import torch
import numpy as np
import argparse
import soundfile as sf
import museval
from museval.metrics import Framing
import norbert
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import scipy.signal
from asteroid.complex_nn import torch_complex_from_magphase
import os
import sys
import yaml

from datasets import SynthSODDataset, AaltoAnechoicOrchestralDataset, URMPDataset
from utils import load_model, prepare_parser_from_dict

def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    # Literally taken from the original one on the asteroid MUSDB18 example.
    return scipy.signal.istft(X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True)[1]

def separate(
    audio,
    models,
    targets,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device="cpu",
    max_frames_for_gpu=1024,
    score=None,
    wiener=False
):
    """
    Performing the separation on audio input

    Modified from the original one on the asteroid MUSDB18 example:
        * Support evaluation of several separation models separating different targets
        * Also return the estimates without applying the Wiener filter

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels)]
        mixture audio

    x_umx_target: asteroid.models
        X-UMX model used for separating

    targets: list
        The list of instruments, e.g., ["strings", "woodwinds", "brass"]

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`

    max_frames_for_gpu: int
        maximum number of frames to send to the GPU at once, defaults to 1024

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.

    estimates_wf: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model
        after applying a Wiener filter.
    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float()

    # STFT (all the models have the same STFT encoder)
    models[0].encoder.cpu()
    spec_mag, spec_phase = models[0].encoder(audio_torch)

    estimates = {}
    masked_mixtures = []
    for i, model in enumerate(models):
        # Model inference
        masked_mixture_segs = []
        for frame_idx in range(spec_mag.shape[0] // max_frames_for_gpu + 1):
            audio_mag_seg = spec_mag[frame_idx*max_frames_for_gpu: min((frame_idx+1)*max_frames_for_gpu, spec_mag.shape[0]), ...].to(device)
            if score is not None and len(score) > 0:
                scores = [score[source] for source in model.sources]
                score_tensor = torch.from_numpy(np.array(scores))[None, ...]
                score_seg = score_tensor[:, :, frame_idx*max_frames_for_gpu: min((frame_idx+1)*max_frames_for_gpu, score_tensor.shape[2]) :].to(device)
                if "noaudio" in model.architecture:
                    est_masks_seg = model.forward_masker(score_seg.clone())
                else:
                    est_masks_seg = model.forward_masker(audio_mag_seg.clone(), score_seg.clone())
            else:
                est_masks_seg = model.forward_masker(audio_mag_seg.clone())

            masked_mixture_segs.append( model.apply_masks(audio_mag_seg, est_masks_seg).cpu().detach() )
        masked_mixtures.append( torch.concatenate(masked_mixture_segs, axis=1) )

        model.decoder.cpu()
        audio_hat = model.decoder(masked_mixtures[-1].permute(0, 2, 3, 4, 1), spec_phase)
        for j, name in enumerate(model.sources):
            estimates[name] = audio_hat[j,0,...].detach().cpu().numpy().T

    if not wiener:
        return estimates, None

    masked_mixtures = torch.concatenate(masked_mixtures, axis=0)

    source_names = []
    V = []

    for j, target in enumerate(targets):
        Vj = masked_mixtures[j, Ellipsis].numpy()
        if softmask:
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, Ellipsis])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    X = torch_complex_from_magphase(spec_mag.permute(1, 2, 3, 0), spec_phase)
    X = X.detach().cpu().numpy()
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(targets) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates_wf = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(Y[..., j].T, rate=models[0].sample_rate, n_fft=models[0].in_chan, n_hopsize=models[0].n_hop)
        estimates_wf[name] = audio_hat.T

    return estimates, estimates_wf


def generate_minus_one_track(track, remove_silent_target_frames=True):
    # Returns a track where the targets are the minus-one version of the original targets.
    output_track = deepcopy(track)
    output_track.name = track.name + "_minus_one"
    output_track.folder = track.folder + "_minus_one"

    for original_target, output_target in zip(track.targets.values(), output_track.targets.values()):
        if np.abs(original_target.audio).sum() == 0.0:
            output_target.audio = np.zeros_like(original_target.audio)  # Exclude it from the minus-one evaluation
        else:
            output_target.audio = track.audio - original_target.audio

            if remove_silent_target_frames:
                framer = Framing(1 * track.rate, 1 * track.rate, output_target.audio.shape[0])
                for t, win in enumerate(framer):
                    if np.abs(original_target.audio[win]).max() < np.abs(original_target.audio).max() / 1000:
                        output_target.audio[win] = 0.0

    return output_track


def generate_minus_one_estimate(estimates, mix=None):
    # Returns an estimates tensor with the minus-one signals generated from the input estimates.
    if mix is None:
        mix = sum(estimates.values())
    output_estimates = {}
    for source in estimates:
        valid_len = min(mix.shape[0], estimates[source].shape[0])
        output_estimates[source] = mix[:valid_len] - estimates[source][:valid_len]
    return output_estimates


def group_families_track(track):
    # Returns a track where the targets are the signals from every family of instruments.
    families = {
        "strings": ["Bass", "Cello", "Viola", "Violin"],
        "woodwinds": ["Bassoon", "Clarinet", "Flute", "Oboe"],
        "brass": ["Horn", "Trombone", "Trumpet", "Tuba"],
        "percussion": ["Harp", "Timpani", "untunedpercussion"],
    }
    for family, instruments in families.items():
        audio = np.zeros_like(track.audio)
        for instrument in instruments:
            if instrument in track.targets:
                audio += track.targets.pop(instrument).audio
        track.targets[family] = SimpleNamespace()
        track.targets[family].audio = audio
    assert len(track.targets) <= len(families)
    return track


def group_families_estimates(estimates):
    # Returns an estimates tensor grouping the estimates according to the families of instruments.
    families = {
        "strings": ["Bass", "Cello", "Viola", "Violin"],
        "woodwinds": ["Bassoon", "Clarinet", "Flute", "Oboe"],
        "brass": ["Horn", "Trombone", "Trumpet", "Tuba"],
        "percussion": ["Harp", "Timpani", "untunedpercussion"],
    }
    for family, instruments in families.items():
        audio = np.zeros_like(list(estimates.values())[0])
        for instrument in instruments:
            if instrument in estimates:
                audio += estimates.pop(instrument)
        estimates[family] = audio
    return estimates


def fake_separation(audio, targets):
    """ Replicates the output format of the separation function but leaving the original mix in every section.

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels)]
        mixture audio

    targets: list
        The list of target instruments
    """
    return {target: audio for target in targets}


def group_families_fake_separation(estimates_ns):
    # Returns a fake estimates tensor grouping the fake estimates according to the families of instruments.
    # The input should be generated with the fake_separation function, not actual estimates.
    grouped_estimates = {"strings": estimates_ns['Violin'],
                         "woodwinds": estimates_ns['Violin'],
                         "brass": estimates_ns['Violin'],
                         "percussion": estimates_ns['Violin']}
    return grouped_estimates


def inference_args(parser, remaining_args):
    # Literally taken from the original one on the asteroid MUSDB18 example.
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument(
        "--softmask",
        dest="softmask",
        action="store_true",
        help=(
            "if enabled, will initialize separation with softmask."
            "otherwise, will use mixture phase with spectrogram"
        ),
    )

    inf_parser.add_argument("--niter", type=int, default=1, help="number of iterations for refining results.")
    inf_parser.add_argument("--alpha", type=float, default=1.0, help="exponent in case of softmask separation")
    inf_parser.add_argument("--samplerate", type=int, default=44100, help="model samplerate")
    inf_parser.add_argument("--residual-model", action="store_true", help="create a model for the residual")
    return inf_parser.parse_args()

def eval_main(
    models,
    targets,
    dataset,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    outdir=None,
    save_wavs=False,
    group_families=False,
    minus_one=False,
    minus_one_model=False,
    minus_one_exclude_silences=False,
    metrics=["base", "ns", "wf"]
):
    """
    Modified from the original one on the asteroid MUSDB18 example:
        * Support the evaluation of several separation models separating different targets
        * Also evaluate the model without applying the Wiener filter
        * Evaluate minus-one signals generated from the estimates (untested, might contain bugs)
        * Evaluate the separated instruments grouped by families (untested, might contain bugs)
        * Allow for selection of which metrics to evaluate. museval is quite slow, and excluding
          the ns and wf metrics effectively yields ~3x speedup.
          base = SDR, SIR, SAR, ISR
          ns = no separation, ie. comparing the mix to the ground truth
          wf = wiener filter separation results before evaluation of base metrics
    """
    if type(save_wavs) is bool:
        save_wavs = len(dataset) if save_wavs else 0

    eval_base = "base" in metrics
    eval_ns = "ns" in metrics
    eval_wf = "wf" in metrics

    if eval_base: results, fp = museval.EvalStore(), open(os.path.join(outdir, "results.txt"), "w")
    if eval_wf: results_wf, fp_wf = museval.EvalStore(), open(os.path.join(outdir, "results_wf.txt"), "w")
    if eval_ns: results_ns, fp_ns = museval.EvalStore(), open(os.path.join(outdir, "results_ns.txt"), "w")

    for i, item in enumerate(dataset):
        track, score = item if type(item) is tuple else (item, None)
        print(f"Processing... {track.folder}", file=sys.stderr)
        
        if eval_base: print(track.folder, file=fp, flush=True)
        if eval_wf: print(track.folder, file=fp_wf, flush=True)
        if eval_ns: print(track.folder, file=fp_ns, flush=True)

        if eval_ns: estimates_ns = fake_separation(track.audio, targets)
        estimates, estimates_wf = separate(track.audio, models, targets, niter=niter, alpha=alpha, softmask=softmask, residual_model=residual_model, device=device, score=score, wiener=eval_wf)
        
        for name, gt in track.targets.items():
            framer = Framing(1 * track.rate, 1 * track.rate, track.audio.shape[0])
            for win in framer:
                if np.allclose(gt.audio[win], 0, atol=1e-2):
                    gt.audio[win] = 0
        
        if group_families:
            track = group_families_track(track)
            if eval_base: estimates = group_families_estimates(estimates)
            if eval_wf: estimates_wf = group_families_estimates(estimates_wf)
            if eval_ns: estimates_ns = group_families_fake_separation(estimates_ns)

            if np.sum(np.array([np.abs(target.audio).sum() for target in track.targets.values()]) > 0) < 2:
                print("Skiping because it only has one family", file=sys.stderr)
                continue

        if minus_one:
            track = generate_minus_one_track(track, remove_silent_target_frames=minus_one_exclude_silences)
            if not minus_one_model:
                if eval_base: estimates = generate_minus_one_estimate(estimates, mix=track.audio)
                if eval_wf: estimates_wf = generate_minus_one_estimate(estimates_wf, mix=track.audio)

        if save_wavs:
            output_path = Path(os.path.join(outdir, track.folder))
            output_path.mkdir(exist_ok=True, parents=True)
            sf.write(str(output_path / Path("_mix").with_suffix(".flac")), track.audio, samplerate)
            for target, estimate in estimates.items():
                gt = track.targets[target].audio
                sf.write(str(output_path / Path(target + "_gt").with_suffix(".flac")), gt, samplerate)
                sf.write(str(output_path / Path(target + "_est").with_suffix(".flac")), estimate, samplerate)
            save_wavs -= 1

        if eval_base:
            track_scores = museval.eval_mus_track(track, estimates)
            results.add_track(track_scores.df)
            print(track_scores, file=sys.stderr)
            print(track_scores, file=fp, flush=True)

        if eval_wf:
            track_scores_wf = museval.eval_mus_track(track, estimates_wf)
            results_wf.add_track(track_scores_wf.df)
            print(track_scores_wf, file=fp_wf, flush=True)

        if eval_ns:
            track_scores_ns = museval.eval_mus_track(track, estimates_ns)
            results_ns.add_track(track_scores_ns.df)
            print(track_scores_ns, file=fp_ns, flush=True)

    if eval_base:
        print(results, file=sys.stderr)
        print(results, file=fp, flush=True)
        results.save(os.path.join(outdir, "results.pandas"))
        results.frames_agg = "mean"
        print(results, file=sys.stderr)
        print(results, file=fp, flush=True)
        fp.close()

    if eval_wf:
        print(results_wf, file=fp_wf, flush=True)
        results_wf.save(os.path.join(outdir, "results_wf.pandas"))
        results_wf.frames_agg = "mean"
        print(results_wf, file=fp_wf, flush=True)
        fp_wf.close()

    if eval_ns:
        print(results_ns, file=fp_ns, flush=True)
        results_ns.save(os.path.join(outdir, "results_ns.pandas"))
        results_ns.frames_agg = "mean"
        print(results_ns, file=fp_ns, flush=True)
        fp_ns.close()


if __name__ == "__main__":
    # Settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)
    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    parser.add_argument(
        "--outdir",
        type=str,
        default="./x-umx_outputs",
        help="Results path and where " "best_model.pth" " is stored if no --models provided",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=[],
        nargs="+",
        help="Results paths where " "best_model.pth" " are stored",
    )
    parser.add_argument(
        "--eval_on",
        type=str,
        default=['aalto', 'urmp', 'ensembleset', 'synthsod_test'],
        nargs="+",
        help="Datasets to evaluate on. Options: aalto, urmp, ensembleset, synthsod_test, synthsod_train",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA inference")
    parser.add_argument(
        "--minus-one", action="store_true", default=False,
        help="evaluate minus one tracks instead of the separated instruments (untested, might contain bugs)"
    )
    parser.add_argument(
        "--minus-one-model", action="store_true", default=False,
        help="the model is generating the minus one trakcs instead of the separated sources (untested, might contain bugs)"
    )
    parser.add_argument(
        "--exclude-silences", action="store_true", default=False,
        help="Exclude frames where the target is silent in the minus-one mode (untested, might contain bugs)"
    )
    parser.add_argument(
        "--group-families", action="store_true", default=False,
        help="evaluate the separated instruments grouped by families (untested, might contain bugs)"
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)
    assert not (args.exclude_silences and not args.minus_one), "--exclude-silences should only be used with --minus-one"
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if len(args.models) > 0:
        models = [load_model(model_name, device) for model_name in args.models]
    else:
        model_name = os.path.abspath(os.path.join(args.outdir, "best_model.pth"))
        models = [load_model(model_name, device)]
    targets = [source for model in models for source in model.sources]

    if 'aalto' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_Aalto" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the Aalto anechoic orchestra recordings.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        aalto_dataset = AaltoAnechoicOrchestralDataset(
            args.aalto_dataset_path,
            sample_rate=args.sample_rate,
            use_score=args.architecture != "noscore",
            window_size=args.window_length,
            hop_size=args.nhop,
            center=True,
            n_fft=args.in_chan,
            )
        eval_main(
            models,
            targets,
            aalto_dataset,
            samplerate=args.samplerate,
            alpha=args.alpha,
            softmask=args.softmask,
            niter=args.niter,
            residual_model=args.residual_model,
            outdir=outdir,
            save_wavs=True,
            group_families=args.group_families,
            minus_one=args.minus_one,
            minus_one_model=args.minus_one_model,
            minus_one_exclude_silences=args.exclude_silences,
            metrics=args.metrics
        )

    if 'urmp' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_URMP" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the URMP recordings.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        urmp_dataset = URMPDataset(
            args.urmp_dataset_path,
            sample_rate=args.sample_rate,
            use_score=args.architecture != "noscore",
            window_size=args.window_length,
            hop_size=args.nhop,
            center=True,
            n_fft=args.in_chan,
            targets=targets,
            )
        eval_main(
            models,
            targets,
            urmp_dataset,
            samplerate=args.samplerate,
            alpha=args.alpha,
            softmask=args.softmask,
            niter=args.niter,
            residual_model=args.residual_model,
            outdir=outdir,
            save_wavs=True,
            group_families=args.group_families,
            minus_one=args.minus_one,
            minus_one_model=args.minus_one_model,
            minus_one_exclude_silences=args.exclude_silences,
            metrics=args.metrics
        )

    if 'synthsod_test' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_SynthSOD_test" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the test partition of the SynthSOD Dataset.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        test_dataset = SynthSODDataset(
            metadata_file_path=args.synthsod_dataset_path + '/SynthSOD_metadata_aligned_test.json',
            synthsod_data_path=args.synthsod_dataset_path + '/SynthSOD_data/',
            score_data_path=args.synthsod_dataset_path + '/score_data',
            window_size=args.window_length,
            hop_size=args.nhop,
            center=True,
            n_fft=args.in_chan,
            sources=args.sources,
            targets=targets,
            convert_to_mono=(args.nb_channels == 1),
            join_violins=args.join_violins,
            segment=0,
            sample_rate=args.sample_rate,
            fake_musdb_format=True,
            max_duration=30*60,
            use_score="noscore" not in args.architecture,
            eval=True
        )

        eval_main(
            models,
            targets,
            test_dataset,
            samplerate=args.samplerate,
            alpha=args.alpha,
            softmask=args.softmask,
            niter=args.niter,
            residual_model=args.residual_model,
            outdir=outdir,
            save_wavs=20,
            group_families=args.group_families,
            minus_one=args.minus_one,
            minus_one_model=args.minus_one_model,
            minus_one_exclude_silences=args.exclude_silences,
            metrics=args.metrics
        )
    
    # you can use song:<song_name> to eval a specific song from synthsod
    if 'song:' in args.eval_on[0]:
        songname = args.eval_on[0].split(":")[1]
        outdir = os.path.join(os.path.abspath(args.outdir), f"EvaluateResults_{songname}" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print(f"Evaluating models on song: {songname}", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        test_dataset = SynthSODDataset(
            metadata_file_path=args.synthsod_dataset_path + '/SynthSOD_metadata_aligned_all.json',
            synthsod_data_path=args.synthsod_dataset_path + '/SynthSOD_data/',
            score_data_path=args.synthsod_dataset_path + '/score_data',
            window_size=args.window_length,
            hop_size=args.nhop,
            center=True,
            n_fft=args.in_chan,
            sources=args.sources,
            targets=targets,
            convert_to_mono=(args.nb_channels == 1),
            join_violins=args.join_violins,
            segment=0,
            sample_rate=args.sample_rate,
            fake_musdb_format=True,
            max_duration=30*60,
            eval=True, 
            eval_song=songname
        )
        eval_main(
            models,
            targets,
            test_dataset,
            samplerate=args.samplerate,
            alpha=args.alpha,
            softmask=args.softmask,
            niter=args.niter,
            residual_model=args.residual_model,
            outdir=outdir,
            save_wavs=20,
            group_families=args.group_families,
            minus_one=args.minus_one,
            minus_one_model=args.minus_one_model,
            minus_one_exclude_silences=args.exclude_silences,
            metrics=args.metrics
        )
        
