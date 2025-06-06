import museval
import numpy as np


instruments = ['Violin', 'Viola', 'Cello', 'Bass', 
               'Flute', 'Clarinet', 'Oboe', 'Bassoon',
               'Horn', 'Trumpet', 'Trombone', 'Tuba',
               'Harp', 'Timpani', 'untunedpercussion']
               

experiments = [
                "exp/train_xumx_synthsod_noscore_eval/EvaluateResults_SynthSOD_test",
                "exp/train_xumx_synthsod_noaudio_eval/EvaluateResults_SynthSOD_test",
                "exp/train_xumx_synthsod_input_concat_eval/EvaluateResults_SynthSOD_test",
              ]

for experiment in experiments:
    print(experiment)

    results = museval.EvalStore()
    results.load(experiment + "/results.pandas")
    results_agg = results.agg_frames_scores()
    original = museval.EvalStore()
    original.load(experiments[0] + "/results_ns.pandas")
    original_agg = original.agg_frames_scores()

    songs = list(set([label[0] for label in list(results_agg.keys())]))

    ensemble_results = []
    ensemble_originals = []
    orchestra_results = []
    orchestra_originals = []

    for song in songs:
        song_original = [original_agg[song][instrument]['SIR'] for instrument in instruments]
        song_result = [results_agg[song][instrument]['SIR'] for instrument in instruments]

        nb_instruments = np.isfinite(np.array([results_agg[song][instrument]['SDR'] for instrument in instruments])).sum()
        if nb_instruments <= 5:
            ensemble_originals.append(song_original)
            ensemble_results.append(song_result)
        else:
            orchestra_originals.append(song_original)
            orchestra_results.append(song_result)
    
    ensemble_originals = np.array(ensemble_originals)
    ensemble_results = np.array(ensemble_results)
    orchestra_originals= np.array(orchestra_originals)
    orchestra_results = np.array(orchestra_results)

    ensemble_originals_agg = np.nanmedian(ensemble_originals, axis=0)
    ensemble_results_agg = np.nanmedian(ensemble_results, axis=0)
    orchestra_originals_agg = np.nanmedian(orchestra_originals, axis=0)
    orchestra_results_agg = np.nanmedian(orchestra_results, axis=0)

    print("Ensemble results:")
    print("Instrument | Original SDR | Separated SDR")
    
    for instrument, original, result in zip(instruments, ensemble_originals_agg, ensemble_results_agg):
        print(f"{instrument:20s} | {original:.4f} | {result:.4f}")

    print("\n")

    print("Orchestra results:")
    print("Instrument | Original SDR | Separated SDR")
    for instrument, original, result in zip(instruments, orchestra_originals_agg, orchestra_results_agg):
        print(f"{instrument:20s} | {original:.4f} | {result:.4f}")

    print("\n------\n")
