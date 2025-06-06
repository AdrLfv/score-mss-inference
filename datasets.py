""" Dataloaders for SynthSOD, Aalto Anechoic Orchestral and URMP datasets.
Based on the dataloader for MUSDB18 from the asteroid library.
"""

import warnings
import torch
from pathlib import Path
import numpy as np
from types import SimpleNamespace
import soundfile as sf
import librosa
import random
import tqdm
import json
import csv
import math

class SynthSODDataset(torch.utils.data.Dataset):
    """SynthSOD orchestra music separation dataset with score information

    Args:
        metadata_file_path (str): Path to the JSON metadata file with the list of
            tracks to load.
        synthsod_data_path (str): Path to the data folder of the SynthSOD dataset
            with the folders of the tracks.
        score_data_path (str): Path to the data folder with the scores.
        window_size (int): Window size used for calculating piano roll, you probably
        want this to be the same as the STFT window size.        
        hop_size (int): Hop size used for calculating piano roll, you probably
        want this to be the same as the STFT hop size.        
        center (bool): Whether to center the framing of the piano roll, you probably
        want this to be the same as for the STFT.
        n_fft (int): Nfft used for calculating piano roll, you probably
        want this to be the same as the STFT nfft.        
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            to laod to the mixtures. Defaults to the 18 SynthSOD sources.
        targets (list or None, optional): List of source names to be used as targets.
            Defaults to None (all the sources are defined as targets).
        join_violins (bool, optional): Join the Violin_1 and Violin_2 sources into
            one single target. Defaults to True.
        join_piccolo_to_flute (bool, optional): Join the Piccolo source to the Flute
            target. Defaults to True.
        join_coranglais_to_oboe (bool, optional): Join the coranglais source to the
            Oboe target. Defaults to True.
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
            Default to False.
        fixed_segments (boolean, optional): Always take the same segments in every track.
            Useful for validation. Not compatible with random_track_mix.
            Default to False.
        random_track_mix (boolean, optional): enables mixing of random sources from
            different tracks to assemble mix. Untested, it might contain bugs.
            Default to False.
        convert_to_mono (bool, optional): Convert the audio to mono. Default to True.
        train_minus_one (bool, optional): Return the targets as the minus on of the
            source instead as the separated stem. Default to False.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
            It will resample the signals after loading them if they have a different
            sample rate (slow). Default to 44100.
        max_duration (float, optional): Maximum duration of the tracks.
            Tracks longer than max_duration [seconds] will be skipped. Default to None.
        mix_only (bool, optional): Return only the mixture without the targets.
            Default to False.
        fake_musdb_format (bool, optional): Return the data in the format of the MUSDB18
            dataset to be used with the museval library. Default to False.
        size_limit (int/float, optional): Limit the number of tracks to load (if integer)
            or ratio of the dataset (if float between 0 and 1).
            Default to None (load all the tracks).
        eval (bool): Whether we are evaluating or not.
            Default to False.
        eval_song (str): Name of song to evaluate. Used to evaluate a specific song. If None
            we evaluate all songs.
            Default to None.
        use_score (bool): Whether to use score or not.
            Default to False.
    """

    dataset_name = "SynthSOD"

    def __init__(
        self,
        metadata_file_path,
        synthsod_data_path,
        score_data_path,
        window_size=4096,
        hop_size=1024,
        center=True,
        n_fft=4096,
        sources=('Violin_1', 'Violin_2', 'Viola', 'Cello', 'Bass', 'Flute', 'Piccolo', 'Clarinet', 'Oboe', 'coranglais',
                 'Bassoon', 'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani', 'untunedpercussion'),
        targets=None,
        join_violins=True,
        join_piccolo_to_flute=True,
        join_coranglais_to_oboe=True,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        fixed_segments=False,
        random_track_mix=False,
        convert_to_mono=True,
        train_minus_one=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
        max_duration=None,
        mix_only=False,
        fake_musdb_format=False,
        size_limit=None,
        eval=False,
        eval_song=None,
        use_score=False,
        times=(None, None)
    ):
        assert not (fixed_segments and random_track_mix)
        assert not (mix_only and fake_musdb_format)
        #TODO: data should only contain strings if only training strings etc.

        self.metadata_file_path = Path(metadata_file_path).expanduser()
        self.synthsod_data_path = Path(synthsod_data_path).expanduser()

        self.use_score = use_score
        if self.use_score:
            self.score_data_path = Path(score_data_path).expanduser()
            self.scores = list(score_data_path)
            if not self.scores:
                raise RuntimeError("No scores found")

        self.sources = sources
        self.targets = targets if targets is not None else sources
        self.join_violins = join_violins
        self.join_piccolo_to_flute = join_piccolo_to_flute
        self.join_coranglais_to_oboe = join_coranglais_to_oboe
        if join_violins:
            self.targets = [target if target != 'Violin_1' else 'Violin'
                            for target in self.targets if target != 'Violin_2']
        if join_piccolo_to_flute:
            self.targets = [target for target in self.targets if target != 'Piccolo']
        if join_coranglais_to_oboe:
            self.targets = [target for target in self.targets if target != 'coranglais']

        self.segment = int(segment * sample_rate) if segment else None
        self.samples_per_track = samples_per_track
        self.random_segments = random_segments
        self.fixed_segments = fixed_segments
        self.random_track_mix = random_track_mix
        self.convert_to_mono = convert_to_mono
        self.train_minus_one = train_minus_one
        self.source_augmentations = source_augmentations
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.mix_only = mix_only
        self.fake_musdb_format = fake_musdb_format

        self.tracks = self.get_tracks(eval_song)
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        if fixed_segments:
            self.tracks_start = [None, ] * (len(self.tracks) * self.samples_per_track)
        if size_limit is not None:
            if 0 < size_limit < 1:
                size_limit = int(size_limit * len(self.tracks))
            self.tracks = self.tracks[:size_limit]

        self.instrument_to_midi_number = {
                "untunedpercussion": 0,
                "Violin": 40,
                "Violin_1": 40,
                "Violin_2": 40,
                "Viola": 41,
                "Cello": 42,
                "Bass": 43,
                "Harp": 46,
                "Timpani": 47,
                "Trumpet": 56,
                "Trombone": 57,
                "Tuba": 58,
                "Horn": 60,
                "Oboe": 68,
                "coranglais": 69,
                "Bassoon": 70,
                "Clarinet": 71,
                "Piccolo": 72,
                "Flute": 73
        }
        self.window_size = window_size
        self.hop_size = hop_size
        self.center = center
        self.n_fft = n_fft
        self.eval = eval
        self.times = times


    def __getitem__(self, index):
        # load sources
        audio_sources = {}
        scores = {} if self.use_score else np.array([])
        track_name = ""
        for source in self.sources:
            track_id = random.choice(range(len(self.tracks))) if self.random_track_mix else index // self.samples_per_track
            track_path = next(iter(self.tracks[track_id].values()))
            track_name = track_path.split("/")[-3]
            start, end = self.get_track_segment(track_id)
            if source in self.tracks[track_id]:
                try:
                    audio, sr = sf.read(str(self.tracks[track_id][source]), start=start, stop=end, always_2d=True)
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sample_rate, res_type='polyphase').T
                except RuntimeError:
                    print(f"Failed to open {Path(self.tracks[track_id][source])} with start={start} and end={end}.")
                    print(f"Replacing the source by silence.")
                    audio = np.zeros((end-start, 2))
                audio = audio.astype(np.float32).T
                if self.convert_to_mono:
                    audio = np.mean(audio, axis=0, keepdims=True)
                audio_sources[source] = self.source_augmentations(audio)

                if self.use_score:
                    score_file = self.score_data_path / (track_name + '.txt')
                    score = self.get_score(score_file, start/self.sample_rate, end/self.sample_rate, source)
                    scores[source] = piano_roll(score, self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, audio_sources[source].shape[1], start)
            else:
                audio_sources[source] = np.zeros((1 if self.convert_to_mono else 2, end-start), dtype=np.float32)
                if self.use_score: scores[source] = piano_roll([], self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, audio_sources[source].shape[1], start)
        audio_mix = np.sum(list(audio_sources.values()), axis=0)

        if self.mix_only:
            return audio_mix, scores

        if self.targets:
            if 'Violin' in self.targets and self.join_violins:
                audio_sources['Violin'] = audio_sources.pop('Violin_1') + audio_sources.pop('Violin_2')
                if self.use_score: scores["Violin"] = np.clip(scores['Violin_1'] + scores.pop('Violin_2'), 0, 1)
            if 'Flute' in self.targets and self.join_piccolo_to_flute:
                audio_sources['Flute'] += audio_sources.pop('Piccolo')
                if self.use_score: scores['Flute'] = np.clip(scores['Flute'] + scores.pop('Piccolo'), 0, 1)
            if 'Oboe' in self.targets and self.join_coranglais_to_oboe:
                audio_sources['Oboe'] += audio_sources.pop('coranglais')
                if self.use_score: scores['Oboe'] = np.clip(scores['Oboe'] + scores.pop('coranglais'), 0, 1)
                
            audio_sources = np.stack([audio_sources[target] for target in self.targets], axis=0)
            if self.eval:
                if self.use_score: scores = {k: v for k, v in scores.items() if k in self.targets}
            else:
                if self.use_score: scores = np.array([scores[target] for target in self.targets])

            #TODO: implement train_minus_one for score data?
            if self.train_minus_one:
                audio_sources = audio_mix - audio_sources
                raise NotImplementedError()
        if self.fake_musdb_format:
            sample_source = list(self.tracks[track_id].keys())[0]  # A source present in the track
            fake_musdb_track = SimpleNamespace()
            fake_musdb_track.name = Path(self.tracks[track_id][sample_source]).parts[-3]
            fake_musdb_track.folder = Path(self.tracks[track_id][sample_source]).parts[-3]
            fake_musdb_track.rate = self.sample_rate
            fake_musdb_track.subset = ""
            fake_musdb_track.audio = audio_mix.T
            fake_musdb_track.targets = {target_name: SimpleNamespace() for target_name in self.targets}
            for target_name, target_audio in zip(self.targets, audio_sources):
                fake_musdb_track.targets[target_name].audio = target_audio.T
            return fake_musdb_track, scores
        
        return audio_mix, scores, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_track_segment(self, track_idx):
        """Return the segment of the track to load"""
        if self.times[0] is not None: return self.times[0]*self.sample_rate, self.times[1]*self.sample_rate
        track_info = sf.info(list(self.tracks[track_idx].values())[0])  # Get the path of the first source (all sources have the same length)
        track_len = track_info.frames
        if self.segment is None or self.segment == 0 or not self.random_segments: return 0, track_len
        if self.fixed_segments:
            if self.tracks_start[track_idx] is None:
                self.tracks_start[track_idx] = int(random.uniform(0, track_len - self.segment))
            start = self.tracks_start[track_idx]
            end = start + self.segment
        else:
            start = int(random.uniform(0, track_len - self.segment))
            end = start + self.segment
        return start, end       

    def get_tracks(self, song=None):
        """Return the path to the audio files"""
        with open(self.metadata_file_path, 'r') as fp:
            db_info_dict = json.load(fp)

        p = self.synthsod_data_path

        tracks = []
        for song_info in db_info_dict['songs'].values():
            if song is not None and song_info['song_name'] != song: continue
            if self.max_duration is None or song_info['duration'] <= self.max_duration:
                track_path = p / song_info['song_name'] / 'Tree'
                if track_path.is_dir():
                    sources_paths = {}
                    for source in self.sources:
                        if (track_path / (source+'.flac')).is_file():
                            sources_paths[source] = str(track_path / (source+'.flac'))
                    # TODO: Check that the track is at least as long as the requested segments?
                    tracks.append(sources_paths)
                else:
                    warnings.warn(f"Track {track_path} not found")

        return tracks

    def get_score(self, file, start, end, instrument):
        """Return the rows of score information in file which are between the start and end times and belong to instrument"""
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            score = []
            for i, row in enumerate(reader):
                if i == 0 or float(row[1]) < start: continue
                elif float(row[0]) > end: break
                elif int(row[3]) == self.instrument_to_midi_number[instrument]:
                    score.append(list(map(float, row[:3])))
            return score

class AaltoAnechoicOrchestralDataset(torch.utils.data.Dataset):
    """Aalto Anechoic Orchestral Dataset.

    Pytorch dataset for the Aalto Anechoic Orchestral recordings.
    Only for evaluation, not for training.

    https://research.cs.aalto.fi/acoustics/virtual-acoustics/research/acoustic-measurement-and-analysis/85-anechoic-recordings.html

    Rather than the original recordings, this class is designed for the denoised versions available in
    https://www.upf.edu/web/mtg/phenicx-anechoic
    """

    def __init__(self,
                 root_path,
                 sources=('Violin_1', 'Violin_2', 'Viola', 'Cello', 'Bass', 'Flute', 'Clarinet', 'Oboe', 'Bassoon',
                          'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani', 'untunedpercussion'),
                 targets=None,
                 join_violins=True,
                 sample_rate=44100,
                 use_score=False,
                 window_size=4096,
                 hop_size=1024,
                 center=True,
                 n_fft=4096,
                 eval_song=None,
                 times=(None, None)
                 ):

        self.sod2aalto = {'Bass': ['doublebass'],
                          'Bassoon': ['bassoon'],
                          'Cello': ['cello'],
                          'Clarinet': ['clarinet'],
                          'Flute': ['flute'],
                          'Harp': [],
                          'Horn': ['horn'],
                          'Oboe': ['oboe'],
                          'Timpani': [],
                          'Trombone': [],
                          'Trumpet': ['trumpet'],
                          'Tuba': [],
                          'Viola': ['viola'],
                          'Violin_1': ['violin'],
                          'Violin_2': [],
                          'Violin': ['violin'],
                          'untunedpercussion': [],
                         }

        self.sources = sources
        self.targets = targets if targets is not None else sources
        self.sources = [source if source != 'Violin_1' else 'Violin'
                        for source in self.sources if source != 'Violin_2']
        self.targets = [target if target != 'Violin_1' else 'Violin'
                        for target in self.targets if target != 'Violin_2']
        self.sources = {source: self.sod2aalto[source] for source in self.sources}
        self.targets = {target: self.sod2aalto[target] for target in self.targets}
        self.join_violins = join_violins
        self.sample_rate = sample_rate
        self.use_score = use_score
        self.window_size = window_size
        self.hop_size = hop_size
        self.center = center
        self.n_fft = n_fft
        self.eval_song = eval_song
        self.times = times

        self.tracks = list(self.get_tracks(root_path))
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def __getitem__(self, index):
        return self.tracks[index]

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self, root_path):
        p = Path(root_path + '/audio/')

        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                if self.eval_song is not None and track_path.stem != self.eval_song: continue
                fake_musdb_track = None
                for inst_path in track_path.iterdir():
                    if inst_path.suffix == ".wav":
                        if self.times[0] is None:
                            signal, fs = sf.read(str(inst_path), always_2d=True)
                        else:
                            signal, fs = sf.read(str(inst_path), always_2d=True, start=self.times[0]*self.sample_rate, stop=self.times[1]*self.sample_rate)
                        if fs != self.sample_rate:
                            signal = librosa.resample(signal.T, orig_sr=fs, target_sr=self.sample_rate, res_type='polyphase').T
                        if fake_musdb_track is None:
                            fake_musdb_track = init_musdb_track(track_path, fs, signal, self.targets)
                        assert fs == fake_musdb_track.rate
                        for source, instruments in self.sources.items():
                            if any([instrument in inst_path.stem for instrument in instruments]):
                                if len(signal) != len(fake_musdb_track.audio):
                                    # Only happens for Beethoven's double-bass
                                    signal = np.pad(signal, ((0, len(fake_musdb_track.audio) - len(signal)), (0, 0)))
                                fake_musdb_track.audio += signal
                                fake_musdb_track.instruments.append(inst_path.stem)
                                if source in self.targets:
                                    fake_musdb_track.targets[source].audio += signal
                                    fake_musdb_track.targets[source].instruments.append(inst_path.stem)
                if fake_musdb_track is not None:
                    scores = {}
                    if self.use_score:
                        for target, instruments in self.targets.items():
                            scores[target] = piano_roll([], self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], 0)
                            for instrument in instruments:
                                score_path = root_path + '/annotations/' + fake_musdb_track.name + '/' + instrument + '.txt'
                                try:
                                    if self.times[0] is None:
                                        scores[target] += piano_roll(self.get_score(score_path), self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], 0)
                                    else:
                                        scores[target] += piano_roll(self.get_score(score_path, start=self.times[0], end=self.times[1]), self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], self.times[0]*self.sample_rate)
                                except FileNotFoundError:
                                    continue
                            scores[target] = np.clip(scores[target], 0, 1)
                    
                    peak = np.max(np.abs(fake_musdb_track.audio))
                    fake_musdb_track.audio /= (peak / 0.75)
                    eps = 1e-4 # the silences in this dataset are not set to exactly zero, so we do it manually, the histogram is so clear and these values are so small that this should be fine
                    fake_musdb_track.audio[np.abs(fake_musdb_track.audio) < eps] = 0
                    for target in fake_musdb_track.targets.values():
                        target.audio /= (peak / 0.75)
                        target.audio[np.abs(target.audio) < eps] = 0
                    if len(scores) > 0:
                        yield fake_musdb_track, scores
                    else:
                        yield fake_musdb_track

    def get_score(self, file, start=0, end=None):
        """
        Return the rows of score information in file which are between the start and end times.
        By default return the entire score. 
        """
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            score = []
            for row in reader:
                if (float(row[1])) < start: continue
                elif end is not None and float(row[0]) > end: break
                score.append([float(row[0]), float(row[1]), librosa.note_to_midi(row[2], round_midi=True)])
            return score


class URMPDataset(torch.utils.data.Dataset):
    """URMP Dataset

    Pytorch dataset for the audio recordings of the URMP dataset.
    Only for evaluation, not for training.

    https://labsites.rochester.edu/air/projects/URMP.html
    """

    def __init__(self,
                 root_path,
                 score_data_path=None,
                 sources=('Violin_1', 'Violin_2', 'Viola', 'Cello', 'Bass', 'Flute', 'Clarinet', 'Oboe', 'Bassoon',
                          'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani', 'untunedpercussion'),
                 targets=None,
                 join_violins=True,
                 exclude_single_instrument_tracks=True,
                 exclude_saxophone_tracks=True,
                 sample_rate=44100,
                 use_score=False,
                 window_size=4096,
                 hop_size=1024,
                 center=True,
                 n_fft=4096,
                 eval_song=None,
                 times=(None, None)
                 ):

        if score_data_path is not None:
            self.scores = list(score_data_path)
            if not self.scores:
                raise RuntimeError("No scores found")
            
        self.sod2urmp = {'Bass': ['_db_'],
                         'Bassoon': ['_bn_'],
                         'Cello': ['_vc_'],
                         'Clarinet': ['_cl_'],
                         'Flute': ['_fl_'],
                         'Harp': [],
                         'Horn': ['_hn_'],
                         'Oboe': ['_ob_'],
                         'Timpani': [],
                         'Trombone': ['_tbn_'],
                         'Trumpet': ['_tpt_'],
                         'Tuba': ['_tba_'],
                         'Viola': ['_va_'],
                         'Violin_1': ['_vn_'],
                         'Violin_2': ['_vn_'],
                         'Violin': ['_vn_'],
                         'untunedpercussion': [],
                        }

        self.urmp2sod = {'db': 'Bass',
                         'bn': 'Bassoon',
                         'vc': 'Cello',
                         'cl': 'Clarinet',
                         'fl': 'Flute',
                         'hn': 'Horn',
                         'ob': 'Oboe',
                         'tbn': 'Trombone',
                         'tpt': 'Trumpet',
                         'tba': 'Tuba',
                         'va': 'Viola',
                         'vn': 'Violin',
                        } 

        sources = [source for source in sources if source in self.sod2urmp.keys()]
        self.sources = sources
        self.targets = targets if targets is not None else sources
        if join_violins:
            self.sources = [source if source != 'Violin_1' else 'Violin'
                            for source in self.sources if source != 'Violin_2']
            self.targets = [target if target != 'Violin_1' else 'Violin'
                            for target in self.targets if target != 'Violin_2']
        self.sources = {source: self.sod2urmp[source] for source in self.sources}
        self.targets = {target: self.sod2urmp[target] for target in self.targets}
        self.join_violins = join_violins
        self.exclude_single_instrument_tracks = exclude_single_instrument_tracks
        self.exclude_saxophone_tracks = exclude_saxophone_tracks
        self.sample_rate = sample_rate
        self.use_score = use_score
        self.window_size = window_size
        self.hop_size = hop_size
        self.center = center
        self.n_fft = n_fft
        self.eval_song = eval_song
        self.times = times

        self.tracks = list(self.get_tracks(root_path))
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def __getitem__(self, index):
        return self.tracks[index]

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self, root_path):
        p = Path(root_path)

        for track_path in tqdm.tqdm(p.iterdir()):
            scores = {}
            if track_path.is_dir():
                if self.eval_song is not None and self.eval_song != track_path.stem: continue
                if self.exclude_saxophone_tracks and '_sax' in track_path.stem: continue
                fake_musdb_track = None
                for inst_path in sorted(track_path.iterdir()):
                    if inst_path.stem[:6] == "AuSep_" and inst_path.stem[-7:] == "cleaned" and inst_path.suffix == ".wav":
                        if self.times[0] is None:
                            signal, fs = sf.read(str(inst_path), always_2d=True)
                        else:
                            signal, fs = sf.read(str(inst_path), always_2d=True, start=self.times[0]*self.sample_rate, stop=self.times[1]*self.sample_rate)
                        if fs != self.sample_rate:
                            signal = librosa.resample(signal.T, orig_sr=fs, target_sr=self.sample_rate, res_type='polyphase').T
                            fs = self.sample_rate
                        if fake_musdb_track is None:
                            fake_musdb_track = init_musdb_track(track_path, fs, signal, self.targets)
                        assert fs == fake_musdb_track.rate
                        for source, instruments in self.sources.items():
                            if any([instrument in inst_path.stem for instrument in instruments]):
                                fake_musdb_track.audio += signal
                                fake_musdb_track.instruments.append(inst_path.stem)
                                if source in self.targets:
                                    fake_musdb_track.targets[source].audio += signal
                                    fake_musdb_track.targets[source].instruments.append(inst_path.stem)
                    elif self.use_score and inst_path.stem[:5] == "Notes":
                        inst = self.urmp2sod[inst_path.stem.split('_')[2]]
                        if inst not in scores:
                            scores[inst] = piano_roll([], self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], 0)

                        if self.times[0] is None:
                            scores[inst] += piano_roll(self.get_score(inst_path), self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], 0)
                        else:
                            scores[inst] += piano_roll(self.get_score(inst_path, start=self.times[0], end=self.times[1]), self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], self.times[0]*self.sample_rate)
                        scores[inst] = np.clip(scores[inst], 0, 1)

                if fake_musdb_track is None or (self.exclude_single_instrument_tracks and len(fake_musdb_track.instruments) == 1):
                    continue
                if fake_musdb_track is not None:
                    if len(scores) == 1:
                        warnings.warn(f"Only 1 source found for song, skipping {track_path}")
                        continue
                    peak = np.max(np.abs(fake_musdb_track.audio))
                    fake_musdb_track.audio /= (peak / 0.75)
                    for target in fake_musdb_track.targets.values():
                        target.audio /= (peak / 0.75)
                    if len(scores) > 0:
                        for target in self.targets:
                            if target not in scores:
                                scores[target] = piano_roll([], self.window_size, self.hop_size, self.sample_rate, self.center, self.n_fft, fake_musdb_track.audio.shape[0], 0)
                        yield fake_musdb_track, scores
                    else:
                        yield fake_musdb_track
    
    def get_score(self, file, start=0, end=None):
        """
        Return the rows of score information in file which are between the start and end times.
        By default return the entire score. 
        """
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            score = []
            for row in reader:
                # indices are weird because the separator is two tabs and the reader class can only handle one character delims
                if (float(row[0]) + float(row[2])) < start: continue
                elif end is not None and float(row[0]) > end: break
                # onset, frequency, duration -> onset, offset, midi note
                score.append([float(row[0]), float(row[0]) + float(row[4]), round(librosa.hz_to_midi(float(row[2])))])
            return score

def init_musdb_track(track_path, fs, signal, targets):
    musdb_track = SimpleNamespace()
    musdb_track.folder = track_path.stem
    musdb_track.name = track_path.stem
    musdb_track.rate = fs
    musdb_track.audio = np.zeros_like(signal)
    musdb_track.instruments = []
    musdb_track.targets = {target_name: SimpleNamespace() for target_name in targets}
    for target in musdb_track.targets.values():
        target.audio = np.zeros_like(signal)
        target.instruments = []
    return musdb_track

def load_synthsod_datasets(parser, args):
    """Loads the SynthSOD dataset from commandline arguments.

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()
    source_augmentations = Compose([globals()["_augment_" + aug] for aug in args.source_augmentations])

    train_dataset = SynthSODDataset(
        metadata_file_path=args.synthsod_dataset_path + '/SynthSOD_metadata_aligned_train.json',
        synthsod_data_path=args.synthsod_dataset_path + '/SynthSOD_data/',
        score_data_path=args.synthsod_dataset_path + '/score_data',
        window_size=args.window_length,
        hop_size=args.nhop,
        center=True,
        n_fft=args.in_chan,
        sources=args.sources,
        targets=args.targets,
        join_violins=args.join_violins,
        convert_to_mono=(args.nb_channels==1),
        source_augmentations=source_augmentations,
        random_track_mix=args.random_track_mix,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        size_limit=args.train_size_limit,
        train_minus_one=args.train_minus_one,
        use_score=args.architecture != "noscore"
    )

    valid_dataset = SynthSODDataset(
        metadata_file_path=args.synthsod_dataset_path + '/SynthSOD_metadata_aligned_evaluation.json',
        synthsod_data_path=args.synthsod_dataset_path + '/SynthSOD_data/',
        score_data_path=args.synthsod_dataset_path + '/score_data',
        window_size=args.window_length,
        hop_size=args.nhop,
        center=True,
        n_fft=args.in_chan,
        sources=args.sources,
        targets=args.targets,
        join_violins=args.join_violins,
        convert_to_mono=(args.nb_channels==1),
        segment=args.seq_dur,
        random_segments=True,
        fixed_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        train_minus_one=args.train_minus_one,
        use_score=args.architecture != "noscore"
    )

    return train_dataset, valid_dataset

class Compose(object):
    """Composes several augmentation transforms.
    Literally taken from the original one on the asteroid MUSDB18 example.

    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`
     Literally taken from the original one on the asteroid MUSDB18 example."""
    gain = low + np.random.rand(1).astype(np.float32) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources
     Literally taken from the original one on the asteroid MUSDB18 example."""
    if audio.shape[0] == 2 and np.random.rand(1) < 0.5:
        return np.flip(audio, [0])

    return audio

def piano_roll(score, window_size, hop_size, sample_rate, center, n_fft, n_samples, start_sample):
    n_frames = 1 + n_samples//hop_size if center else 1 + (n_samples-n_fft)//hop_size #from torch.stft docs
    out = torch.zeros(n_frames, 128)
    frame_rate = sample_rate / hop_size
    start_time = start_sample / sample_rate
    for note in score:
        if not center:
            note_start_frame = max(0, math.floor((note[0]-start_time)*frame_rate - (window_size-hop_size)/sample_rate))
            note_end_frame = math.ceil((note[1]-start_time)*frame_rate)
        else:
            note_start_frame = max(0, math.floor((note[0]-start_time)*frame_rate-(hop_size/2)/sample_rate + (window_size-hop_size)/sample_rate))
            note_end_frame = math.ceil((note[1]-start_time)*frame_rate - (hop_size/2)/sample_rate)
        out[note_start_frame:note_end_frame, int(note[2])] = 1
    return out
