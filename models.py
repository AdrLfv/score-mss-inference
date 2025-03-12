"""
Code taken from https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/x_umx.py#L319,
and modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LSTM, Linear, BatchNorm1d, Parameter

from asteroid.models.base_models import BaseModel
from asteroid.utils.hub_utils import cached_download


class XUMX(BaseModel):
    r"""

    Args:
        sources (list): The list of instruments, e.g., ["bass", "drums", "vocals"],
            defined in conf.yml.
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        sample_rate (int): sampling rate of input wavs
        nb_layers (int): Number of (B)LSTM layers in network.
        input_mean (torch.tensor): Mean for each frequency bin calculated
            in advance to normalize the mixture magnitude spectrogram.
        input_scale (torch.tensor): Standard deviation for each frequency bin
            calculated in advance to normalize the mixture magnitude spectrogram.
        max_bin (int): Maximum frequency bin index of the mixture that X-UMX
            should consider. Set to None to use all frequency bins.
        bidirectional (bool): whether we use LSTM or BLSTM.
        spec_power (int): Exponent for spectrogram calculation.
        return_time_signals (bool): Set to true if you are using a time-domain
            loss., i.e., applies ISTFT. If you select ``MDL=True'' via
            conf.yml, this is set as True.

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(
        self,
        sources,
        architecture="noscore",
        window_length=4096,
        in_chan=4096,
        n_hop=1024,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        bidirectional=True,
        spec_power=1,
        return_time_signals=False,
        score_hidden_size=32,
    ):
        super().__init__(sample_rate)

        self.score_dim = 128
        self.architecture = architecture
        self.window_length = window_length
        self.in_chan = in_chan
        self.n_hop = n_hop
        self.sources = sources
        self._return_time_signals = return_time_signals
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.bidirectional = bidirectional
        self.nb_output_bins = in_chan // 2 + 1
        if max_bin:
            self.max_bin = max_bin
        else:
            self.max_bin = self.nb_output_bins

        self.hidden_size = hidden_size
        self.score_hidden_size = score_hidden_size
        self.spec_power = spec_power

        input_mean = torch.from_numpy(-input_mean[: self.max_bin]).float() if input_mean is not None else torch.zeros(self.max_bin)
        input_scale = torch.from_numpy(1.0 / input_scale[: self.max_bin]).float() if input_scale is not None else torch.ones(self.max_bin)

        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        spec = _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1))
        self.encoder = nn.Sequential(stft, spec) 

        lstm_hidden_size = hidden_size // 2 if bidirectional else hidden_size
        lstm_input_size = hidden_size
        src_enc = {}
        src_lstm = {}
        src_dec = {}
        mean_scale = {}
        for src in sources:
            src_enc[src] = _InstrumentBackboneEnc(
                nb_bins=self.max_bin,
                architecture=self.architecture,
                hidden_size=hidden_size,
                nb_channels=nb_channels,
                score_hidden_size=self.score_hidden_size
            )

            src_lstm[src] = LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=nb_layers,
                bidirectional=bidirectional,
                batch_first=False,
                dropout=0.4,
            )

            src_dec[src] = _InstrumentBackboneDec(
                nb_output_bins=self.nb_output_bins,
                hidden_size=hidden_size,
                nb_channels=nb_channels,
            )

            mean_scale["input_mean_{}".format(src)] = Parameter(input_mean.clone())
            mean_scale["input_scale_{}".format(src)] = Parameter(input_scale.clone())
            mean_scale["output_mean_{}".format(src)] = Parameter(torch.ones(self.nb_output_bins).float())
            mean_scale["output_scale_{}".format(src)] = Parameter(torch.ones(self.nb_output_bins).float())
            
        self.layer_enc = nn.ModuleDict(src_enc)
        self.layer_lstm = nn.ModuleDict(src_lstm)
        self.layer_dec = nn.ModuleDict(src_dec)
        self.mean_scale = nn.ParameterDict(mean_scale)

        self.decoder = _ISTFT(window=stft.window, n_fft=in_chan, hop_length=n_hop, center=True)
    
    def forward(self, wav, piano_roll=None):
        """Model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            masked_mixture (torch.Tensor): estimated spectrograms masked by
                X-UMX's output of shape $(sources, frames, batch_size, channels, bins)$
            time_signals (torch.Tensor): estimated time signals of shape $(sources, batch_size, channels, time_length)$ if `return_time_signals` is `True`
        """
        mixture, ang = self.encoder(wav)

        est_masks = self.forward_masker(mixture.clone(), piano_roll)

        masked_mixture = self.apply_masks(mixture, est_masks)

        if self._return_time_signals:
            spec = masked_mixture.permute(0, 2, 3, 4, 1)
            time_signals = self.decoder(spec, ang)
        else:
            time_signals = None

        return masked_mixture, time_signals

    def forward_masker(self, input_spec, piano_roll=None):
        shapes = input_spec.data.shape

        x = input_spec[..., : self.max_bin]

        if piano_roll is not None: piano_roll = piano_roll.transpose(0, 1)

        inputs = [x]
        for i in range(1, len(self.sources)):
            inputs.append(x.clone())

        for i, src in enumerate(self.sources):
            inputs[i] += self.mean_scale[f"input_mean_{src}"]
            inputs[i] *= self.mean_scale[f"input_scale_{src}"]
            if self.architecture == "input_concat":
                inputs[i] = self.layer_enc[src](inputs[i], shapes, piano_roll[i])
            else:
                inputs[i] = self.layer_enc[src](inputs[i], shapes)

        cross_1 = sum(inputs) / len(self.sources)
        cross_2 = 0.0
        for i, src in enumerate(self.sources):
            tmp_lstm_out = self.layer_lstm[src](cross_1)
            cross_2 += torch.cat([inputs[i], tmp_lstm_out[0]], -1)

        cross_2 /= len(self.sources)
        mask_list = []
        for i, src in enumerate(self.sources):
            x_tmp = self.layer_dec[src](cross_2, shapes)
            x_tmp *= self.mean_scale[f"output_scale_{src}"]
            x_tmp += self.mean_scale[f"output_mean_{src}"]
            mask_list.append(F.relu(x_tmp))
        est_masks = torch.stack(mask_list, dim=0)

        return est_masks

    def apply_masks(self, mixture, est_masks):
        masked_tf_rep = torch.stack([mixture * est_masks[i] for i in range(len(self.sources))])
        return masked_tf_rep

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = {
            "window_length": self.window_length,
            "in_chan": self.in_chan,
            "n_hop": self.n_hop,
            "sample_rate": self.sample_rate,
        }

        net_config = {
            "sources": self.sources,
            "hidden_size": self.hidden_size,
            "nb_channels": self.nb_channels,
            "input_mean": None,
            "input_scale": None,
            "max_bin": self.max_bin,
            "nb_layers": self.nb_layers,
            "bidirectional": self.bidirectional,
            "spec_power": self.spec_power,
            "return_time_signals": False,
            "architecture": self.architecture,
            "score_hidden_size": self.score_hidden_size
        }

        model_args = {
            **fb_config,
            **net_config,
        }
        return model_args

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu", weights_only=False)
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)
        model = XUMX(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

class _InstrumentBackboneEnc(nn.Module):
    """Encoder structure that maps the mixture magnitude spectrogram to
    smaller-sized features which are the input for the LSTM layers.

    Args:
        nb_bins (int): Number of frequency bins of the mixture.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model
            (1 for mono (spectral downmix is applied,) 2 for stereo).
    """

    def __init__(
        self,
        nb_bins,
        architecture,
        hidden_size=512,
        nb_channels=2,
        score_hidden_size=32
    ):
        super().__init__()

        self.score_dim = 128
        self.score_hidden_size = score_hidden_size
        self.architecture = architecture
        self.max_bin = nb_bins
        if architecture == "input_concat":
            self.max_bin = nb_bins + 2*score_hidden_size
        self.hidden_size = hidden_size
        self.enc = nn.Sequential(
            Linear(self.max_bin * nb_channels, hidden_size, bias=False),
            BatchNorm1d(hidden_size),
        )
        if architecture == "input_concat":
            self.score_enc = LSTM(input_size=self.score_dim, hidden_size=self.score_hidden_size, num_layers=3, bidirectional=True, batch_first=False, dropout=0.3)
            self.score_norm = BatchNorm1d(2*self.score_hidden_size)

    def forward(self, x, shapes, piano_roll=None):
        nb_frames, nb_samples, nb_channels, _ = shapes
        if self.architecture == "input_concat":
            piano_roll = piano_roll.transpose(0, 1)
            piano_roll = self.score_enc(piano_roll)[0].permute(1, 2, 0)
            piano_roll = F.relu(self.score_norm(piano_roll)).permute(2, 0, 1)
            x = torch.cat((x.squeeze(), piano_roll.squeeze()), dim=-1)

        x = x.reshape(-1, nb_channels * self.max_bin)
        x = self.enc(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)
        return x


class _InstrumentBackboneDec(nn.Module):
    """Decoder structure that maps output of LSTM layers to
    magnitude estimate of an instrument.

    Args:
        nb_output_bins (int): Number of frequency bins of the instrument estimate.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): Number of output bins depending on STFT size.
            It is generally calculated ``(STFT size) // 2 + 1''.
    """

    def __init__(
        self,
        nb_output_bins,
        hidden_size=512,
        nb_channels=2,
    ):
        super().__init__()
        self.nb_output_bins = nb_output_bins
        self.dec = nn.Sequential(
            Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False),
            BatchNorm1d(hidden_size),
            nn.ReLU(),
            Linear(
                in_features=hidden_size, out_features=self.nb_output_bins * nb_channels, bias=False
            ),
            BatchNorm1d(self.nb_output_bins * nb_channels),
        )

    def forward(self, x, shapes):
        nb_frames, nb_samples, nb_channels, _ = shapes
        x = self.dec(x.reshape(-1, x.shape[-1]))
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        return x


class _STFT(nn.Module):
    def __init__(self, window_length, n_fft=4096, n_hop=1024, center=True):
        super(_STFT, self).__init__()
        self.window = Parameter(torch.hann_window(window_length), requires_grad=False)
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        x = x.reshape(nb_samples * nb_channels, -1)

        stft_f = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        stft_f = torch.view_as_real(stft_f)

        stft_f = stft_f.contiguous().view(nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2)
        return stft_f


class _Spectrogram(nn.Module):
    def __init__(self, spec_power=1, mono=True):
        super(_Spectrogram, self).__init__()
        self.spec_power = spec_power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram and the corresponding phase
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        phase = stft_f.detach().clone()
        phase = torch.atan2(phase[Ellipsis, 1], phase[Ellipsis, 0])

        stft_f = stft_f.transpose(2, 3)

        stft_f = stft_f.pow(2).sum(-1).pow(self.spec_power / 2.0)

        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)
            phase = torch.mean(phase, 1, keepdim=True)

        return [stft_f.permute(2, 0, 1, 3), phase]


class _ISTFT(nn.Module):
    def __init__(self, window, n_fft=4096, hop_length=1024, center=True, sample_rate=44100):
        super(_ISTFT, self).__init__()
        self.window = window
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

    def forward(self, spec, ang):
        sources, bsize, channels, fbins, frames = spec.shape
        x_r = spec * torch.cos(ang)
        x_i = spec * torch.sin(ang)
        x = torch.stack([x_r, x_i], dim=-1)
        x = x.view(sources * bsize * channels, fbins, frames, 2)
        x = torch.view_as_complex(x)

        wav = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=self.center
        )
        wav = wav.view(sources, bsize, channels, wav.shape[-1])

        return wav


class NoaudioXUMX(BaseModel):
    r"""https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/x_umx.py#L319
    This codebase was previously using the xumx implementation from asteroid, I have just copied
    it here and modified it.

    Args:
        sources (list): The list of instruments, e.g., ["bass", "drums", "vocals"],
            defined in conf.yml.
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        sample_rate (int): sampling rate of input wavs
        nb_layers (int): Number of (B)LSTM layers in network.
        input_mean (torch.tensor): Mean for each frequency bin calculated
            in advance to normalize the mixture magnitude spectrogram.
        input_scale (torch.tensor): Standard deviation for each frequency bin
            calculated in advance to normalize the mixture magnitude spectrogram.
        max_bin (int): Maximum frequency bin index of the mixture that X-UMX
            should consider. Set to None to use all frequency bins.
        bidirectional (bool): whether we use LSTM or BLSTM.
        spec_power (int): Exponent for spectrogram calculation.
        return_time_signals (bool): Set to true if you are using a time-domain
            loss., i.e., applies ISTFT. If you select ``MDL=True'' via
            conf.yml, this is set as True.

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(
        self,
        sources,
        architecture,
        window_length=4096,
        in_chan=4096,
        n_hop=1024,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        max_bin=None,
        bidirectional=True,
        spec_power=1,
        return_time_signals=False,
    ):
        super().__init__(sample_rate)

        self.architecture = architecture
        assert self.architecture == "noaudio"
        self.window_length = window_length
        self.in_chan = in_chan
        self.n_hop = n_hop
        self.sources = sources
        self._return_time_signals = return_time_signals
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.bidirectional = bidirectional
        self.nb_output_bins = in_chan // 2 + 1
        if max_bin:
            self.max_bin = max_bin
        else:
            self.max_bin = self.nb_output_bins
        self.hidden_size = hidden_size
        self.spec_power = spec_power

        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        spec = _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1))
        self.encoder = nn.Sequential(stft, spec) 

        lstm_hidden_size = hidden_size // 2 if bidirectional else hidden_size
        lstm_input_size = hidden_size
        src_enc = {}
        src_lstm = {}
        src_dec = {}
        for src in sources:
            src_enc[src] = _NoaudioInstrumentBackboneEnc(
                nb_bins=self.max_bin,
                architecture=self.architecture,
                hidden_size=hidden_size,
                nb_channels=nb_channels,
            )

            src_lstm[src] = LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=nb_layers,
                bidirectional=bidirectional,
                batch_first=False,
                dropout=0.4,
            )

            src_dec[src] = _NoaudioInstrumentBackboneDec(
                nb_output_bins=self.nb_output_bins,
                hidden_size=hidden_size,
                nb_channels=nb_channels,
            )

        self.layer_enc = nn.ModuleDict(src_enc)
        self.layer_lstm = nn.ModuleDict(src_lstm)
        self.layer_dec = nn.ModuleDict(src_dec)
        self.decoder = _ISTFT(window=stft.window, n_fft=in_chan, hop_length=n_hop, center=True)
    
    def forward(self, wav, piano_roll):
        """Model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            masked_mixture (torch.Tensor): estimated spectrograms masked by
                X-UMX's output of shape $(sources, frames, batch_size, channels, bins)$
            time_signals (torch.Tensor): estimated time signals of shape $(sources, batch_size, channels, time_length)$ if `return_time_signals` is `True`
        """
        mixture, ang = self.encoder(wav)
        est_masks = self.forward_masker(piano_roll)
        masked_mixture = self.apply_masks(mixture, est_masks)

        if self._return_time_signals:
            spec = masked_mixture.permute(0, 2, 3, 4, 1)
            time_signals = self.decoder(spec, ang)
        else:
            time_signals = None

        return masked_mixture, time_signals

    def forward_masker(self, piano_roll):
        piano_roll = piano_roll.permute(1, 2, 0, 3)
        shapes = piano_roll.shape

        enc_out = torch.zeros((shapes[0], shapes[1], shapes[2], self.hidden_size)).to(piano_roll.device)
        for i, src in enumerate(self.sources):
            enc_out[i] = self.layer_enc[src](piano_roll[i])

        cross_1 = sum(enc_out) / len(self.sources)
        cross_2 = 0.0
        for i, src in enumerate(self.sources):
            tmp_lstm_out = self.layer_lstm[src](cross_1)
            cross_2 += torch.cat([enc_out[i], tmp_lstm_out[0]], -1)

        cross_2 /= len(self.sources)
        mask_list = []
        for i, src in enumerate(self.sources):
            x_tmp = self.layer_dec[src](cross_2)
            mask_list.append(F.relu(x_tmp))
        est_masks = torch.stack(mask_list, dim=0)

        return est_masks

    def apply_masks(self, mixture, est_masks):
        masked_tf_rep = torch.stack([mixture * est_masks[i] for i in range(len(self.sources))])
        return masked_tf_rep

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = {
            "window_length": self.window_length,
            "in_chan": self.in_chan,
            "n_hop": self.n_hop,
            "sample_rate": self.sample_rate,
        }

        net_config = {
            "sources": self.sources,
            "hidden_size": self.hidden_size,
            "nb_channels": self.nb_channels,
            "max_bin": self.max_bin,
            "nb_layers": self.nb_layers,
            "bidirectional": self.bidirectional,
            "spec_power": self.spec_power,
            "return_time_signals": False,
            "architecture": self.architecture
        }

        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **net_config,
        }
        return model_args

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu", weights_only=False)
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs) 
        model = NoaudioXUMX(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

class _NoaudioInstrumentBackboneEnc(nn.Module):
    """Encoder structure that maps the mixture magnitude spectrogram to
    smaller-sized features which are the input for the LSTM layers.

    Args:
        nb_bins (int): Number of frequency bins of the mixture.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model
            (1 for mono (spectral downmix is applied,) 2 for stereo).
    """

    def __init__(
        self,
        nb_bins,
        architecture,
        hidden_size=512,
        nb_channels=2,
    ):
        super().__init__()

        self.architecture = architecture
        self.max_bin = nb_bins
        self.hidden_size = hidden_size
        self.enc = nn.Sequential(
            Linear(self.max_bin * nb_channels, hidden_size, bias=False),
            BatchNorm1d(hidden_size),
        )

    def forward(self, x):
        nb_frames = x.shape[0]
        nb_samples = x.shape[1]
        x = x.reshape(nb_frames*nb_samples, self.max_bin)
        x = self.enc(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)
        return x


class _NoaudioInstrumentBackboneDec(nn.Module):
    """Decoder structure that maps output of LSTM layers to
    magnitude estimate of an instrument.

    Args:
        nb_output_bins (int): Number of frequency bins of the instrument estimate.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): Number of output bins depending on STFT size.
            It is generally calculated ``(STFT size) // 2 + 1''.
    """

    def __init__(
        self,
        nb_output_bins,
        hidden_size=512,
        nb_channels=2,
    ):
        super().__init__()
        self.nb_output_bins = nb_output_bins
        self.nb_channels = nb_channels
        self.dec = nn.Sequential(
            Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False),
            BatchNorm1d(hidden_size),
            nn.ReLU(),
            Linear(
                in_features=hidden_size, out_features=self.nb_output_bins * nb_channels, bias=False
            ),
            BatchNorm1d(self.nb_output_bins * nb_channels),
        )

    def forward(self, x):
        nb_frames, nb_samples = x.shape[0], x.shape[1]
        x = self.dec(x.reshape(-1, x.shape[-1]))
        x = x.reshape(nb_frames, nb_samples, self.nb_channels, self.nb_output_bins)
        return x

