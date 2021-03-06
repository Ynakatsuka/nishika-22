import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from kvt.augmentation import SpecAugmentationPlusPlus
from kvt.models.layers import AttBlockV2

from .audio_features import (
    Loudness,
    PCENTransform,
    add_frequency_encoding,
    add_time_encoding,
    make_delta,
)
from .conformer import ConformerBlock

try:
    from torchlibrosa.augmentation import SpecAugmentation
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram
except ImportError:
    SpecAugmentation, LogmelFilterBank, Spectrogram = None, None, None


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


def gem(x, kernel_size, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), kernel_size).pow(1.0 / p)


def do_mixup(x, lam, indices):
    shuffled_x = x[indices]
    x = lam * x + (1 - lam) * shuffled_x
    return x


class SED(nn.Module):
    def __init__(
        self,
        backbone,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        dropout_rate=0.5,
        freeze_spectrogram_parameters=True,
        freeze_logmel_parameters=True,
        use_spec_augmentation=True,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        use_gru_layer=False,
        apply_tta=False,
        use_loudness=False,
        use_spectral_centroid=False,
        apply_delta_spectrum=False,
        apply_time_freq_encoding=False,
        min_db=120,
        apply_pcen=False,
        freeze_pcen_parameters=False,
        use_multisample_dropout=False,
        multisample_dropout=0.5,
        num_multisample_dropout=5,
        pooling_kernel_size=3,
        apply_dropout_second=False,
        use_batch_norm=True,
        **params,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dropout_rate = dropout_rate
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob
        self.use_gru_layer = use_gru_layer
        self.apply_tta = apply_tta
        self.use_loudness = use_loudness
        self.use_spectral_centroid = use_spectral_centroid
        self.apply_delta_spectrum = apply_delta_spectrum
        self.apply_time_freq_encoding = apply_time_freq_encoding
        self.apply_pcen = apply_pcen
        self.use_multisample_dropout = use_multisample_dropout
        self.num_multisample_dropout = num_multisample_dropout
        self.pooling_kernel_size = pooling_kernel_size
        self.apply_dropout_second = apply_dropout_second
        self.use_batch_norm = use_batch_norm

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=freeze_spectrogram_parameters,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=freeze_logmel_parameters,
        )

        # Spec augmenter
        self.spec_augmenter = None
        if use_spec_augmentation and (spec_augmentation_method is None):
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
            )
        elif use_spec_augmentation and (spec_augmentation_method is not None):
            self.spec_augmenter = SpecAugmentationPlusPlus(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
                method=spec_augmentation_method,
            )

        if self.use_gru_layer:
            self.gru = nn.GRU(in_features, in_features, batch_first=True)

        if self.use_loudness:
            self.loudness_bn = nn.BatchNorm1d(1)
            self.loudness_extractor = Loudness(
                sr=sample_rate, n_fft=n_fft, min_db=min_db,
            )

        if self.use_spectral_centroid:
            self.spectral_centroid_bn = nn.BatchNorm1d(1)

        if self.apply_pcen:
            self.pcen_transform = PCENTransform(
                trainable=~freeze_pcen_parameters,
            )

        self.bn0 = nn.BatchNorm2d(n_mels)
        self.backbone = backbone

        if self.use_multisample_dropout:
            self.big_dropout = nn.Dropout(p=multisample_dropout)

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid"
        )

    def forward(self, input, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)

        additional_features = []
        if self.use_loudness:
            loudness = self.loudness_extractor(x)
            loudness = self.loudness_bn(loudness)
            loudness = loudness.unsqueeze(-1)
            loudness = loudness.repeat(1, 1, 1, self.n_mels)
            additional_features.append(loudness)

        if self.use_spectral_centroid:
            spectral_centroid = x.mean(-1)
            spectral_centroid = self.spectral_centroid_bn(spectral_centroid)
            spectral_centroid = spectral_centroid.unsqueeze(-1)
            spectral_centroid = spectral_centroid.repeat(1, 1, 1, self.n_mels)
            additional_features.append(spectral_centroid)

        # logmel
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]

        if self.use_batch_norm:
            x = x.transpose(1, 3).contiguous()
            x = self.bn0(x)
            x = x.transpose(1, 3).contiguous()

        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (
            self.spec_augmenter is not None
        ):
            x = self.spec_augmenter(x)

        # additional features
        if self.apply_delta_spectrum:
            delta_1 = make_delta(x)
            delta_2 = make_delta(delta_1)
            additional_features.extend([delta_1, delta_2])

        if self.apply_time_freq_encoding:
            freq_encode = add_frequency_encoding(x)
            time_encode = add_time_encoding(x)
            additional_features.extend([freq_encode, time_encode])

        if self.apply_pcen:
            pcen = self.pcen_transform(x)
            additional_features.append(pcen)

        if len(additional_features) > 0:
            additional_features.append(x)
            x = torch.cat(additional_features, dim=1)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        x = x.transpose(2, 3).contiguous()
        # (batch_size, channels, freq, frames)
        x = self.backbone(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # GRU
        if self.use_gru_layer:
            # (batch_size, channels, frames) -> (batch_size, channels, frames)
            x, _ = self.gru(x.transpose(1, 2).contiguous())
            x = x.transpose(1, 2).contiguous()

        # channel smoothing
        # (batch_size, channels, frames)
        x = gem(x, kernel_size=self.pooling_kernel_size)

        if self.use_multisample_dropout:
            x = x.transpose(1, 2).contiguous()
            x = torch.mean(
                torch.stack(
                    [
                        F.relu_(self.fc1(self.big_dropout(x)))
                        for _ in range(self.num_multisample_dropout)
                    ],
                    dim=0,
                ),
                dim=0,
            )
            x = x.transpose(1, 2).contiguous()
        else:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x.transpose(1, 2).contiguous()

            x = F.relu_(self.fc1(x))
            x = x.transpose(1, 2).contiguous()

        if self.apply_dropout_second:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2).contiguous()
        segmentwise_output = segmentwise_output.transpose(1, 2).contiguous()

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }

        return output_dict


class ImageSED(nn.Module):
    def __init__(
        self,
        backbone,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        dropout_rate=0.5,
        freeze_spectrogram_parameters=True,
        freeze_logmel_parameters=True,
        use_spec_augmentation=True,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        use_gru_layer=False,
        apply_tta=False,
        use_loudness=False,
        use_spectral_centroid=False,
        apply_delta_spectrum=False,
        apply_time_freq_encoding=False,
        min_db=120,
        apply_pcen=False,
        freeze_pcen_parameters=False,
        use_multisample_dropout=False,
        multisample_dropout=0.5,
        num_multisample_dropout=5,
        pooling_kernel_size=3,
        **params,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dropout_rate = dropout_rate
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob
        self.use_gru_layer = use_gru_layer
        self.apply_tta = apply_tta
        self.use_loudness = use_loudness
        self.use_spectral_centroid = use_spectral_centroid
        self.apply_delta_spectrum = apply_delta_spectrum
        self.apply_time_freq_encoding = apply_time_freq_encoding
        self.apply_pcen = apply_pcen
        self.use_multisample_dropout = use_multisample_dropout
        self.num_multisample_dropout = num_multisample_dropout
        self.pooling_kernel_size = pooling_kernel_size

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=freeze_spectrogram_parameters,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=freeze_logmel_parameters,
            is_log=False,
        )

        self.power_to_db = torchaudio.transforms.AmplitudeToDB()

        # Spec augmenter
        self.spec_augmenter = None
        if use_spec_augmentation and (spec_augmentation_method is None):
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
            )
        elif use_spec_augmentation and (spec_augmentation_method is not None):
            self.spec_augmenter = SpecAugmentationPlusPlus(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
                method=spec_augmentation_method,
            )

        if self.use_loudness:
            self.loudness_bn = nn.BatchNorm1d(1)
            self.loudness_extractor = Loudness(
                sr=sample_rate, n_fft=n_fft, min_db=min_db,
            )

        if self.use_spectral_centroid:
            self.spectral_centroid_bn = nn.BatchNorm1d(1)

        if self.apply_pcen:
            self.pcen_transform = PCENTransform(
                trainable=~freeze_pcen_parameters,
            )

        # layers = list(backbone.children())[:-2]
        # self.backbone = nn.Sequential(*layers)
        self.backbone = backbone

        if self.use_multisample_dropout:
            self.big_dropout = nn.Dropout(p=multisample_dropout)

    def mono_to_color(self, X, eps=1e-6):
        bs = X.size(0)
        original_shape = X.shape

        X = X.view(bs, -1)
        mean = X.mean(dim=-1).unsqueeze(-1)
        std = X.std(dim=-1).unsqueeze(-1)

        X = (X - mean) / (std + eps)

        _min, _ = X.min(dim=-1)
        _max, _ = X.max(dim=-1)
        mask = (_max - _min) <= eps

        V = torch.min(torch.max(X, _min[:, None]), _max[:, None])  # clamp
        _min = _min.unsqueeze(-1)
        _max = _max.unsqueeze(-1)
        V = 255 * (V - _min) / (_max - _min)
        V[mask] = 0

        return V.view(*original_shape)

    def forward(self, input, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)

        additional_features = []
        if self.use_loudness:
            loudness = self.loudness_extractor(x)
            loudness = self.loudness_bn(loudness)
            loudness = loudness.unsqueeze(-1)
            loudness = loudness.repeat(1, 1, 1, self.n_mels)
            additional_features.append(loudness)

        if self.use_spectral_centroid:
            spectral_centroid = x.mean(-1)
            spectral_centroid = self.spectral_centroid_bn(spectral_centroid)
            spectral_centroid = spectral_centroid.unsqueeze(-1)
            spectral_centroid = spectral_centroid.repeat(1, 1, 1, self.n_mels)
            additional_features.append(spectral_centroid)

        # logmel
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        # augmentation
        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (
            self.spec_augmenter is not None
        ):
            x = self.spec_augmenter(x)

        # additional features
        if self.apply_delta_spectrum:
            delta_1 = make_delta(x)
            delta_2 = make_delta(delta_1)
            additional_features.extend([delta_1, delta_2])

        if self.apply_time_freq_encoding:
            freq_encode = add_frequency_encoding(x)
            time_encode = add_time_encoding(x)
            additional_features.extend([freq_encode, time_encode])

        if self.apply_pcen:
            pcen = self.pcen_transform(x)
            additional_features.append(pcen)

        if len(additional_features) > 0:
            additional_features.append(x)
            x = torch.cat(additional_features, dim=1)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        # power to db
        x = self.power_to_db(x)  # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(2, 3).contiguous()

        # normalize
        x = self.mono_to_color(x) / 255

        # x = F.interpolate(x, (384, 384))
        # to color
        # x = x.repeat(1, 3, 1, 1)

        # (batch_size, channels, freq, frames)
        output = self.backbone(x)

        return output


class ConformerSED(nn.Module):
    def __init__(
        self,
        backbone,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        dropout_rate=0.1,
        freeze_spectrogram_parameters=True,
        freeze_logmel_parameters=True,
        use_spec_augmentation=True,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        use_gru_layer=False,
        apply_tta=False,
        apply_encoder=False,
        **params,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dropout_rate = dropout_rate
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob
        self.use_gru_layer = use_gru_layer
        self.apply_tta = apply_tta

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=freeze_spectrogram_parameters,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=freeze_logmel_parameters,
            is_log=False,
        )

        # Spec augmenter
        self.spec_augmenter = None
        if use_spec_augmentation and (spec_augmentation_method is None):
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
            )
        elif use_spec_augmentation and (spec_augmentation_method is not None):
            self.spec_augmenter = SpecAugmentationPlusPlus(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
                method=spec_augmentation_method,
            )

        # encoder
        self.conformer = nn.Sequential(
            *[
                ConformerBlock(
                    dim=n_mels,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=31,
                    attn_dropout=dropout_rate,
                    ff_dropout=dropout_rate,
                    conv_dropout=dropout_rate,
                )
                for _ in range(3)
            ]
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(n_mels, num_classes),
        )

    def forward(self, input, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, n_mels)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, n_mels)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (
            self.spec_augmenter is not None
        ):
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        x = x.squeeze()  # -> (batch_size, time_steps, n_mels)
        x = self.conformer(x)  # -> (batch_size, time_steps, n_mels)
        x = torch.mean(x, dim=1)  # -> (batch_size, n_mels)
        logit = self.fc(x)  # -> (batch_size, num_classes)

        return logit
