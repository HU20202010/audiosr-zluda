import contextlib
import importlib
from huggingface_hub import hf_hub_download
import numpy as np
import torch

from inspect import isfunction
import os
import subprocess
import tempfile
import json
import soundfile as sf
import time
import wave
import torchaudio
import progressbar
from librosa.filters import mel as librosa_mel_fn
from audiosr.lowpass import lowpass

hann_window = {}
mel_basis = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def _locate_cutoff_freq(stft, percentile=0.97):
    def _find_cutoff(x, percentile=0.95):
        percentile = x[-1] * percentile
        for i in range(1, x.shape[0]):
            if x[-i] < percentile:
                return x.shape[0] - i
        return 0

    magnitude = torch.abs(stft)
    energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
    return _find_cutoff(energy, percentile)


def pad_wav(waveform, target_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    if waveform_length == target_length:
        return waveform

    # Pad
    temp_wav = np.zeros((1, target_length), dtype=np.float32)
    rand_start = 0

    temp_wav[:, rand_start : rand_start + waveform_length] = waveform
    return temp_wav


def lowpass_filtering_prepare_inference(dl_output):
    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]

    cutoff_freq = (
        _locate_cutoff_freq(dl_output["stft"], percentile=0.985) / 1024
    ) * 24000
    
    # If the audio is almost empty. Give up processing
    if(cutoff_freq < 1000):
        cutoff_freq = 24000

    order = 8
    ftype = np.random.choice(["butter", "cheby1", "ellip", "bessel"])
    filtered_audio = lowpass(
        waveform.numpy().squeeze(),
        highcut=cutoff_freq,
        fs=sampling_rate,
        order=order,
        _type=ftype,
    )

    filtered_audio = torch.FloatTensor(filtered_audio.copy()).unsqueeze(0)

    if waveform.size(-1) <= filtered_audio.size(-1):
        filtered_audio = filtered_audio[..., : waveform.size(-1)]
    else:
        filtered_audio = torch.functional.pad(
            filtered_audio, (0, waveform.size(-1) - filtered_audio.size(-1))
        )

    return {"waveform_lowpass": filtered_audio}


def mel_spectrogram_train(y):
    global mel_basis, hann_window

    sampling_rate = 48000
    filter_length = 2048
    hop_length = 480
    win_length = 2048
    n_mel = 256
    mel_fmin = 20
    mel_fmax = 24000

    if 24000 not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis[str(mel_fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2)),
        mode="reflect",
    )

    y = y.squeeze(1)

    stft_spec = torch.stft(
        y,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[str(y.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    stft_spec = torch.abs(stft_spec)

    mel = spectral_normalize_torch(
        torch.matmul(mel_basis[str(mel_fmax) + "_" + str(y.device)], stft_spec)
    )

    return mel[0], stft_spec[0]


def pad_spec(log_mel_spec, target_frame):
    n_frames = log_mel_spec.shape[0]
    p = target_frame - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        log_mel_spec = m(log_mel_spec)
    elif p < 0:
        log_mel_spec = log_mel_spec[0:target_frame, :]

    if log_mel_spec.size(-1) % 2 != 0:
        log_mel_spec = log_mel_spec[..., :-1]

    return log_mel_spec


def wav_feature_extraction(waveform, target_frame):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft = mel_spectrogram_train(waveform.unsqueeze(0))

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = pad_spec(log_mel_spec, target_frame), pad_spec(
        stft, target_frame
    )
    return log_mel_spec, stft


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename):
    waveform, sr = torchaudio.load(filename)
    duration = waveform.size(-1) / sr

    if(duration > 10.24):
        print("\033[93m {}\033[00m" .format("Warning: audio is longer than 10.24 seconds, may degrade the model performance. It's recommand to truncate your audio to 5.12 seconds before input to AudioSR to get the best performance."))

    if(duration % 5.12 != 0):
        pad_duration = duration + (5.12 - duration % 5.12)
    else:
        pad_duration = duration

    target_frame = int(pad_duration * 100)

    waveform = torchaudio.functional.resample(waveform, sr, 48000)

    waveform = waveform.numpy()[0, ...]

    waveform = normalize_wav(
        waveform
    )  # TODO rescaling the waveform will cause low LSD score

    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, target_length=int(48000 * pad_duration))
    return waveform, target_frame, pad_duration

def read_audio_file(filename):
    waveform, target_frame, duration = read_wav_file(filename)
    log_mel_spec, stft = wav_feature_extraction(waveform, target_frame)
    return log_mel_spec, stft, waveform, duration, target_frame


def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip("\n")
            result.append(each)
    return result


def get_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth


def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # ZLUDA (AMD GPU): MIOpen is not installed, so cuDNN (ZLUDA->MIOpen) cannot
    # run. Disable cuDNN to make conv use the ATen im2col+rocBLAS path instead.
    # (Also handled by zluda_patch.py; kept here as a belt-and-suspenders.)
    _is_z = bool(os.environ.get("ZLUDA")) or (torch.cuda.is_available() and "NVIDIA" not in torch.cuda.get_device_name(0).upper())
    if _is_z:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False



def _wav_duration_seconds(path):
    """Read duration of a WAV file using Python's stdlib `wave`.

    Avoids an external ffprobe/ffmpeg dependency. The shipped
    `ffprobe.exe` first in PATH on some machines is built with
    `--disable-everything` (no audio demuxer), which makes it unable to
    probe WAV files and breaks the original `strip_silence` flow.
    """
    with contextlib.closing(wave.open(path, "rb")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def strip_silence(orignal_path, input_path, output_path):
    """Trim the super-resolved temp file to the original input's duration.

    The previous implementation called `ffprobe` + `ffmpeg` to get the
    duration and remux the file. Both have to be present and the
    `ffprobe` first in PATH must understand the WAV container — which
    isn't always the case on Windows. We now use the stdlib `wave` module
    to read the original duration and `soundfile` to read/write the
    truncated samples. This keeps `save_wave` self-contained.
    """
    # Read original input duration (in seconds) via Python's wave module.
    duration_sec = _wav_duration_seconds(orignal_path)
    if duration_sec <= 0:
        duration_sec = _wav_duration_seconds(input_path)

    # Read the super-resolved audio (already a WAV written by sf.write)
    data, sr = sf.read(input_path, always_2d=True)
    n_samples = data.shape[0]
    max_samples = int(round(duration_sec * sr))
    if 0 < max_samples < n_samples:
        data = data[:max_samples]

    # Write the trimmed audio to the final output path and remove the
    # intermediate temp file.
    sf.write(output_path, data, samplerate=sr)
    try:
        os.remove(input_path)
    except OSError:
        pass



def save_wave(waveform, inputpath, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if waveform.shape[0] > 1:
            fname = "%s_%s.wav" % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            )
        else:
            fname = (
                "%s.wav" % os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0]
            )
            # Avoid the file name too long to be saved
            if len(fname) > 255:
                fname = f"{hex(hash(fname))}.wav"

        save_path = os.path.join(savepath, fname)
        temp_path = os.path.join(tempfile.gettempdir(), fname)
        print("\033[98m {}\033[00m" .format("Don't forget to try different seeds by setting --seed <int> so that AudioSR can have optimal performance on your hardware."))
        print("Save audio to %s." % save_path)
        sf.write(temp_path, waveform[i, 0], samplerate=samplerate)
        strip_silence(inputpath, temp_path, save_path)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    except:
        import ipdb

        ipdb.set_trace()


def default_audioldm_config(model_name="basic"):
    basic_config = get_basic_config()
    return basic_config


class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_checkpoint(checkpoint_name="basic"):
    # Use locally downloaded model files directly to avoid relying on the
    # HuggingFace hub cache layout (snapshots/refs/blobs). The checkpoint
    # files are expected to live flat under the model cache directory.
    _LOCAL_HUB_DIR = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )

    if checkpoint_name == "basic":
        local_file = os.path.join(
            _LOCAL_HUB_DIR, "models--haoheliu--audiosr_basic", "pytorch_model.bin"
        )
        model_id = "haoheliu/audiosr_basic"
    elif checkpoint_name == "speech":
        local_file = os.path.join(
            _LOCAL_HUB_DIR, "models--haoheliu--audiosr_speech", "pytorch_model.bin"
        )
        model_id = "haoheliu/audiosr_speech"
    else:
        raise ValueError("Invalid Model Name %s" % checkpoint_name)

    if os.path.exists(local_file):
        print(f"Using local checkpoint: {local_file}")
        return local_file

    # Fallback to the hub (only works when online or properly cached).
    return hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")


def get_basic_config():
    return {
        "preprocessing": {
            "audio": {
                "sampling_rate": 48000,
                "max_wav_value": 32768,
                "duration": 10.24,
            },
            "stft": {"filter_length": 2048, "hop_length": 480, "win_length": 2048},
            "mel": {"n_mel_channels": 256, "mel_fmin": 20, "mel_fmax": 24000},
        },
        "augmentation": {"mixup": 0.5},
        "model": {
            "target": "audiosr.latent_diffusion.models.ddpm.LatentDiffusion",
            "params": {
                "first_stage_config": {
                    "base_learning_rate": 0.000008,
                    "target": "audiosr.latent_encoder.autoencoder.AutoencoderKL",
                    "params": {
                        "reload_from_ckpt": "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/vae/vae_48k_256/ds_8_kl_1/checkpoints/ckpt-checkpoint-484999.ckpt",
                        "sampling_rate": 48000,
                        "batchsize": 4,
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 16,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "mel_bins": 256,
                            "z_channels": 16,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4, 8],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.1,
                        },
                    },
                },
                "base_learning_rate": 0.0001,
                "warmup_steps": 5000,
                "optimize_ddpm_parameter": True,
                "sampling_rate": 48000,
                "batchsize": 16,
                "beta_schedule": "cosine",
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "unconditional_prob_cfg": 0.1,
                "parameterization": "v",
                "first_stage_key": "fbank",
                "latent_t_size": 128,
                "latent_f_size": 32,
                "channels": 16,
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audiosr.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "in_channels": 32,
                        "out_channels": 16,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "extra_sa_layer": True,
                        "use_spatial_transformer": True,
                        "transformer_depth": 1,
                    },
                },
                "evaluation_params": {
                    "unconditional_guidance_scale": 3.5,
                    "ddim_sampling_steps": 200,
                    "n_candidates_per_samples": 1,
                },
                "cond_stage_config": {
                    "concat_lowpass_cond": {
                        "cond_stage_key": "lowpass_mel",
                        "conditioning_key": "concat",
                        "target": "audiosr.latent_diffusion.modules.encoders.modules.VAEFeatureExtract",
                        "params": {
                            "first_stage_config": {
                                "base_learning_rate": 0.000008,
                                "target": "audiosr.latent_encoder.autoencoder.AutoencoderKL",
                                "params": {
                                    "sampling_rate": 48000,
                                    "batchsize": 4,
                                    "monitor": "val/rec_loss",
                                    "image_key": "fbank",
                                    "subband": 1,
                                    "embed_dim": 16,
                                    "time_shuffle": 1,
                                    "ddconfig": {
                                        "double_z": True,
                                        "mel_bins": 256,
                                        "z_channels": 16,
                                        "resolution": 256,
                                        "downsample_time": False,
                                        "in_channels": 1,
                                        "out_ch": 1,
                                        "ch": 128,
                                        "ch_mult": [1, 2, 4, 8],
                                        "num_res_blocks": 2,
                                        "attn_resolutions": [],
                                        "dropout": 0.1,
                                    },
                                },
                            }
                        },
                    }
                },
            },
        },
    }