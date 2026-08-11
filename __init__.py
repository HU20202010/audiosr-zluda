# --- torchaudio.load shim -------------------------------------------------
# torchaudio >= 2.9 forces the torchcodec backend for `load`, which requires
# FFmpeg on the system. On a CPU-only Windows box without FFmpeg this raises
# ImportError. Fall back to the already-installed `soundfile` backend so audio
# loading works out of the box. The shim returns the same (tensor[ch,time], sr)
# contract as torchaudio.load. If torchcodec is available, the original loader
# is left untouched.
import torchaudio as _torchaudio

try:
    import torchcodec  # noqa: F401
    _torchcodec_available = True
except Exception:
    _torchcodec_available = False

if not _torchcodec_available and not getattr(_torchaudio, "_audiosr_sf_shim", False):
    import soundfile as _soundfile
    import torch as _torch

    def _audiosr_sf_load(uri, *args, **kwargs):
        # soundfile returns [time, channels] float numpy; torchaudio.load returns
        # [channels, time] float32 tensor.
        data, sr = _soundfile.read(uri, always_2d=True, dtype="float32")
        waveform = _torch.from_numpy(data).t().contiguous()
        return waveform, sr

    _torchaudio.load = _audiosr_sf_load
    _torchaudio._audiosr_sf_shim = True
# --- end shim -------------------------------------------------------------

# --- ZLUDA (AMD GPU) cuBLASLt workaround ----------------------------------
# hipBLASlt has no RDNA2 (gfx1030/1031) kernels on Windows, so cuBLASLt ops
# (addmm / F.linear on 2-D input) fail. Redirect them to the cuBLAS/rocBLAS
# path (mm + bias). No-op if zluda_patch is missing or on non-CUDA tensors.
try:
    import zluda_patch
    zluda_patch.apply()
    # region debug-point init-post-patch
    import torch as _torch, os as _os
    print(f"[DBG-init] post-patch cudnn.enabled={_torch.backends.cudnn.enabled} benchmark={_torch.backends.cudnn.benchmark} deterministic={_torch.backends.cudnn.deterministic}", flush=True)
    # endregion
except Exception as _e:  # pragma: no cover
    import os as _os
    if _os.environ.get("ZLUDA_DEBUG"):
        print("ZLUDA patch skipped:", _e)
# --- end ZLUDA workaround ------------------------------------------------

from .utils import seed_everything, save_wave, get_time, get_duration, read_list
from .pipeline import *
