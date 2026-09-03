"""Device selection with a hard CUDA requirement.

Nothing in this pipeline is meant to run on CPU. A 20 s clip at 512x512 pushed
through ResNet-50 is orders of magnitude slower there, so a silent CPU fallback
does not look like an error -- it looks like a hang, hours after launch. Every
entry point therefore resolves its device through here, which prints what it
picked and refuses to continue without a GPU unless CPU is explicitly requested.
"""
import torch


CPU_FALLBACK_MESSAGE = """\
CUDA is not available, and this pipeline must not run on CPU.

On this Jetson the usual cause is a missing library path. torch needs:

    export LD_LIBRARY_PATH=/mnt/pag_disk/rr_venv/syslibs:$LD_LIBRARY_PATH

Verify with:

    python -c "import torch; print(torch.cuda.is_available())"

Other things to check: another process holding the GPU (`tegrastats`), or a
torch build without CUDA support (`python -c "import torch; print(torch.version.cuda)"`).

If you really do want CPU (debugging only, expect it to be far slower), pass
--allow-cpu.\
"""


def resolve_device(allow_cpu: bool = False, require_cuda: bool = True) -> torch.device:
    """Return the device to run on, refusing CPU unless it was asked for.

    Args:
      allow_cpu: set by the caller's --allow-cpu flag; permits a CPU run.
      require_cuda: config-level switch (train.require_cuda). When False, a
        missing GPU degrades to CPU with a warning instead of raising.

    Raises:
      RuntimeError: CUDA is unavailable and CPU was not explicitly permitted.
    """
    available = torch.cuda.is_available()
    print(f"[device] torch={torch.__version__} cuda_available={available}")

    if available:
        props = torch.cuda.get_device_properties(0)
        print(f"[device] using gpu#0={props.name} cc={props.major}.{props.minor} "
              f"mem={props.total_memory / 2**30:.1f}GiB cuda={torch.version.cuda}")
        return torch.device("cuda")

    if allow_cpu or not require_cuda:
        print("[device] WARNING: CUDA unavailable; continuing on CPU because it was "
              "explicitly permitted. This will be very slow.")
        return torch.device("cpu")

    raise RuntimeError(CPU_FALLBACK_MESSAGE)


def assert_module_on_cuda(model: torch.nn.Module, device: torch.device) -> None:
    """Confirm the parameters really landed on the intended device.

    Guards against a .to(device) that silently did nothing -- cheap insurance
    given that the whole point is to never train on CPU by accident.
    """
    if device.type != "cuda":
        return
    wrong = {p.device.type for p in model.parameters()} - {"cuda"}
    if wrong:
        raise RuntimeError(
            f"Model parameters are on {sorted(wrong)} after .to({device}); expected all on cuda."
        )
    print(f"[device] verified: all model parameters on {device}")
