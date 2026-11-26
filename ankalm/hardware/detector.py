"""Comprehensive hardware discovery, backend verification, and memory estimation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
import torch.nn as nn

from ankalm.core.config import ModelProfileConfig, TrainingProfileConfig, PREDEFINED_MODEL_PROFILES
from ankalm.core.constants import PROJECT_ROOT


@dataclass
class GPUDeviceInfo:
    name: str
    vram_gb: float
    driver_version: str
    vendor: str
    torch_accessible: bool
    backend_type: str  # cuda, rocm, mps, wmi_detected_only


@dataclass
class BackendVerificationResult:
    backend: str
    device_string: str
    verified: bool
    tensor_allocation_ok: bool
    matmul_ok: bool
    forward_pass_ok: bool
    backward_pass_ok: bool
    optimizer_step_ok: bool
    parameter_updated: bool
    fp32_gflops: float
    bf16_gflops: float
    recommended_precision: str
    peak_memory_mb: float
    error: Optional[str] = None


@dataclass
class MemoryEstimate:
    model_name: str
    parameters: int
    precision: str
    weights_mb: float
    gradients_mb: float
    optimizer_state_mb: float
    activations_mb: float
    overhead_mb: float
    total_estimated_mb: float
    total_estimated_gb: float
    fits_in_ram: bool
    fits_in_vram: bool


@dataclass
class HardwareReport:
    timestamp: str
    os_name: str
    os_version: str
    python_version: str
    pytorch_version: str
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    cuda_available: bool
    rocm_available: bool
    mps_available: bool
    mkldnn_available: bool
    openmp_available: bool
    compilers: Dict[str, Optional[str]]
    gpus: List[GPUDeviceInfo]
    selected_backend: BackendVerificationResult
    hardware_profile: str  # cpu, low_vram, standard_gpu, high_memory, custom
    recommended_model_profile: str
    recommended_config_path: str
    memory_estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                timeout=5,
            ).strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _detect_windows_gpus_via_registry_and_wmi() -> List[GPUDeviceInfo]:
    gpus: List[GPUDeviceInfo] = []
    if platform.system() != "Windows":
        return gpus
    try:
        cmd = (
            "$reg = Get-ItemProperty -Path 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
            "-ErrorAction SilentlyContinue | Where-Object { $_.DriverDesc -and $_.DriverDesc -notmatch 'Remote' }; "
            "$res = @(); "
            "foreach ($r in $reg) { "
            "  $vram = 0; if ($r.'HardwareInformation.qwMemorySize') { $vram = [math]::Round($r.'HardwareInformation.qwMemorySize'/1GB, 2) }; "
            "  $res += [PSCustomObject]@{ Name=$r.DriverDesc; VRAM_GB=$vram; Driver=$r.DriverVersion } "
            "}; $res | ConvertTo-Json -Compress"
        )
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            timeout=6,
        ).strip()
        if out:
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                name = str(item.get("Name", "Unknown GPU"))
                vram = float(item.get("VRAM_GB") or 0.0)
                driver = str(item.get("Driver") or "Unknown")
                vendor = "AMD" if "AMD" in name.upper() or "RADEON" in name.upper() else (
                    "NVIDIA" if "NVIDIA" in name.upper() or "GEFORCE" in name.upper() or "RTX" in name.upper() else "Other"
                )
                gpus.append(
                    GPUDeviceInfo(
                        name=name,
                        vram_gb=vram,
                        driver_version=driver,
                        vendor=vendor,
                        torch_accessible=False,
                        backend_type="wmi_detected_only",
                    )
                )
    except Exception:
        pass
    return gpus


def _detect_compilers() -> Dict[str, Optional[str]]:
    compilers: Dict[str, Optional[str]] = {}
    for tool in ("cargo", "rustc", "gcc", "g++", "clang", "cl"):
        path = shutil.which(tool)
        compilers[tool] = path
    return compilers


def verify_and_benchmark_backend(device_str: str = "cpu") -> BackendVerificationResult:
    """Run real tensor allocation, matmul, forward, backward, and optimizer verification."""
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)
    try:
        device = torch.device(device_str)
        if device.type == "cpu":
            phys_cores = psutil.cpu_count(logical=False) or 4
            torch.set_num_threads(min(phys_cores, 16))

        # 1. Tensor allocation
        a = torch.randn(512, 512, device=device, dtype=torch.float32)
        b = torch.randn(512, 512, device=device, dtype=torch.float32)
        alloc_ok = a.shape == (512, 512)

        # 2. Matrix multiplication & FP32 GFLOPS
        _ = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        iters = 35
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_fp32 = max(time.perf_counter() - t0, 1e-6)
        fp32_gflops = round((2 * (512 ** 3) * iters) / dt_fp32 / 1e9, 2)
        matmul_ok = bool(torch.isfinite(c).all().item())

        # BF16 benchmark
        bf16_gflops = 0.0
        try:
            a_bf16 = a.to(torch.bfloat16)
            b_bf16 = b.to(torch.bfloat16)
            _ = a_bf16 @ b_bf16
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(iters):
                _ = a_bf16 @ b_bf16
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_bf16 = max(time.perf_counter() - t1, 1e-6)
            bf16_gflops = round((2 * (512 ** 3) * iters) / dt_bf16 / 1e9, 2)
        except Exception:
            bf16_gflops = 0.0

        # 3. Forward pass, backward pass, and optimizer update
        torch.manual_seed(42)
        mini_layer = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 32, bias=False),
        ).to(device)
        w_before = mini_layer[0].weight.detach().clone()
        opt = torch.optim.AdamW(mini_layer.parameters(), lr=1e-2)
        x = torch.randn(8, 16, 64, device=device)
        targets = torch.randint(0, 32, (8, 16), device=device)

        logits = mini_layer(x)
        forward_ok = logits.shape == (8, 16, 32)
        loss = nn.functional.cross_entropy(logits.view(-1, 32), targets.view(-1))
        loss.backward()
        backward_ok = mini_layer[0].weight.grad is not None and bool(torch.isfinite(mini_layer[0].weight.grad).all().item())
        opt.step()
        opt.zero_grad()
        w_after = mini_layer[0].weight.detach()
        param_changed = not torch.allclose(w_before, w_after)
        opt_ok = param_changed

        mem_after = proc.memory_info().rss / (1024 * 1024)
        peak_mem = max(mem_after, mem_before)
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / (1024 * 1024))

        rec_precision = "bf16" if bf16_gflops > fp32_gflops * 1.15 else "fp32"
        return BackendVerificationResult(
            backend="cuda" if (device.type == "cuda" and not getattr(torch.version, "hip", None)) else (
                "rocm" if (device.type == "cuda" and getattr(torch.version, "hip", None)) else device.type
            ),
            device_string=device_str,
            verified=alloc_ok and matmul_ok and forward_ok and backward_ok and opt_ok,
            tensor_allocation_ok=alloc_ok,
            matmul_ok=matmul_ok,
            forward_pass_ok=forward_ok,
            backward_pass_ok=backward_ok,
            optimizer_step_ok=opt_ok,
            parameter_updated=param_changed,
            fp32_gflops=fp32_gflops,
            bf16_gflops=bf16_gflops,
            recommended_precision=rec_precision,
            peak_memory_mb=round(peak_mem, 2),
            error=None,
        )
    except Exception as exc:
        return BackendVerificationResult(
            backend=device_str,
            device_string=device_str,
            verified=False,
            tensor_allocation_ok=False,
            matmul_ok=False,
            forward_pass_ok=False,
            backward_pass_ok=False,
            optimizer_step_ok=False,
            parameter_updated=False,
            fp32_gflops=0.0,
            bf16_gflops=0.0,
            recommended_precision="fp32",
            peak_memory_mb=round(mem_before, 2),
            error=str(exc),
        )


def estimate_training_memory(
    model_cfg: ModelProfileConfig,
    train_cfg: TrainingProfileConfig,
    available_ram_gb: float = 16.0,
    available_vram_gb: float = 0.0,
) -> MemoryEstimate:
    """Estimate memory in MB/GB including weights, gradients, AdamW states, and activations."""
    params = model_cfg.estimate_parameters()
    bytes_per_param = 2 if train_cfg.precision in ("fp16", "bf16") else 4
    # Master weights in FP32 + working precision copy if mixed precision
    weights_bytes = params * 4
    gradients_bytes = params * 4
    # AdamW maintains fp32 exp_avg and exp_avg_sq (8 bytes per parameter)
    optimizer_bytes = params * 8

    # Activations per micro-batch: batch_size * seq_len * n_layers * d_model * factor
    b = train_cfg.batch_size
    s = model_cfg.max_seq_len
    l = model_cfg.n_layers
    d = model_cfg.d_model
    v = model_cfg.vocab_size
    layer_act_bytes = b * s * l * (12 * d) * bytes_per_param
    if train_cfg.activation_checkpointing:
        layer_act_bytes = int(layer_act_bytes * 0.35)
    logits_act_bytes = b * s * v * 4  # cross-entropy logits in fp32
    activations_bytes = layer_act_bytes + logits_act_bytes

    weights_mb = round(weights_bytes / (1024 * 1024), 2)
    gradients_mb = round(gradients_bytes / (1024 * 1024), 2)
    optimizer_mb = round(optimizer_bytes / (1024 * 1024), 2)
    activations_mb = round(activations_bytes / (1024 * 1024), 2)
    overhead_mb = 256.0  # PyTorch runtime & allocator buffers
    total_mb = round(weights_mb + gradients_mb + optimizer_mb + activations_mb + overhead_mb, 2)
    total_gb = round(total_mb / 1024.0, 3)

    return MemoryEstimate(
        model_name=model_cfg.name,
        parameters=params,
        precision=train_cfg.precision,
        weights_mb=weights_mb,
        gradients_mb=gradients_mb,
        optimizer_state_mb=optimizer_mb,
        activations_mb=activations_mb,
        overhead_mb=overhead_mb,
        total_estimated_mb=total_mb,
        total_estimated_gb=total_gb,
        fits_in_ram=total_gb < (available_ram_gb * 0.8),
        fits_in_vram=(available_vram_gb > 0 and total_gb < (available_vram_gb * 0.85)),
    )


def discover_hardware() -> HardwareReport:
    """Inspect actual computer hardware, verify accelerators, and select optimal profile."""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT.anchor or PROJECT_ROOT))
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    avail_ram_gb = round(vm.available / (1024 ** 3), 2)
    disk_total_gb = round(disk.total / (1024 ** 3), 2)
    disk_used_gb = round(disk.used / (1024 ** 3), 2)
    disk_free_gb = round(disk.free / (1024 ** 3), 2)

    cuda_avail = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
    rocm_avail = torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mkldnn_avail = hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available()
    openmp_avail = hasattr(torch.backends, "openmp") and torch.backends.openmp.is_available()

    gpus: List[GPUDeviceInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
            is_rocm = getattr(torch.version, "hip", None) is not None
            gpus.append(
                GPUDeviceInfo(
                    name=props.name,
                    vram_gb=vram_gb,
                    driver_version=str(getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", "unknown")),
                    vendor="AMD" if is_rocm else "NVIDIA",
                    torch_accessible=True,
                    backend_type="rocm" if is_rocm else "cuda",
                )
            )
    else:
        gpus.extend(_detect_windows_gpus_via_registry_and_wmi())

    # Verify candidate backends in order of priority: CUDA/ROCm -> MPS -> CPU
    selected: Optional[BackendVerificationResult] = None
    if torch.cuda.is_available():
        res = verify_and_benchmark_backend("cuda:0")
        if res.verified:
            selected = res
    elif mps_avail:
        res = verify_and_benchmark_backend("mps")
        if res.verified:
            selected = res

    if selected is None:
        selected = verify_and_benchmark_backend("cpu")

    # Classify hardware profile
    accessible_vram = max((g.vram_gb for g in gpus if g.torch_accessible), default=0.0)
    if selected.device_string.startswith("cuda"):
        if accessible_vram >= 14.0:
            hw_profile = "high_memory"
            rec_model = "small"
            rec_cfg = "configs/standard.yaml"
        elif accessible_vram >= 7.0:
            hw_profile = "low_vram"
            rec_model = "small"
            rec_cfg = "configs/low_vram.yaml"
        else:
            hw_profile = "low_vram"
            rec_model = "tiny"
            rec_cfg = "configs/tiny.yaml"
    else:
        # High-performance CPU mode (e.g. AMD Ryzen 9 9900X3D with 31 GB RAM & AVX-512 BF16)
        hw_profile = "cpu"
        if total_ram_gb >= 16.0 and selected.bf16_gflops >= 500.0:
            rec_model = "small"
            rec_cfg = "configs/small.yaml"
        else:
            rec_model = "tiny"
            rec_cfg = "configs/tiny.yaml"

    default_train_cfg = TrainingProfileConfig(precision=selected.recommended_precision)
    estimates: Dict[str, Dict[str, Any]] = {}
    for pname, pcfg in PREDEFINED_MODEL_PROFILES.items():
        est = estimate_training_memory(pcfg, default_train_cfg, avail_ram_gb, accessible_vram)
        estimates[pname] = asdict(est)

    return HardwareReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        cpu_name=_detect_cpu_name(),
        cpu_physical_cores=psutil.cpu_count(logical=False) or 1,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
        total_ram_gb=total_ram_gb,
        available_ram_gb=avail_ram_gb,
        disk_total_gb=disk_total_gb,
        disk_used_gb=disk_used_gb,
        disk_free_gb=disk_free_gb,
        cuda_available=cuda_avail,
        rocm_available=rocm_avail,
        mps_available=mps_avail,
        mkldnn_available=mkldnn_avail,
        openmp_available=openmp_avail,
        compilers=_detect_compilers(),
        gpus=gpus,
        selected_backend=selected,
        hardware_profile=hw_profile,
        recommended_model_profile=rec_model,
        recommended_config_path=rec_cfg,
        memory_estimates=estimates,
    )


def save_hardware_report(report: HardwareReport, out_path: Optional[Path] = None) -> Path:
    target = out_path or (PROJECT_ROOT / "logs" / "hardware_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return target
