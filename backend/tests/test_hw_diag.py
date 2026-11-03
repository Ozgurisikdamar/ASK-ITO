"""GPU / offload diagnostics: verdicts from vendor-tool output and Ollama /api/ps."""
from app import hw


def _no_tools(cmd, timeout=10):
    return None


def test_no_gpu_tools_means_cpu_only():
    assert hw.gpus(_no_tools) == [] or hw.platform.system() == "Darwin"
    assert hw.verdict([], []) == "cpu_only"
    cpu_model = [{"name": "qwen3.5:9b-q8_0", "size": 10_000, "size_vram": 0}]
    assert hw.verdict([], hw.ollama_offload(cpu_model)) == "cpu_only"


def test_nvidia_smi_output_is_parsed():
    def fake(cmd, timeout=10):
        return "NVIDIA GeForce RTX 4070, 12282, 812, 560.94\n" if cmd[0] == "nvidia-smi" else None

    g = hw.nvidia_gpus(fake)
    assert g == [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4070",
                  "vram_total_mb": 12282, "vram_used_mb": 812, "driver": "560.94"}]


def test_offload_verdicts_from_ollama_ps():
    gpu = [{"vendor": "nvidia"}]
    full = hw.ollama_offload([{"name": "m", "size": 100, "size_vram": 100}])
    part = hw.ollama_offload([{"name": "m", "size": 100, "size_vram": 40}])
    none = hw.ollama_offload([{"name": "m", "size": 100, "size_vram": 0}])
    assert hw.verdict(gpu, full) == "gpu_full"
    assert hw.verdict(gpu, part) == "gpu_partial" and part[0]["vram_fraction"] == 0.4
    assert hw.verdict(gpu, none) == "gpu_present_unused"
    assert hw.verdict(gpu, []) == "unknown"


def test_diagnose_reports_host_and_never_raises_without_ollama():
    r = hw.diagnose(None, run=_no_tools)
    assert r["host"]["logical_cpus"] and r["ollama_loaded"] == []
    assert r["verdict"] in ("cpu_only", "unknown")
