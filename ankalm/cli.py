"""AnkaLM Command Line Interface (CLI) powered by Typer and Rich."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import webbrowser

from rich.console import Console
from rich.table import Table
import typer

from ankalm.core.config import AnkaLMConfig
from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.acquisition import acquire_all_turkish_datasets, prepare_corpus_splits, stream_raw_dataset_file
from ankalm.data.cleaning import clean_document
from ankalm.data.dedupe import DeduplicationEngine
from ankalm.data.instruction_data import build_turkish_instruction_dataset
from ankalm.data.manifest import APPROVED_LICENSES
from ankalm.evaluation.suite import run_turkish_evaluation_lab
from ankalm.hardware.detector import discover_hardware, save_hardware_report
from ankalm.inference.engine import AnkaLMInferenceEngine, GenerationConfig
from ankalm.quantization.quantizer import run_quantization_suite
from ankalm.storage.manager import StorageManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.tokenizer.trainer import train_and_evaluate_tokenizers
from ankalm.training.process_manager import TrainingProcessManager
from ankalm.training.sft_trainer import AnkaLMSFTTrainer
from ankalm.training.trainer import AnkaLMTrainer

console = Console()
app = typer.Typer(help="AnkaLM: Zero-to-Trained Independent Turkish Language Model CLI")
data_app = typer.Typer(help="Turkish Dataset Engine commands (inspect, clean, dedupe, validate, stats, sources, licenses)")
tokenizer_app = typer.Typer(help="AnkaLM Turkish Tokenizer commands")
storage_app = typer.Typer(help="AnkaLM Storage and Checkpoint management commands")

app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(storage_app, name="storage")


@app.command("doctor")
def doctor_command() -> None:
    """Inspect system hardware, verify compute backends, estimate memory, and recommend safe config."""
    report = discover_hardware()
    report_path = save_hardware_report(report)

    console.print("\n[bold cyan]=== AnkaLM Hardware & Accelerator Doctor Report ===[/bold cyan]")
    console.print(f"[bold]OS:[/bold] {report.os_name} ({report.os_version})")
    console.print(f"[bold]Python / PyTorch:[/bold] {report.python_version} / {report.pytorch_version}")
    console.print(f"[bold]CPU:[/bold] {report.cpu_name} ({report.cpu_physical_cores} physical / {report.cpu_logical_cores} logical cores)")
    console.print(f"[bold]RAM:[/bold] {report.total_ram_gb} GB Total | {report.available_ram_gb} GB Available")
    console.print(f"[bold]Disk Free:[/bold] {report.disk_free_gb} GB / {report.disk_total_gb} GB")

    if report.gpus:
        for idx, g in enumerate(report.gpus):
            console.print(
                f"[bold]Detected GPU #{idx}:[/bold] {g.name} | VRAM: {g.vram_gb} GB | Driver: {g.driver_version} | "
                f"PyTorch Accessible: {g.torch_accessible} ({g.backend_type})"
            )
    else:
        console.print("[bold]Detected GPU:[/bold] None")

    b = report.selected_backend
    console.print(
        f"[bold green]Verified Active Backend:[/bold green] {b.backend} ({b.device_string}) | "
        f"Verified={b.verified} | FP32={b.fp32_gflops} GFLOPS | BF16={b.bf16_gflops} GFLOPS | "
        f"Recommended Precision={b.recommended_precision}"
    )
    console.print(
        f"[bold yellow]Hardware Profile:[/bold yellow] {report.hardware_profile} | "
        f"Recommended Model: {report.recommended_model_profile} ({report.recommended_config_path})"
    )

    table = Table(title="AnkaLM Memory Estimates (Weights + Gradients + AdamW State + Activations)")
    table.add_column("Profile", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Precision", justify="center")
    table.add_column("Total Est. (MB)", justify="right")
    table.add_column("Fits RAM", justify="center")
    for pname, est in report.memory_estimates.items():
        table.add_row(
            est["model_name"],
            f"{est['parameters']:,}",
            est["precision"],
            f"{est['total_estimated_mb']:.1f} MB",
            "YES" if est["fits_in_ram"] else "NO",
        )
    console.print(table)
    console.print(f"Saved full hardware report to: [underline]{report_path}[/underline]\n")


# --- DATASET COMMANDS (Section 7.5) ---
def _load_or_build_manifest() -> dict:
    m_path = PROJECT_ROOT / "datasets" / "manifests" / "dataset_manifest.json"
    if not m_path.exists():
        tok_path = PROJECT_ROOT / "tokenizer" / "ankalm_tokenizer.json"
        cnt_fn = None
        if tok_path.exists():
            tok = AnkaLMTokenizer.load(PROJECT_ROOT / "tokenizer")
            cnt_fn = lambda s: len(tok.encode(s))
        prepare_corpus_splits(token_counter_fn=cnt_fn)
    with open(m_path, "r", encoding="utf-8") as f:
        return json.load(f)


@data_app.command("inspect")
def data_inspect() -> None:
    """Inspect dataset files, document counts, and sample records."""
    data = _load_or_build_manifest()
    console.print(f"[bold cyan]AnkaLM Dataset Inspection[/bold cyan] (Total Docs: {data['total_documents']}, Total Tokens: {data['total_tokens']})")
    for entry in data.get("entries", []):
        console.print(
            f"- [bold]{entry['name']}[/bold] [{entry['format']}] | Category: {entry['category']} | "
            f"Docs: {entry['document_count']} | Tokens: {entry['token_count']} | Avg Quality: {entry['average_quality_score']}"
        )


@data_app.command("clean")
def data_clean() -> None:
    """Run UTF-8 normalization, HTML/Markdown cleanup, language ID, and quality scoring."""
    tok_path = PROJECT_ROOT / "tokenizer" / "ankalm_tokenizer.json"
    cnt_fn = (lambda s: len(AnkaLMTokenizer.load().encode(s))) if tok_path.exists() else None
    report = prepare_corpus_splits(token_counter_fn=cnt_fn)
    build_turkish_instruction_dataset()
    console.print(
        f"[bold green]Dataset cleaning complete:[/bold green] {report.total_documents} valid Turkish documents "
        f"({report.total_tokens} tokens). Splits: {report.split_counts}"
    )


@data_app.command("dedupe")
def data_dedupe() -> None:
    """Run exact SHA-256 and near-duplicate shingle Jaccard filtering across raw datasets."""
    sources = acquire_all_turkish_datasets()
    deduper = DeduplicationEngine(near_dup_threshold=0.85)
    kept = 0
    for fpath, _, cat, _, _ in sources:
        for rec in stream_raw_dataset_file(fpath, cat):
            cleaned = clean_document(rec["text"])
            if not cleaned.is_valid:
                continue
            dup, _ = deduper.is_duplicate(cleaned.cleaned_text, rec["id"])
            if not dup:
                kept += 1
    console.print(
        f"[bold green]Deduplication Summary:[/bold green] Unique Kept={kept} | "
        f"Exact Duplicates Removed={deduper.exact_duplicates_removed} | "
        f"Near Duplicates Removed={deduper.near_duplicates_removed}"
    )


@data_app.command("validate")
def data_validate() -> None:
    """Validate dataset splits, UTF-8 integrity, license compliance, and zero cross-split leakage."""
    data = _load_or_build_manifest()
    zero_leak = data.get("zero_leakage_verified", False)
    all_licensed = all(e.get("license_verified", False) for e in data.get("entries", []))
    console.print(f"[bold]Zero Train/Val/Eval Leakage Verified:[/bold] {zero_leak}")
    console.print(f"[bold]All Dataset Licenses Verified:[/bold] {all_licensed}")
    console.print(f"[bold]Split Document Counts:[/bold] {data.get('split_counts')}")
    if not zero_leak or not all_licensed:
        raise typer.Exit(code=1)


@data_app.command("stats")
def data_stats() -> None:
    """Display comprehensive statistics and generate a dataset quality report."""
    data = _load_or_build_manifest()
    table = Table(title="AnkaLM Turkish Dataset Statistics & Quality Report")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Format")
    table.add_column("Category")
    table.add_column("Docs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Quality", justify="right")
    for e in data.get("entries", []):
        table.add_row(
            e["name"],
            e["format"],
            e["category"],
            str(e["document_count"]),
            f"{e['token_count']:,}",
            f"{e['average_quality_score']:.3f}",
        )
    console.print(table)
    q_report = PROJECT_ROOT / "datasets" / "manifests" / "quality_report.json"
    with open(q_report, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"Quality report written to: {q_report}")


@data_app.command("sources")
def data_sources() -> None:
    """List all dataset sources, formats, and SHA-256 checksums."""
    data = _load_or_build_manifest()
    for e in data.get("entries", []):
        console.print(f"- [bold]{e['name']}[/bold]: source='{e['source']}' | file='{e['file_path']}' | sha256={e['sha256_checksum'][:16]}...")


@data_app.command("licenses")
def data_licenses() -> None:
    """Verify and display license status for every dataset source."""
    data = _load_or_build_manifest()
    for e in data.get("entries", []):
        lic = e["license"]
        desc = APPROVED_LICENSES.get(lic, "Approved Open License")
        console.print(f"- [bold]{e['name']}[/bold]: [green]{lic}[/green] ({desc}) | Verified={e['license_verified']}")


# --- TOKENIZER COMMANDS ---
@tokenizer_app.command("train")
def tokenizer_train(vocab_size: int = typer.Option(8192, "--vocab-size")) -> None:
    """Train and benchmark BPE and Unigram Turkish tokenizers."""
    _load_or_build_manifest()
    tok, rep = train_and_evaluate_tokenizers(target_vocab_size=vocab_size)
    console.print(
        f"[bold green]Trained AnkaLM Turkish Tokenizer:[/bold green] algo={tok.algorithm} | "
        f"vocab_size={tok.vocab_size} | fingerprint={rep.vocab_fingerprint[:16]}"
    )


@tokenizer_app.command("inspect")
def tokenizer_inspect() -> None:
    """Inspect the trained Turkish tokenizer and Turkish Unicode roundtrip verification."""
    tok = AnkaLMTokenizer.load()
    sample = "Pijamalı hasta yağız şoföre çabucak güvendi: ç, Ç, ğ, Ğ, ı, I, İ, i, ö, Ö, ş, Ş, ü, Ü."
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    console.print(f"[bold]Algorithm:[/bold] {tok.algorithm} | [bold]Vocab Size:[/bold] {tok.vocab_size}")
    console.print(f"[bold]Fingerprint:[/bold] {tok.vocab_fingerprint()}")
    console.print(f"[bold]Sample Exact Roundtrip:[/bold] {decoded == sample} ({len(ids)} tokens)")


# --- TRAINING & RESUME COMMANDS ---
@app.command("train")
def train_command(
    config: str = typer.Option("configs/small.yaml", "--config", "-c", help="Path to YAML training configuration"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Checkpoint name or path to resume from (e.g. 'latest')"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Override maximum training steps"),
    continuous: bool = typer.Option(False, "--continuous", help="Run continuous training from latest valid checkpoint without restarting from random init"),
    duration_minutes: Optional[float] = typer.Option(None, "--duration-minutes", help="Optional wall-clock duration in minutes for --continuous mode"),
    sft: bool = typer.Option(False, "--sft", help="Run Turkish Instruction Tuning (SFT) after or on pretrained checkpoint"),
    sft_steps: int = typer.Option(80, "--sft-steps", help="Number of instruction tuning steps"),
) -> None:
    """Run genuine from-scratch pretraining, resume from checkpoint, run continuous mode, or run Turkish instruction tuning."""
    cfg_path = (PROJECT_ROOT / config).resolve() if not Path(config).is_absolute() else Path(config).resolve()
    ank_cfg = AnkaLMConfig.from_yaml(cfg_path)

    # Ensure dataset and tokenizer exist before training
    if not (PROJECT_ROOT / "datasets" / "processed" / "train.jsonl").exists():
        prepare_corpus_splits()
    if not (PROJECT_ROOT / "tokenizer" / "ankalm_tokenizer.json").exists():
        train_and_evaluate_tokenizers(target_vocab_size=ank_cfg.model.vocab_size)

    if sft:
        sft_trainer = AnkaLMSFTTrainer(ank_cfg)
        base_ckpt = resume or "small_final"
        res = sft_trainer.run_sft(base_checkpoint_name=base_ckpt, sft_steps=sft_steps)
        console.print("[bold green]Turkish Instruction Tuning (SFT) Complete![/bold green]")
        console.print_json(data=res.to_dict())
        return

    trainer = AnkaLMTrainer(ank_cfg, config_path_str=config)
    if continuous:
        import time as _time
        from ankalm.storage.checkpoint import CheckpointManager

        ckpt_mgr = CheckpointManager()
        slot_prefix = ank_cfg.model.profile
        resume_target = resume
        if resume_target is None:
            for cand in (f"{slot_prefix}_final", "latest", "best"):
                if (PROJECT_ROOT / "checkpoints" / cand / "metadata.json").exists():
                    resume_target = cand
                    break

        t_start = _time.time()
        max_sec = (duration_minutes * 60.0) if duration_minutes is not None else 86400.0
        chunk_steps = 25
        last_res = None
        while (_time.time() - t_start) < max_sec and not trainer.proc_manager.should_stop():
            cur_step = 0
            if resume_target is not None:
                try:
                    m, _ = ckpt_mgr.load_checkpoint(resume_target, restore_rng=False)
                    cur_step = m.global_step
                except Exception:
                    cur_step = 0
            next_target = (max_steps if max_steps and max_steps > cur_step else cur_step + chunk_steps)
            last_res = trainer.train(
                resume_from=resume_target,
                max_steps_override=next_target,
                slot_prefix=slot_prefix,
                maturity_target="moe_trained" if ank_cfg.model.use_moe else "pretrained",
            )
            resume_target = f"{slot_prefix}_final"
            if max_steps is not None and last_res.final_step >= max_steps:
                break
        if last_res is not None:
            console.print("[bold green]AnkaLM Continuous Training Cycle Complete![/bold green]")
            console.print_json(data=last_res.to_dict())
        return

    res = trainer.train(resume_from=resume, max_steps_override=max_steps)
    console.print("[bold green]AnkaLM Pretraining Run Complete![/bold green]")
    console.print_json(data=res.to_dict())


@app.command("stop-training")
def stop_training_command() -> None:
    """Safely signal any running AnkaLM training process to save a checkpoint and exit."""
    pm = TrainingProcessManager()
    pm.request_stop()
    console.print("[bold yellow]Safe stop signal sent to active AnkaLM training process.[/bold yellow]")


@app.command("evaluate")
def evaluate_command(checkpoint: str = typer.Option("latest", "--checkpoint", "-k")) -> None:
    """Run the 16-category Turkish Evaluation Lab on a trained checkpoint."""
    rep = run_turkish_evaluation_lab(checkpoint_name=checkpoint)
    console.print(
        f"[bold green]Evaluation Complete for {rep.model_name} ({rep.checkpoint_name}):[/bold green] "
        f"Held-out Eval Loss={rep.held_out_eval_loss:.4f} | Eval Perplexity={rep.held_out_perplexity:.2f} | "
        f"Benchmark Score={rep.overall_benchmark_score * 100:.1f}%"
    )


@app.command("quantize")
def quantize_command(checkpoint: str = typer.Option("latest", "--checkpoint", "-k")) -> None:
    """Benchmark and export FP32, BF16, FP16, INT8, and INT4 quantized variants."""
    results = run_quantization_suite(checkpoint_name=checkpoint)
    table = Table(title=f"AnkaLM Quantization Benchmark ({checkpoint})")
    table.add_column("Precision", style="cyan")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Compression", justify="right")
    table.add_column("Speed (tok/s)", justify="right")
    table.add_column("Val Perplexity", justify="right")
    for r in results:
        table.add_row(
            r.precision,
            f"{r.model_storage_mb:.2f}",
            f"{r.compression_ratio:.2f}x",
            f"{r.inference_tokens_per_sec:.1f}",
            f"{r.validation_perplexity:.2f}",
        )
    console.print(table)


@app.command("chat")
def chat_command(
    checkpoint: str = typer.Option("auto", "--checkpoint", "-k"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Single non-interactive prompt to generate"),
    max_tokens: int = typer.Option(80, "--max-tokens"),
) -> None:
    """Interactive terminal chat or single-prompt generation with AnkaLM's trained model."""
    target = checkpoint
    if target == "auto":
        for cand in ("small_sft", "small_final", "latest", "best", "tiny_final", "micro_final"):
            if (PROJECT_ROOT / "checkpoints" / cand / "metadata.json").exists():
                target = cand
                break
    engine = AnkaLMInferenceEngine.from_checkpoint(target, allow_random_init=False)
    gen_cfg = GenerationConfig(max_new_tokens=max_tokens, temperature=0.65, top_k=35, top_p=0.9)

    if prompt is not None:
        formatted_prompt = prompt
        if engine.metadata.maturity_level == "instruction_tuned" and "<|assistant|>" not in prompt:
            formatted_prompt = (
                "<|bos|><|system|>\nSen AnkaLM, yerel bilgisayarda sıfırdan eğitilmiş, doğru ve güvenilir bir Türkçe yapay zeka asistanısın.<|eot|>\n"
                f"<|user|>\n{prompt}<|eot|>\n<|assistant|>\n"
            )
            gen_cfg.greedy = True
        res = engine.generate(formatted_prompt, gen_cfg=gen_cfg)
        console.print(f"[bold cyan]{engine.metadata.model_name} ({engine.metadata.maturity_level}):[/bold cyan] {res.generated_text.strip()}")
        return

    console.print(
        f"[bold green]AnkaLM Yerel Türkçe Sohbet[/bold green] | Model: {engine.metadata.model_name} "
        f"({engine.metadata.checkpoint_name}, [{engine.metadata.maturity_level}])\nÇıkmak için 'exit' veya 'çıkış' yazın.\n"
    )
    while True:
        try:
            user_in = input("Siz > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_in or user_in.lower() in ("exit", "quit", "çıkış", "cikis"):
            break
        formatted = (
            "<|bos|><|system|>\nSen AnkaLM, yerel donanımda sıfırdan eğitilmiş Türkçe yapay zeka asistanısın.<|eot|>\n"
            f"<|user|>\n{user_in}<|eot|>\n<|assistant|>\n"
        )
        res = engine.generate(formatted, gen_cfg=gen_cfg)
        console.print(f"[bold cyan]AnkaLM >[/bold cyan] {res.generated_text.strip()}\n")


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8420, "--port"),
    open_browser: bool = typer.Option(False, "--open-browser"),
) -> None:
    """Launch the localhost-only AnkaLM API, Chat Application, and Training Control Center."""
    from ankalm.api.server import run_local_server

    url = f"http://{host}:{port}"
    console.print(f"[bold green]Starting AnkaLM Local AI Server at {url}[/bold green]")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    run_local_server(host=host, port=port)


@storage_app.command("inspect")
def storage_inspect() -> None:
    """Report disk space, dataset sizes, checkpoint sizes, and cache usage."""
    sm = StorageManager()
    summary = sm.inspect_storage()
    console.print(
        f"[bold cyan]AnkaLM Storage Report[/bold cyan] | Project Footprint: {summary.total_project_mb:.2f} MB | "
        f"Free Disk: {summary.disk_free_gb:.2f} GB"
    )
    for k, d in summary.directories.items():
        console.print(f"- [bold]{k}[/bold]: {d.size_mb:.2f} MB ({d.file_count} files) -> {d.path}")


@storage_app.command("clean-cache")
def storage_clean_cache() -> None:
    """Clean temporary cache files while preserving conversations and checkpoints."""
    sm = StorageManager()
    freed = sm.clean_cache(preserve_conversations=True)
    console.print(f"[bold green]Cleaned {freed / 1024:.1f} KB from cache.[/bold green]")


@app.command("distill")
def distill_command(
    teacher: str = typer.Option("small_final", "--teacher", "-t"),
    student_slot: str = typer.Option("edge_final", "--student", "-s"),
    steps: int = typer.Option(100, "--steps"),
) -> None:
    """Distill AnkaLM's own trained teacher checkpoint into the compact 2GB VRAM AnkaLM-Edge model."""
    from ankalm.training.distillation import AnkaLMDistillationTrainer

    distiller = AnkaLMDistillationTrainer()
    res = distiller.distill(teacher_checkpoint_name=teacher, student_slot_name=student_slot, steps=steps)
    console.print("[bold green]AnkaLM-Edge Knowledge Distillation Complete![/bold green]")
    console.print_json(data=res.to_dict())


@app.command("train-vision")
def train_vision_command(steps: int = typer.Option(80, "--steps")) -> None:
    """Train the independent AnkaLM-Vision encoder and multimodal projector from random initialization."""
    from ankalm.model.vision import train_ankalm_vision_from_scratch

    rep = train_ankalm_vision_from_scratch(steps=steps)
    console.print("[bold green]AnkaLM-Vision Training Complete![/bold green]")
    console.print_json(data=rep)


@app.command("benchmark-low-vram")
def benchmark_low_vram_command() -> None:
    """Benchmark 2GB Low-VRAM Edge profiles, Sparse MoE, and Speculative Decoding."""
    from ankalm.inference.speculative import benchmark_low_vram_profiles

    rep = benchmark_low_vram_profiles()
    console.print("[bold green]Low-VRAM & Speculative Decoding Benchmarks Complete![/bold green]")
    console.print_json(data=rep)


if __name__ == "__main__":
    app()
