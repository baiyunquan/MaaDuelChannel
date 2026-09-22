from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Duel Channel dataset and training pipeline.")
assets_app = typer.Typer(no_args_is_help=True, help="Manage traceable enemy assets.")
annotate_app = typer.Typer(no_args_is_help=True, help="Exchange annotations with Ultralytics Platform.")
app.add_typer(assets_app, name="assets")
app.add_typer(annotate_app, name="annotate")


@assets_app.command("sync")
def assets_sync(
    catalog: Annotated[Path, typer.Option(help="Enemy catalog CSV or JSON.")],
    workspace: Annotated[Path, typer.Option(help="External workspace directory.")],
    background_dir: Annotated[Path | None, typer.Option(help="Directory containing clean arena backgrounds.")] = None,
) -> None:
    from maa_duel.assets import sync_assets

    manifest = sync_assets(catalog, workspace, background_dir=background_dir)
    typer.echo(f"Synced {len(manifest.enemies)} enemies to {workspace / 'assets'}")


@assets_app.command("fetch-prts")
def assets_fetch_prts(
    catalog: Annotated[Path, typer.Option(help="CSV or JSON with id, name, and exact PRTS original_name.")],
    workspace: Annotated[Path, typer.Option(help="External workspace directory.")],
    force: Annotated[bool, typer.Option(help="Redownload files even when the source URL is unchanged.")] = False,
    request_interval: Annotated[
        float, typer.Option(min=0.0, max=10.0, help="Delay between PRTS requests in seconds.")
    ] = 0.25,
) -> None:
    from maa_duel.prts_assets import sync_prts_assets

    result = sync_prts_assets(catalog, workspace, force=force, request_interval=request_interval)
    typer.echo(
        f"Fetched {result.portrait_count}/{len(result.manifest.enemies)} thumbnails and "
        f"{result.spine_package_count}/{len(result.manifest.enemies)} battlefield Spine packages "
        f"({result.spine_variant_count} battle variants) to {workspace / 'assets'}"
    )
    if result.missing_portrait_ids:
        typer.echo(f"Missing thumbnails for enemy IDs: {', '.join(map(str, result.missing_portrait_ids))}", err=True)
    if result.missing_spine_ids:
        typer.echo(f"Missing Spine packages for enemy IDs: {', '.join(map(str, result.missing_spine_ids))}", err=True)
    for error in result.errors:
        typer.echo(error, err=True)
    if result.missing_portrait_ids or result.missing_spine_ids or result.errors:
        raise typer.Exit(code=1)


@assets_app.command("fetch-prts-combat")
def assets_fetch_prts_combat(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False, help="External workspace directory.")],
    request_interval: Annotated[
        float, typer.Option(min=0.0, max=10.0, help="Delay between PRTS requests in seconds.")
    ] = 0.25,
) -> None:
    from maa_duel.prts_combat import sync_prts_combat_knowledge

    result = sync_prts_combat_knowledge(workspace, request_interval=request_interval)
    typer.echo(
        f"Fetched {result.profile_count} VS-2 combat profiles; "
        f"mapped {result.mapped_profile_count} to model enemy IDs; sha256={result.knowledge_sha256}"
    )
    typer.echo(f"Combat knowledge: {result.path}")
    if result.missing_page_names:
        typer.echo(
            f"Stage stats retained; missing enemy pages: {', '.join(result.missing_page_names)}",
            err=True,
        )


@app.command()
def scan(
    input_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    workspace: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    from maa_duel.video import scan_videos

    records = scan_videos(input_dir, workspace)
    typer.echo(f"Scanned {len(records)} videos")


@app.command()
def synth(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    portrait_variants: Annotated[int, typer.Option(min=1)] = 40,
    detection_images: Annotated[int, typer.Option(min=0)] = 1000,
    seed: Annotated[int, typer.Option()] = 20260920,
    empty_slot_image: Annotated[Path | None, typer.Option(help="Optional path to empty slot reference image.")] = None,
) -> None:
    from maa_duel.synthetic import generate_synthetic_dataset

    result = generate_synthetic_dataset(
        workspace,
        portrait_variants=portrait_variants,
        detection_images=detection_images,
        seed=seed,
        empty_slot_image=empty_slot_image,
    )
    empty_msg = f" (including {result.empty_slot_images} empty slot samples)" if result.empty_slot_images else ""
    typer.echo(
        f"Generated {result.portrait_images} portrait{empty_msg} and {result.detection_images} battlefield images"
    )


@app.command("train-vision")
def train_vision(
    task: Annotated[str, typer.Argument(help="roster, battlefield, or ocr")],
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    all_samples: Annotated[bool, typer.Option("--all", help="Use every available sample.")] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    image_size: Annotated[int | None, typer.Option(min=64)] = None,
    device: Annotated[str, typer.Option(help="Ultralytics device, for example 0 or cpu.")] = "0",
    base_model: Annotated[str | None, typer.Option(help="Override the default YOLO checkpoint.")] = None,
    dataset_version: Annotated[
        str | None, typer.Option(help="Reviewed dataset version under workspace/datasets.")
    ] = None,
    batch: Annotated[float, typer.Option(help="Batch size, -1 for auto, or a 0-1 GPU memory fraction.")] = -1,
    workers: Annotated[int, typer.Option(min=0, help="DataLoader worker processes.")] = 4,
    cache: Annotated[str, typer.Option(help="Ultralytics cache mode: disk, ram, or none.")] = "ram",
    amp: Annotated[bool, typer.Option("--amp/--no-amp", help="Use automatic mixed precision.")] = True,
    deterministic: Annotated[bool, typer.Option("--deterministic/--no-deterministic")] = True,
    seed: Annotated[int, typer.Option()] = 20260920,
    patience: Annotated[int, typer.Option(min=0)] = 50,
) -> None:
    from maa_duel.training.vision import train_vision_model

    if cache.casefold() not in {"disk", "ram", "none"}:
        raise typer.BadParameter("--cache must be disk, ram, or none")
    if batch > 1 and not batch.is_integer():
        raise typer.BadParameter("--batch values above 1 must be whole numbers")
    path = train_vision_model(
        task,
        workspace,
        all_samples=all_samples,
        epochs=epochs,
        image_size=image_size,
        device=device,
        base_model=base_model,
        dataset_version=dataset_version,
        batch=int(batch) if batch >= 1 and batch.is_integer() else batch,
        workers=workers,
        cache=False if cache.casefold() == "none" else cache.casefold(),
        amp=amp,
        deterministic=deterministic,
        seed=seed,
        patience=patience,
    )
    typer.echo(f"Vision checkpoint: {path}")


@app.command()
def extract(
    input_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    workspace: Annotated[Path, typer.Option(file_okay=False)],
    device: Annotated[str, typer.Option(help="Ultralytics inference device, for example 0 or cpu.")] = "0",
    half: Annotated[bool, typer.Option("--half/--no-half", help="Use FP16 inference on supported GPUs.")] = True,
    batch_size: Annotated[int, typer.Option(min=1, help="YOLO inference batch size.")] = 32,
    scan_fps: Annotated[
        float | None, typer.Option(min=1.0, max=30.0, help="Sampling frame rate for phase analysis.")
    ] = None,
    max_videos: Annotated[int | None, typer.Option(min=1, help="Limit number of videos to extract.")] = None,
    force: Annotated[
        bool, typer.Option("--force/--no-force", help="Force re-extraction of all videos from scratch.")
    ] = False,
    workers: Annotated[int, typer.Option(min=1, max=16, help="Number of parallel extraction workers.")] = 8,
) -> None:
    from maa_duel.extraction import extract_rounds

    samples = extract_rounds(
        input_dir,
        workspace,
        device=device,
        half=half,
        batch_size=batch_size,
        scan_fps=scan_fps,
        max_videos=max_videos,
        force=force,
        workers=workers,
    )
    typer.echo(f"Extracted {len(samples)} round samples")


@app.command()
def review(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    video_dir: Annotated[
        Path | None,
        typer.Option(help="Directory containing source raw videos for timeline adjustment."),
    ] = None,
    port: Annotated[int, typer.Option(help="Local review web server port (if using --web).")] = 7860,
    platform: Annotated[
        bool,
        typer.Option("--platform", help="Export Ultralytics Platform packages instead of local offline review."),
    ] = False,
    web: Annotated[
        bool,
        typer.Option("--web", help="Use legacy Gradio web interface instead of PyQt6 desktop app."),
    ] = False,
    skip_prep_roster: Annotated[
        bool,
        typer.Option(
            "--skip-prep-roster",
            help="Skip batch prep roster review and directly open single-round workbench.",
        ),
    ] = False,
    only_prep_roster: Annotated[
        bool,
        typer.Option(
            "--only-prep-roster",
            help="Only run batch prep roster review without entering single-round workbench.",
        ),
    ] = False,
) -> None:
    if platform:
        from maa_duel.review import launch_review

        exported = launch_review(workspace)
        typer.echo(f"Platform export: {exported.directory}")
        for task, archive in exported.archives.items():
            typer.echo(f"  {task.value}: {archive} ({exported.item_counts[task]} items)")
    elif web:
        from maa_duel.review import launch_local_review

        launch_local_review(workspace, port=port)
    else:
        from maa_duel.review import launch_local_review_qt

        launch_local_review_qt(
            workspace,
            video_dir=video_dir,
            skip_prep_roster=skip_prep_roster,
            only_prep_roster=only_prep_roster,
        )


@annotate_app.command("export")
def annotate_export(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    export_id: Annotated[str | None, typer.Option(help="Stable export name; defaults to a UTC timestamp.")] = None,
) -> None:
    from maa_duel.annotations import export_platform_annotations

    exported = export_platform_annotations(workspace, export_id=export_id)
    typer.echo(f"Upload these archives at https://platform.ultralytics.com/: {exported.directory}")
    for task, archive in exported.archives.items():
        typer.echo(f"  {task.value}: {archive} ({exported.item_counts[task]} items)")


@annotate_app.command("import")
def annotate_import(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    export_directory: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    roster: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, help="Roster Platform NDJSON.")] = None,
    battlefield: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Battlefield Platform NDJSON.")
    ] = None,
    ocr: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, help="OCR Platform NDJSON.")] = None,
    version: Annotated[str | None, typer.Option(help="Local immutable annotation version.")] = None,
    parent_version: Annotated[str | None, typer.Option()] = None,
) -> None:
    from maa_duel.annotations import import_platform_annotations
    from maa_duel.contracts import AnnotationTask

    exports = {
        task: path
        for task, path in (
            (AnnotationTask.ROSTER_CLASSIFICATION, roster),
            (AnnotationTask.BATTLEFIELD_DETECTION, battlefield),
            (AnnotationTask.OCR_CLASSIFICATION, ocr),
        )
        if path is not None
    }
    if not exports:
        raise typer.BadParameter("provide at least one of --roster, --battlefield, or --ocr")
    contract = import_platform_annotations(
        workspace,
        export_directory,
        exports,
        version=version,
        parent_version=parent_version,
    )
    typer.echo(f"Imported annotation version {contract.dataset_version}: {contract.task_counts}")


@annotate_app.command("sample")
def annotate_sample(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    annotation_version: Annotated[str, typer.Option(help="Version under annotations/versions.")],
    output_version: Annotated[str | None, typer.Option()] = None,
    maximum_samples: Annotated[int | None, typer.Option(min=1)] = None,
    hard_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    replay_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.3,
    base_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.2,
    seed: Annotated[int, typer.Option()] = 20260920,
) -> None:
    from maa_duel.contracts import SamplingPolicy
    from maa_duel.sampling import sample_training_sets

    contract = sample_training_sets(
        workspace,
        annotation_version,
        output_version=output_version,
        policy=SamplingPolicy(
            hard_fraction=hard_fraction,
            replay_fraction=replay_fraction,
            base_fraction=base_fraction,
            maximum_samples=maximum_samples,
            seed=seed,
        ),
    )
    typer.echo(
        f"Built dataset {contract.dataset_version}: tasks={contract.task_counts}, buckets={contract.bucket_counts}"
    )


@app.command("build-dataset")
def build_dataset(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    from maa_duel.dataset import build_predictor_dataset

    rows = build_predictor_dataset(workspace)
    typer.echo(f"Wrote {len(rows)} accepted predictor samples")


@app.command("train-predictor")
def train_predictor(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    all_samples: Annotated[bool, typer.Option("--all", help="Use every accepted sample.")] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    embedding_dim: Annotated[int, typer.Option(min=8)] = 128,
    heads: Annotated[int, typer.Option(min=1)] = 4,
    layers: Annotated[int, typer.Option(min=1)] = 3,
    dropout: Annotated[float, typer.Option(min=0.0, max=0.9)] = 0.1,
    learning_rate: Annotated[float, typer.Option(min=1e-8)] = 3e-4,
    device: Annotated[str | None, typer.Option(help="Torch device such as cuda or cpu.")] = None,
    seed: Annotated[int, typer.Option()] = 20260920,
    workers: Annotated[int, typer.Option(min=0)] = 4,
    amp: Annotated[bool, typer.Option("--amp/--no-amp")] = True,
    amp_dtype: Annotated[str, typer.Option(help="float16 or bfloat16.")] = "float16",
    compile_model: Annotated[bool, typer.Option("--compile/--no-compile")] = False,
    pin_memory: Annotated[bool, typer.Option("--pin-memory/--no-pin-memory")] = True,
    tf32: Annotated[bool, typer.Option("--tf32/--no-tf32")] = True,
) -> None:
    from maa_duel.training.predictor import train_predictor_model

    result = train_predictor_model(
        workspace,
        all_samples=all_samples,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        heads=heads,
        layers=layers,
        dropout=dropout,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        workers=workers,
        amp=amp,
        amp_dtype=amp_dtype,
        compile_model=compile_model,
        pin_memory=pin_memory,
        tf32=tf32,
    )
    typer.echo(f"Trained on {result['training_samples']} samples; metrics are training-only")


@app.command("predict-duel")
def predict_duel_command(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False, help="BattleState JSON.")],
    checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option(dir_okay=False, help="Optional prediction JSON output.")] = None,
    device: Annotated[str | None, typer.Option(help="Torch device such as cuda or cpu.")] = None,
    explain: Annotated[bool, typer.Option("--explain/--no-explain")] = False,
) -> None:
    from maa_duel.dataset import BattleState
    from maa_duel.training.inference import predict_duel

    state = BattleState.model_validate_json(input_path.read_text(encoding="utf-8"))
    result = predict_duel(workspace, state, checkpoint=checkpoint, device=device, explain=explain)
    payload = result.model_dump_json(indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload, nl=False)


@app.command()
def report(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    from maa_duel.reporting import write_report

    path = write_report(workspace)
    typer.echo(f"Report: {path}")
