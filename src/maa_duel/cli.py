from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Duel Channel dataset and training pipeline.")
assets_app = typer.Typer(no_args_is_help=True, help="Manage traceable enemy assets.")
app.add_typer(assets_app, name="assets")


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
) -> None:
    from maa_duel.synthetic import generate_synthetic_dataset

    result = generate_synthetic_dataset(
        workspace,
        portrait_variants=portrait_variants,
        detection_images=detection_images,
        seed=seed,
    )
    typer.echo(f"Generated {result.portrait_images} portrait and {result.detection_images} battlefield images")


@app.command("train-vision")
def train_vision(
    task: Annotated[str, typer.Argument(help="roster or battlefield")],
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    all_samples: Annotated[bool, typer.Option("--all", help="Use every available sample.")] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    image_size: Annotated[int, typer.Option(min=64)] = 640,
    device: Annotated[str, typer.Option(help="Ultralytics device, for example 0 or cpu.")] = "0",
    base_model: Annotated[str | None, typer.Option(help="Override the default YOLO checkpoint.")] = None,
) -> None:
    from maa_duel.training.vision import train_vision_model

    path = train_vision_model(
        task,
        workspace,
        all_samples=all_samples,
        epochs=epochs,
        image_size=image_size,
        device=device,
        base_model=base_model,
    )
    typer.echo(f"Vision checkpoint: {path}")


@app.command()
def extract(
    input_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    workspace: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    from maa_duel.extraction import extract_rounds

    samples = extract_rounds(input_dir, workspace)
    typer.echo(f"Extracted {len(samples)} round samples")


@app.command()
def review(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    from maa_duel.review import launch_review

    launch_review(workspace)


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
    )
    typer.echo(f"Trained on {result['training_samples']} samples; metrics are training-only")


@app.command()
def report(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    from maa_duel.reporting import write_report

    path = write_report(workspace)
    typer.echo(f"Report: {path}")
