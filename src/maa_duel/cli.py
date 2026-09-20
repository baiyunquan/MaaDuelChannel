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
) -> None:
    from maa_duel.assets import sync_assets

    sync_assets(catalog, workspace)


@app.command()
def scan(
    input_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    workspace: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    from maa_duel.video import scan_videos

    scan_videos(input_dir, workspace)


@app.command()
def synth(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    from maa_duel.synthetic import generate_synthetic_dataset

    generate_synthetic_dataset(workspace)


@app.command("train-vision")
def train_vision(
    task: Annotated[str, typer.Argument(help="roster or battlefield")],
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    all_samples: Annotated[bool, typer.Option("--all", help="Use every available sample.")] = False,
) -> None:
    from maa_duel.training.vision import train_vision_model

    train_vision_model(task, workspace, all_samples=all_samples)


@app.command()
def extract(
    input_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    workspace: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    from maa_duel.extraction import extract_rounds

    extract_rounds(input_dir, workspace)


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

    build_predictor_dataset(workspace)


@app.command("train-predictor")
def train_predictor(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    all_samples: Annotated[bool, typer.Option("--all", help="Use every accepted sample.")] = False,
) -> None:
    from maa_duel.training.predictor import train_predictor_model

    train_predictor_model(workspace, all_samples=all_samples)


@app.command()
def report(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    from maa_duel.reporting import write_report

    write_report(workspace)
