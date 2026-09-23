"""Deterministic OCR backend assignment for a selected set of videos."""

from __future__ import annotations

from maa_duel.video.inventory import VideoRecord

GPU_SPEED_WEIGHT = 1.25
VideoJob = tuple[int, VideoRecord]


def schedule_ocr_jobs(
    pending_records: list[VideoJob],
    *,
    workers: int,
    gpu_enabled: bool,
) -> tuple[list[VideoJob], list[VideoJob]]:
    """Reserve the longest video for CUDA, then balance estimated lane work.

    The caller applies video selection and resume filtering before this
    function. CPU lanes share one executor, so their virtual loads only decide
    which videos remain available to the dedicated GPU lane.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    ordered = sorted(
        pending_records,
        key=lambda job: (-job[1].duration, job[1].relative_path.casefold()),
    )
    if not gpu_enabled:
        return [], ordered

    loads = [0.0] * workers
    gpu_jobs: list[VideoJob] = []
    cpu_jobs: list[VideoJob] = []
    for index, job in enumerate(ordered):
        lane = 0 if index == 0 else min(range(workers), key=lambda item: (loads[item], item))
        loads[lane] += job[1].duration / (GPU_SPEED_WEIGHT if lane == 0 else 1.0)
        (gpu_jobs if lane == 0 else cpu_jobs).append(job)
    return gpu_jobs, cpu_jobs
