from maa_duel.extraction_schedule import schedule_ocr_jobs
from maa_duel.video.inventory import Arena, VideoRecord


def record(name: str, duration: float) -> VideoRecord:
    return VideoRecord(
        relative_path=name,
        sha256="a" * 64,
        size_bytes=1,
        mtime_ns=1,
        width=1920,
        height=864,
        duration=duration,
        fps=30.0,
        arena=Arena.GREEN_VINE,
    )


def names(jobs: list[tuple[int, VideoRecord]]) -> list[str]:
    return [item.relative_path for _, item in jobs]


def test_longest_video_is_reserved_for_gpu_and_remaining_load_is_balanced():
    jobs = [
        (1, record("short.mp4", 10)),
        (2, record("long.mp4", 100)),
        (3, record("medium.mp4", 80)),
        (4, record("extra.mp4", 20)),
    ]

    gpu, cpu = schedule_ocr_jobs(jobs, workers=2, gpu_enabled=True)

    assert names(gpu) == ["long.mp4", "extra.mp4"]
    assert names(cpu) == ["medium.mp4", "short.mp4"]


def test_equal_durations_use_path_to_choose_gpu_video():
    jobs = [(1, record("z.mp4", 100)), (2, record("A.mp4", 100))]

    gpu, cpu = schedule_ocr_jobs(jobs, workers=2, gpu_enabled=True)

    assert names(gpu) == ["A.mp4"]
    assert names(cpu) == ["z.mp4"]


def test_single_worker_and_cpu_only_mode():
    jobs = [(1, record("short.mp4", 10)), (2, record("long.mp4", 100))]

    gpu, cpu = schedule_ocr_jobs(jobs, workers=1, gpu_enabled=True)
    assert names(gpu) == ["long.mp4", "short.mp4"]
    assert cpu == []

    gpu, cpu = schedule_ocr_jobs(jobs, workers=2, gpu_enabled=False)
    assert gpu == []
    assert names(cpu) == ["long.mp4", "short.mp4"]
