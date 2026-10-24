from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
import urllib.parse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from playwright.async_api import Browser, async_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_sprites")

# Extra battlefield sub-entities split from composite leaders
EXTRA_BATTLEFIELD_ENTITIES = [
    {
        "enemy_id": 92,
        "name": "凋零骑士",
        "original_name": "凋零骑士",
        "parent_contestant_id": 57,
        "parent_contestant_name": "并驾骑士",
    },
    {
        "enemy_id": 93,
        "name": "枣大刀",
        "original_name": "枣大刀",
        "parent_contestant_id": 58,
        "parent_contestant_name": "侠客三人行",
    },
    {
        "enemy_id": 94,
        "name": "炭长矛",
        "original_name": "炭长矛",
        "parent_contestant_id": 58,
        "parent_contestant_name": "侠客三人行",
    },
    {
        "enemy_id": 95,
        "name": "Mon2tr",
        "original_name": "Mon2tr",
        "parent_contestant_id": 74,
        "parent_contestant_name": "调停的意志",
    },
]

# Mapping of composite contestants to all their battlefield entity IDs
CONTESTANT_TO_BATTLEFIELD_ENTITIES = {
    57: [57, 92],  # 并驾骑士 -> 腐败骑士 (57), 凋零骑士 (92)
    58: [58, 93, 94],  # 侠客三人行 -> 玉双剑 (58), 枣大刀 (93), 炭长矛 (94)
    74: [74, 95],  # 调停的意志 -> 凯尔希 (74), Mon2tr (95)
}


def clean_and_crop(raw_image: np.ndarray) -> np.ndarray | None:
    """Zero out translucent noise pixels and crop tightly to non-zero alpha bounding box."""
    if raw_image is None or raw_image.ndim != 3 or raw_image.shape[2] != 4:
        return None
    alpha = raw_image[:, :, 3]
    # Threshold out noise below alpha 10
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = raw_image[y : y + h, x : x + w].copy()
    cropped[cropped[:, :, 3] <= 10] = 0
    return cropped


def extract_webm_move_frames(
    webm_path: Path,
    target_dir: Path,
    num_frames: int = 6,
) -> tuple[int, list[dict]]:
    """Extract clean RGBA move frames from an existing WebM video file with fps_mode vfr."""
    target_dir.mkdir(parents=True, exist_ok=True)
    provenance_records = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use libvpx to preserve transparency (yuva420p -> png), -fps_mode vfr to prevent frame duplication
        cmd = [
            "ffmpeg",
            "-y",
            "-c:v",
            "libvpx",
            "-i",
            str(webm_path),
            "-vf",
            "select=not(mod(n\\,5))",
            "-fps_mode",
            "vfr",
            "-vframes",
            str(num_frames),
            f"{tmpdir}/frame_%03d.png",
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            logger.warning(f"ffmpeg failed for {webm_path}")
            return 0, []

        frames = sorted(Path(tmpdir).glob("*.png"))
        count = 0
        for i, frame_file in enumerate(frames):
            img = cv2.imread(str(frame_file), cv2.IMREAD_UNCHANGED)
            cropped = clean_and_crop(img)
            if cropped is not None and cropped.shape[0] > 10 and cropped.shape[1] > 10:
                out_path = target_dir / f"{count:02d}.png"
                cv2.imwrite(str(out_path), cropped)
                data = out_path.read_bytes()
                provenance_records.append(
                    {
                        "file": f"move/{count:02d}.png",
                        "pose": "move",
                        "source_type": "local_webm",
                        "source_origin": str(webm_path),
                        "animation_action": "Move",
                        "frame_index": i,
                        "width": int(cropped.shape[1]),
                        "height": int(cropped.shape[0]),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "md5": hashlib.md5(data).hexdigest(),
                    }
                )
                count += 1
        return count, provenance_records


IDLE_PRIORITY = [
    "Idle",
    "idle",
    "A_Idle",
    "B_Idle",
    "Default",
    "default",
    "A_Default",
    "B_Default",
    "Relax",
    "Stand",
    "Wait",
]
MOVE_PRIORITY = [
    "Move",
    "Move_Loop",
    "Move_loop",
    "B_Move",
    "A_Move",
    "B_Run",
    "C1_Move",
    "Move_red",
    "Run",
    "Walk",
]
FORBIDDEN_KEYWORDS = ["attack", "die", "hit", "skill", "stun", "appear", "special", "hurt", "craft"]


def select_best_option(available_options: Sequence[str], kind: str) -> str | None:
    """Find the most appropriate animation option for idle or move."""
    avail_lower = {opt.lower().strip(): opt for opt in available_options}

    if kind == "idle":
        for cand in IDLE_PRIORITY:
            if cand.lower() in avail_lower:
                return avail_lower[cand.lower()]
        for opt in available_options:
            lower = opt.lower()
            if any(f in lower for f in FORBIDDEN_KEYWORDS):
                continue
            if "idle" in lower or "default" in lower:
                return opt

    elif kind == "move":
        for cand in MOVE_PRIORITY:
            if cand.lower() in avail_lower:
                return avail_lower[cand.lower()]
        for opt in available_options:
            lower = opt.lower()
            if any(f in lower for f in FORBIDDEN_KEYWORDS):
                continue
            if "move" in lower or "run" in lower or "walk" in lower:
                return opt

    return None


async def capture_frames_from_canvas(
    page, canvas, num_frames: int, interval_ms: int = 250
) -> list[tuple[np.ndarray, bytes]]:
    """Capture RGBA frames directly from the WebGL canvas using requestAnimationFrame."""
    frames = []
    for _ in range(num_frames):
        await page.wait_for_timeout(interval_ms)
        data_url = await canvas.evaluate(
            "el => new Promise(res => requestAnimationFrame(() => res(el.toDataURL('image/png'))))"
        )
        if not data_url or "," not in data_url:
            continue
        raw_b64 = data_url.split(",", 1)[1]
        raw_bytes = base64.b64decode(raw_b64)
        data = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        cropped = clean_and_crop(img)
        if cropped is not None and cropped.shape[0] > 10 and cropped.shape[1] > 10:
            frames.append((cropped, raw_bytes))
    return frames


async def process_prts_enemy(
    browser: Browser,
    enemy: dict,
    sprites_dir: Path,
    *,
    need_idle: bool,
    need_move: bool,
    num_idle: int = 4,
    num_move: int = 4,
    max_retries: int = 2,
) -> tuple[int, int, list[dict]]:
    enemy_id = enemy["enemy_id"]
    orig_name = enemy["original_name"]
    name = enemy["name"]

    enemy_dir = sprites_dir / f"{enemy_id:04d}"
    idle_dir = enemy_dir / "idle"
    move_dir = enemy_dir / "move"

    provenance_records: list[dict] = []

    if not need_idle and not need_move:
        idle_count = len(list(idle_dir.glob("*.png"))) if idle_dir.exists() else 0
        move_count = len(list(move_dir.glob("*.png"))) if move_dir.exists() else 0
        return idle_count, move_count, []

    url = f"https://prts.wiki/w/{urllib.parse.quote(orig_name)}"

    for attempt in range(1, max_retries + 1):
        page = None
        try:
            page = await browser.new_page(viewport={"width": 1000, "height": 800})
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)

            spine_root = page.locator("#spine-root")
            if await spine_root.count() == 0:
                logger.error(f"[{enemy_id:04d} {name}] #spine-root not found on PRTS")
                return 0, 0, []

            await spine_root.scroll_into_view_if_needed()
            load_btn = spine_root.locator("button:has-text('点此载入模型')")
            if await load_btn.count() > 0:
                await load_btn.click()
                await page.wait_for_timeout(2500)

            canvas = spine_root.locator("canvas")
            if await canvas.count() == 0:
                logger.error(f"[{enemy_id:04d} {name}] canvas not found after loading model")
                return 0, 0, []

            # Ensure loop playback (循环播放) is enabled
            switch = spine_root.locator(".n-switch")
            if await switch.count() > 0:
                is_checked = await switch.get_attribute("aria-checked")
                if is_checked != "true":
                    await switch.click()
                    await page.wait_for_timeout(300)

            # Find animation select dropdown (the 3rd .n-select)
            selects = spine_root.locator(".n-select")
            if await selects.count() < 3:
                logger.error(f"[{enemy_id:04d} {name}] Less than 3 selects found")
                return 0, 0, []

            anim_select = selects.nth(2)

            # 1. Idle extraction
            idle_saved = 0
            if need_idle:
                idle_dir.mkdir(parents=True, exist_ok=True)
                await anim_select.click()
                await page.wait_for_timeout(300)
                options = [t.strip() for t in await page.locator(".n-base-select-option").all_inner_texts()]
                best_idle = select_best_option(options, "idle")

                if best_idle:
                    opt_loc = page.locator(f".n-base-select-option:has-text('{best_idle}')").first
                    await opt_loc.click()
                    await page.wait_for_timeout(400)
                else:
                    await page.keyboard.press("Escape")
                    logger.warning(f"[{enemy_id:04d} {name}] No suitable idle animation among {options}")

                frames = await capture_frames_from_canvas(page, canvas, num_frames=num_idle, interval_ms=250)
                for i, (frame, _) in enumerate(frames):
                    out_path = idle_dir / f"{i:02d}.png"
                    cv2.imwrite(str(out_path), frame)
                    data = out_path.read_bytes()
                    provenance_records.append(
                        {
                            "file": f"idle/{i:02d}.png",
                            "pose": "idle",
                            "source_type": "prts_spine_live",
                            "source_origin": url,
                            "animation_action": best_idle or "unknown",
                            "frame_index": i,
                            "width": int(frame.shape[1]),
                            "height": int(frame.shape[0]),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "md5": hashlib.md5(data).hexdigest(),
                        }
                    )
                idle_saved = len(frames)

            # 2. Move extraction
            move_saved = 0
            if need_move:
                move_dir.mkdir(parents=True, exist_ok=True)
                await anim_select.click()
                await page.wait_for_timeout(300)
                options = [t.strip() for t in await page.locator(".n-base-select-option").all_inner_texts()]
                best_move = select_best_option(options, "move")

                if best_move:
                    opt_loc = page.locator(f".n-base-select-option:has-text('{best_move}')").first
                    await opt_loc.click()
                    await page.wait_for_timeout(400)
                    frames = await capture_frames_from_canvas(page, canvas, num_frames=num_move, interval_ms=250)
                    for i, (frame, _) in enumerate(frames):
                        out_path = move_dir / f"{i:02d}.png"
                        cv2.imwrite(str(out_path), frame)
                        data = out_path.read_bytes()
                        provenance_records.append(
                            {
                                "file": f"move/{i:02d}.png",
                                "pose": "move",
                                "source_type": "prts_spine_live",
                                "source_origin": url,
                                "animation_action": best_move,
                                "frame_index": i,
                                "width": int(frame.shape[1]),
                                "height": int(frame.shape[0]),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "md5": hashlib.md5(data).hexdigest(),
                            }
                        )
                    move_saved = len(frames)
                else:
                    await page.keyboard.press("Escape")
                    logger.info(
                        f"[{enemy_id:04d} {name}] Stationary unit without Move anim "
                        f"(options: {options}). Falling back to idle frames."
                    )
                    # Fallback to idle frames
                    idle_files = sorted(idle_dir.glob("*.png"))
                    for i, idle_f in enumerate(idle_files):
                        target_f = move_dir / f"{i:02d}.png"
                        shutil.copy2(idle_f, target_f)
                        data = target_f.read_bytes()
                        provenance_records.append(
                            {
                                "file": f"move/{i:02d}.png",
                                "pose": "move",
                                "source_type": "fallback_idle",
                                "source_origin": f"idle/{idle_f.name}",
                                "animation_action": "Fallback from Idle (Stationary unit)",
                                "frame_index": i,
                                "width": int(cv2.imread(str(target_f)).shape[1]),
                                "height": int(cv2.imread(str(target_f)).shape[0]),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "md5": hashlib.md5(data).hexdigest(),
                            }
                        )
                    move_saved = len(idle_files)

            logger.info(f"[{enemy_id:04d} {name}] Success: idle={idle_saved}, move={move_saved}")
            return idle_saved, move_saved, provenance_records

        except Exception as exc:
            logger.warning(f"[{enemy_id:04d} {name}] Attempt {attempt} failed: {exc}")
            if attempt == max_retries:
                logger.error(f"[{enemy_id:04d} {name}] All retries failed.")
                return 0, 0, []
            await asyncio.sleep(2)
        finally:
            if page:
                await page.close()

    return 0, 0, []


async def main_async(args):
    workspace = Path(args.workspace).resolve()
    cat_path = workspace / "assets" / "catalog.json"
    sprites_dir = workspace / "assets" / "sprites"
    anim_dir = workspace / "assets" / "animations"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    with open(cat_path, encoding="utf-8") as f:
        catalog = json.load(f)

    # Full list including split sub-entities 92, 93, 94, 95
    enemies = catalog["enemies"] + EXTRA_BATTLEFIELD_ENTITIES

    if args.only:
        only_ids = {int(x.strip()) for x in args.only.split(",")}
        enemies = [e for e in enemies if e["enemy_id"] in only_ids]

    logger.info(f"Total enemies to process: {len(enemies)}")

    unit_provenance_map: dict[int, list[dict]] = {}

    # Step 1: Extract existing clean Move WebMs with -fps_mode vfr
    webm_move_extracted = 0
    for enemy in enemies:
        eid = enemy["enemy_id"]
        move_dir = sprites_dir / f"{eid:04d}" / "move"
        webm_file = anim_dir / f"{eid:04d}" / "animation.webm"
        if not webm_file.is_file():
            # Check CannotMax directory
            cand_files = list(
                (workspace / "CannotMax" / "tools" / "battlefield_composite" / "monster_images").glob(
                    f"*{enemy['name']}*.webm"
                )
            ) + list(
                (workspace / "CannotMax" / "tools" / "battlefield_composite" / "monster_images").glob(
                    f"*{enemy['original_name']}*.webm"
                )
            )
            cand_files = [f for f in cand_files if "attack" not in f.name.lower()]
            if cand_files:
                webm_file = cand_files[0]

        if webm_file.is_file():
            has_move_frames = move_dir.is_dir() and len(list(move_dir.glob("*.png"))) >= 2
            if not has_move_frames or args.force:
                count, records = extract_webm_move_frames(webm_file, move_dir, num_frames=args.move_frames)
                if count > 0:
                    webm_move_extracted += 1
                    unit_provenance_map.setdefault(eid, []).extend(records)

    logger.info(f"Step 1 Complete: {webm_move_extracted} units extracted/verified from clean Move WebMs")

    # Step 2: PRTS Extraction
    tasks_to_run = []
    for enemy in enemies:
        eid = enemy["enemy_id"]
        idle_dir = sprites_dir / f"{eid:04d}" / "idle"
        move_dir = sprites_dir / f"{eid:04d}" / "move"

        has_idle = idle_dir.is_dir() and len(list(idle_dir.glob("*.png"))) >= 2
        has_move = move_dir.is_dir() and len(list(move_dir.glob("*.png"))) >= 2

        need_idle = not has_idle or args.force
        need_move = not has_move or args.force

        if need_idle or need_move:
            tasks_to_run.append((enemy, need_idle, need_move))

    logger.info(f"Step 2: {len(tasks_to_run)} units need PRTS extraction")

    if tasks_to_run:
        sem = asyncio.Semaphore(args.concurrency)

        async def worker(browser, enemy, n_idle, n_move):
            async with sem:
                _, _, records = await process_prts_enemy(
                    browser,
                    enemy,
                    sprites_dir,
                    need_idle=n_idle,
                    need_move=n_move,
                    num_idle=args.idle_frames,
                    num_move=args.move_frames,
                )
                if records:
                    unit_provenance_map.setdefault(enemy["enemy_id"], []).extend(records)

        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            tasks = [worker(browser, enemy, n_idle, n_move) for enemy, n_idle, n_move in tasks_to_run]
            await asyncio.gather(*tasks)
            await browser.close()

    # Step 3: Verification & Summary & Provenance Export
    logger.info("=== Sprite Extraction Verification & Provenance Summary ===")
    complete_count = 0
    total_idle_frames = 0
    total_move_frames = 0
    single_frame_move_units = []
    global_provenance = []

    entity_to_contestant = {}
    for enemy in enemies:
        eid = enemy["enemy_id"]
        entity_to_contestant[str(eid)] = enemy.get("parent_contestant_id", eid)

    for enemy in enemies:
        eid = enemy["enemy_id"]
        name = enemy["name"]
        unit_dir = sprites_dir / f"{eid:04d}"
        idle_frames = sorted((unit_dir / "idle").glob("*.png"))
        move_frames = sorted((unit_dir / "move").glob("*.png"))

        total_idle_frames += len(idle_frames)
        total_move_frames += len(move_frames)

        # Check unique hashes in move
        move_hashes = {hashlib.md5(f.read_bytes()).hexdigest() for f in move_frames}
        if len(move_hashes) <= 1 and len(move_frames) > 1:
            single_frame_move_units.append((eid, name))

        if idle_frames and move_frames:
            complete_count += 1

        # Save unit-level provenance
        existing_records = unit_provenance_map.get(eid, [])
        unit_provenance = {
            "enemy_id": eid,
            "name": name,
            "original_name": enemy.get("original_name", name),
            "parent_contestant_id": enemy.get("parent_contestant_id", eid),
            "idle_frames_count": len(idle_frames),
            "move_frames_count": len(move_frames),
            "move_unique_hashes": len(move_hashes),
            "records": existing_records,
        }
        (unit_dir / "provenance.json").write_text(
            json.dumps(unit_provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        global_provenance.append(unit_provenance)

    # Save global provenance and entity mapping
    (sprites_dir / "provenance.json").write_text(
        json.dumps(global_provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (sprites_dir / "entity_to_contestant.json").write_text(
        json.dumps(entity_to_contestant, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Total units checked: {len(enemies)}")
    logger.info(f"Complete units (has both idle & move): {complete_count}/{len(enemies)}")
    logger.info(f"Total idle frames: {total_idle_frames}, total move frames: {total_move_frames}")
    if single_frame_move_units:
        logger.warning(
            f"Units with static single-hash move frames ({len(single_frame_move_units)}): {single_frame_move_units}"
        )
    else:
        logger.info("All units verified with dynamic, distinct walk frames!")


def main():
    parser = argparse.ArgumentParser(description="Extract battlefield sprite frames for all 95 units")
    parser.add_argument("--workspace", default="/mnt/data/MAA-DuelChannel/workspace", help="Path to workspace")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent browser tabs")
    parser.add_argument("--idle-frames", type=int, default=4, help="Number of idle frames to capture")
    parser.add_argument("--move-frames", type=int, default=6, help="Number of move frames to capture")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of existing sprites")
    parser.add_argument("--only", type=str, default="", help="Comma-separated enemy IDs to process")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
