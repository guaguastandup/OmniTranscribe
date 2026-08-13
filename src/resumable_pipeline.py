from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from project_store import ProjectStore


ProgressCallback = Optional[Callable[[int, int, str], None]]
StopCallback = Optional[Callable[[], bool]]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm", ".m4v"}

from subtitle_utils import SubtitleRow, ms_to_srt, parse_srt, render_srt, srt_to_ms


def resource_path(relative: str) -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parent.parent
    return base / relative


def bootstrap_runtime() -> None:
    """Put bundled FFmpeg/ffprobe and font resources on the runtime path."""
    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        candidates.extend(
            [
                Path(sys.executable).resolve().parent / "bin",
                Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "bin",
            ]
        )
    candidates.append(Path(__file__).resolve().parent.parent / "bin")
    for candidate in candidates:
        if candidate.exists():
            current = os.environ.get("PATH", "")
            if str(candidate) not in current.split(os.pathsep):
                os.environ["PATH"] = str(candidate) + os.pathsep + current
    bundled_font = resource_path("ChillDuanSansVF.ttf")
    if bundled_font.exists() and not os.getenv("OMNITRANSCRIBE_FONT_PATH"):
        os.environ["OMNITRANSCRIBE_FONT_PATH"] = str(bundled_font)


def _tool(name: str) -> str:
    bootstrap_runtime()
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(
            f"Required media tool '{name}' was not found. "
            "Install FFmpeg or use the packaged OmniTranscribe build."
        )
    return resolved


def media_duration(path: str | Path) -> float:
    result = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _progress(callback: ProgressCallback, current: int, total: int, message: str) -> None:
    if callback:
        callback(current, total, message)


class ResumablePipeline:
    """Project-aware transcription/translation runner with chunk checkpoints."""

    def __init__(self, store: Optional[ProjectStore] = None):
        self.store = store or ProjectStore()

    def create_project(self, source_path: str, **kwargs):
        project = self.store.create_project(source_path, **kwargs)
        try:
            duration = media_duration(project.source_path)
            project = self.store.update_project(project.id, duration_seconds=duration)
        except Exception:
            # Duration is discovered again before transcription; importing a project
            # should not fail solely because ffprobe is missing in a dev checkout.
            pass
        return project

    def _transcription_units(self, duration_seconds: float, chunk_seconds: int) -> List[Dict]:
        total_ms = max(1, int(duration_seconds * 1000))
        chunk_ms = max(30_000, int(chunk_seconds * 1000))
        return [
            {
                "unit_index": index,
                "start_ms": start_ms,
                "end_ms": min(total_ms, start_ms + chunk_ms),
                "payload": {"chunk_seconds": chunk_seconds},
            }
            for index, start_ms in enumerate(range(0, total_ms, chunk_ms))
        ]

    def _extract_chunk(
        self,
        source_path: str,
        output_path: Path,
        start_ms: int,
        end_ms: int,
        overlap_ms: int,
        media_end_ms: int,
    ) -> Tuple[int, int]:
        extraction_start = max(0, start_ms - overlap_ms)
        extraction_end = min(media_end_ms, end_ms + overlap_ms)
        duration = max(0.1, (extraction_end - extraction_start) / 1000.0)
        subprocess.run(
            [
                _tool("ffmpeg"), "-y", "-v", "error",
                "-ss", f"{extraction_start / 1000.0:.3f}",
                "-i", source_path,
                "-t", f"{duration:.3f}",
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(output_path),
            ],
            check=True,
        )
        return extraction_start, extraction_end

    def transcribe(
        self,
        project_id: str,
        *,
        chunk_seconds: int = 300,
        overlap_seconds: float = 1.5,
        progress: ProgressCallback = None,
        should_stop: StopCallback = None,
    ) -> Dict:
        project = self.store.get_project(project_id)
        duration = project.duration_seconds or media_duration(project.source_path)
        if project.duration_seconds is None:
            self.store.update_project(project_id, duration_seconds=duration)

        checkpoint = self.store.get_checkpoint(project_id, "transcription")
        signature = {
            "source_hash": project.source_hash,
            "whisper_model": project.whisper_model,
            "source_language": project.source_language,
            "chunk_seconds": int(chunk_seconds),
            "overlap_seconds": float(overlap_seconds),
        }
        existing_units = self.store.task_units(project_id, "transcription")
        if existing_units:
            metadata = checkpoint.get("metadata") or {}
            for key in ("source_hash", "whisper_model", "source_language"):
                if metadata.get(key) and metadata.get(key) != signature[key]:
                    raise RuntimeError(
                        "这个项目的转录已经开始，恢复时必须保持源语言和 Whisper 模型不变。"
                        "如需更换设置，请新建项目后重新转录。"
                    )
            units = existing_units
            overlap_seconds = float(metadata.get("overlap_seconds", overlap_seconds))
        else:
            units = self._transcription_units(duration, chunk_seconds)
            self.store.ensure_task_units(project_id, "transcription", units)
            self.store.refresh_checkpoint(
                project_id, "transcription", status="paused", metadata=signature
            )

        self.store.reset_running_units(project_id, "transcription")
        pending = self.store.pending_task_units(project_id, "transcription")
        if not pending:
            return self.store.get_checkpoint(project_id, "transcription")

        self.store.update_project(project_id, stage="transcribing")
        from transcribe import AudioTranscriber

        transcriber = AudioTranscriber(
            model_size=project.whisper_model,
            device="auto",
            use_cache=False,
        )
        overlap_ms = max(0, int(overlap_seconds * 1000))
        work_dir = self.store.project_dir(project_id) / "work" / "transcription"
        work_dir.mkdir(parents=True, exist_ok=True)
        total = len(units)

        for unit in pending:
            if should_stop and should_stop():
                checkpoint = self.store.refresh_checkpoint(
                    project_id, "transcription", status="paused"
                )
                self.store.update_project(project_id, stage="transcription_paused")
                return checkpoint

            index = int(unit["unit_index"])
            start_ms = int(unit["start_ms"])
            end_ms = int(unit["end_ms"])
            self.store.set_task_unit_status(project_id, "transcription", index, "running")
            _progress(progress, index, total, f"转录 {ms_to_srt(start_ms)} – {ms_to_srt(end_ms)}")
            chunk_path = work_dir / f"chunk_{index:05d}.wav"
            try:
                extraction_start, _ = self._extract_chunk(
                    project.source_path,
                    chunk_path,
                    start_ms,
                    end_ms,
                    overlap_ms,
                    int(duration * 1000),
                )
                language_arg = None if project.source_language == "auto" else project.source_language
                local_srt = transcriber.transcribe_audio(str(chunk_path), language=language_arg)
                shifted = []
                for local_index, row in enumerate(parse_srt(local_srt), 1):
                    absolute_start = extraction_start + row.start_ms
                    absolute_end = extraction_start + row.end_ms
                    midpoint = (absolute_start + absolute_end) / 2
                    if midpoint < start_ms or midpoint >= end_ms:
                        continue
                    clipped_start = max(absolute_start, start_ms)
                    clipped_end = min(absolute_end, end_ms)
                    if clipped_end <= clipped_start:
                        continue
                    shifted.append(
                        {
                            "sequence": (index + 1) * 100_000 + local_index,
                            "start_ms": clipped_start,
                            "end_ms": clipped_end,
                            "source_text": row.text,
                        }
                    )
                self.store.replace_segments_for_unit(project_id, index, shifted)
                self.store.set_task_unit_status(project_id, "transcription", index, "complete")
            except BaseException as exc:
                self.store.set_task_unit_status(
                    project_id, "transcription", index, "failed", error=str(exc)
                )
                self.store.update_project(project_id, stage="transcription_paused")
                raise
            finally:
                try:
                    chunk_path.unlink(missing_ok=True)
                except Exception:
                    pass

        self.store.renumber_segments(project_id)
        checkpoint = self.store.refresh_checkpoint(project_id, "transcription", status="complete")
        self.store.update_project(project_id, stage="transcribed")
        _progress(progress, total, total, "转录完成，可以开始校对")
        return checkpoint

    def translate(
        self,
        project_id: str,
        *,
        batch_size: int = 20,
        include_manual: bool = False,
        progress: ProgressCallback = None,
        should_stop: StopCallback = None,
    ) -> Dict:
        project = self.store.get_project(project_id)
        pending_segments = self.store.segments_needing_translation(
            project_id, include_manual=include_manual
        )
        if not pending_segments:
            self.store.update_project(project_id, stage="translated")
            return self.store.refresh_checkpoint(project_id, "translation", status="complete")

        size = max(1, batch_size)
        batches = [pending_segments[i:i + size] for i in range(0, len(pending_segments), size)]
        units = [
            {"unit_index": index, "payload": {"sequences": [row["sequence"] for row in batch]}}
            for index, batch in enumerate(batches)
        ]
        # Completed translations live in subtitle_segments. Rebuilding the current
        # plan from only missing/stale rows makes an interrupted run naturally resume.
        self.store.replace_task_units(project_id, "translation", units)
        self.store.update_project(project_id, stage="translating")
        from translator import UniversalTranslator

        translator = UniversalTranslator(model=project.translation_model)
        total = len(batches)
        for index, batch in enumerate(batches):
            if should_stop and should_stop():
                checkpoint = self.store.refresh_checkpoint(
                    project_id, "translation", status="paused"
                )
                self.store.update_project(project_id, stage="translation_paused")
                return checkpoint

            self.store.set_task_unit_status(project_id, "translation", index, "running")
            _progress(progress, index, total, f"翻译第 {index + 1}/{total} 组字幕")
            source_srt = render_srt(
                SubtitleRow(
                    sequence=local_index,
                    start_ms=int(row["start_ms"]),
                    end_ms=int(row["end_ms"]),
                    text=str(row["source_text"]),
                )
                for local_index, row in enumerate(batch, 1)
            )
            try:
                translated_srt = translator.translate_srt(
                    source_srt,
                    prompt_path=str(resource_path("prompt.md")),
                    chunk_size=len(batch),
                    target_language=project.target_language,
                )
                translated_rows = parse_srt(translated_srt)
                if len(translated_rows) != len(batch):
                    raise RuntimeError(
                        "Translation response changed the subtitle row count "
                        f"({len(translated_rows)} != {len(batch)})."
                    )
                self.store.save_translation(
                    project_id,
                    {
                        int(original["sequence"]): translated.text
                        for original, translated in zip(batch, translated_rows)
                    },
                )
                self.store.set_task_unit_status(project_id, "translation", index, "complete")
            except BaseException as exc:
                self.store.set_task_unit_status(
                    project_id, "translation", index, "failed", error=str(exc)
                )
                self.store.update_project(project_id, stage="translation_paused")
                raise

        checkpoint = self.store.refresh_checkpoint(project_id, "translation", status="complete")
        self.store.update_project(project_id, stage="translated")
        _progress(progress, total, total, "翻译完成，可以继续校对或导出")
        return checkpoint

    def save_editor_rows(self, project_id: str, rows: Sequence[Sequence]) -> None:
        for row in rows:
            if len(row) < 5:
                continue
            sequence = int(row[0])
            start_ms = srt_to_ms(str(row[1]))
            end_ms = srt_to_ms(str(row[2]))
            if end_ms <= start_ms:
                raise ValueError(f"第 {sequence} 条字幕结束时间必须晚于开始时间")
            self.store.update_segment(
                project_id,
                sequence,
                start_ms=start_ms,
                end_ms=end_ms,
                source_text=str(row[3] or "").strip(),
                translated_text=str(row[4] or "").strip(),
            )

    def editor_rows(self, project_id: str) -> List[List]:
        return [
            [
                row["sequence"],
                ms_to_srt(row["start_ms"]),
                ms_to_srt(row["end_ms"]),
                row["source_text"],
                row["translated_text"] or "",
                row["translation_status"],
            ]
            for row in self.store.list_segments(project_id)
        ]

    def export_subtitles(self, project_id: str, mode: str = "bilingual") -> Path:
        project = self.store.get_project(project_id)
        rows = self.store.list_segments(project_id)
        if not rows:
            raise RuntimeError("项目还没有字幕，请先完成转录")
        if mode in {"translation", "bilingual"}:
            missing = [row for row in rows if not (row["translated_text"] or "").strip()]
            if missing:
                raise RuntimeError(
                    f"还有 {len(missing)} 条字幕没有译文。请完成翻译/校对，或选择“原文”导出。"
                )

        output_rows = []
        for row in rows:
            source = (row["source_text"] or "").strip()
            translated = (row["translated_text"] or "").strip()
            if mode == "source":
                text = source
            elif mode == "translation":
                text = translated
            else:
                text = f"{source}\n{translated}"
            output_rows.append(
                SubtitleRow(
                    sequence=int(row["sequence"]),
                    start_ms=int(row["start_ms"]),
                    end_ms=int(row["end_ms"]),
                    text=text,
                )
            )

        exports = self.store.project_dir(project_id) / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        output = exports / f"{project.name}_{mode}.srt"
        output.write_text(render_srt(output_rows), encoding="utf-8")
        self.store.update_project(project_id, stage="ready_to_export")
        return output

    def export_media(
        self,
        project_id: str,
        *,
        subtitle_mode: str = "bilingual",
        output_kind: str = "srt",
        background_image: Optional[str] = None,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        album: Optional[str] = None,
        font_size: int = 24,
    ) -> Path:
        subtitle_path = self.export_subtitles(project_id, subtitle_mode)
        if output_kind == "srt":
            return subtitle_path

        project = self.store.get_project(project_id)
        source = Path(project.source_path)
        exports = self.store.project_dir(project_id) / "exports"
        work = self.store.project_dir(project_id) / "work"
        exports.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)

        if output_kind == "mp4":
            if source.suffix.lower() in VIDEO_EXTENSIONS:
                return self.burn_subtitles_to_video(project_id, subtitle_path, font_size=font_size)

            from converter import SubtitleConverter
            from final_video_generator import FinalVideoGenerator

            background = (
                Path(background_image)
                if background_image
                else resource_path("assets/default_background.png")
            )
            if not background.exists():
                raise FileNotFoundError("生成音频型 MP4 需要背景图片")
            audio_mp3 = work / "source_audio.mp3"
            if not audio_mp3.exists():
                subprocess.run(
                    [
                        _tool("ffmpeg"), "-y", "-i", str(source), "-vn",
                        "-c:a", "libmp3lame", "-q:a", "2", str(audio_mp3),
                    ],
                    check=True,
                    capture_output=True,
                )
            lyrics_path = SubtitleConverter().convert_file(str(subtitle_path), "lrc")
            output = exports / f"{project.name}_subtitled.mp4"
            result = FinalVideoGenerator().create_video_from_existing_files(
                audio_path=str(audio_mp3),
                lyrics_path=lyrics_path,
                image_path=str(background),
                output_path=str(output),
            )
            self.store.update_project(project_id, stage="exported")
            return Path(result)

        if output_kind == "mp3":
            from simple_mp3_embedder import SimpleMP3Embedder

            audio_mp3 = work / "source_audio.mp3"
            if source.suffix.lower() == ".mp3":
                audio_mp3 = source
            elif not audio_mp3.exists():
                subprocess.run(
                    [
                        _tool("ffmpeg"), "-y", "-i", str(source), "-vn",
                        "-c:a", "libmp3lame", "-q:a", "2", str(audio_mp3),
                    ],
                    check=True,
                    capture_output=True,
                )
            output = exports / f"{project.name}_subtitled.mp3"
            result = SimpleMP3Embedder().embed_metadata(
                audio_path=str(audio_mp3),
                lyrics_path=str(subtitle_path),
                cover_path=background_image,
                title=title or project.name,
                artist=artist,
                album=album,
                output_path=str(output),
            )
            self.store.update_project(project_id, stage="exported")
            return Path(result)

        raise ValueError(f"Unsupported output kind: {output_kind}")

    def burn_subtitles_to_video(
        self,
        project_id: str,
        subtitle_path: str | Path,
        *,
        font_size: int = 24,
    ) -> Path:
        project = self.store.get_project(project_id)
        source = Path(project.source_path)
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError("当前输入是音频，请在导出页提供背景图后生成 MP4")
        exports = self.store.project_dir(project_id) / "exports"
        output = exports / f"{project.name}_subtitled.mp4"
        subtitle = Path(subtitle_path).resolve()
        filter_path = str(subtitle).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        style = f"FontSize={max(12, int(font_size))},Outline=1,Shadow=0,MarginV=24"
        subprocess.run(
            [
                _tool("ffmpeg"), "-y", "-i", str(source),
                "-vf", f"subtitles='{filter_path}':force_style='{style}'",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", str(output),
            ],
            check=True,
        )
        self.store.update_project(project_id, stage="exported")
        return output
