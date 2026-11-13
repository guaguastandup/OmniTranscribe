from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from platformdirs import user_data_dir
except ImportError:  # pragma: no cover
    user_data_dir = None


SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_data_dir() -> Path:
    if user_data_dir:
        return Path(user_data_dir("OmniTranscribe", "OmniTranscribe"))
    return Path.home() / ".omnitranscribe"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    source_path: str
    original_source_path: str
    source_hash: str
    source_language: str
    target_language: str
    whisper_model: str
    translation_model: str
    stage: str
    duration_seconds: Optional[float]
    working_audio_path: Optional[str]
    created_at: str
    updated_at: str


class ProjectStore:
    """SQLite-backed project state used by the resumable product workflow."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "omnitranscribe.db"
        self._migrate()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _migrate(self) -> None:
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    original_source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    source_language TEXT NOT NULL DEFAULT 'auto',
                    target_language TEXT NOT NULL DEFAULT 'zh',
                    whisper_model TEXT NOT NULL DEFAULT 'base',
                    translation_model TEXT NOT NULL DEFAULT 'google',
                    stage TEXT NOT NULL DEFAULT 'imported',
                    duration_seconds REAL,
                    working_audio_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_units (
                    project_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    unit_index INTEGER NOT NULL,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    payload_json TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, task_name, unit_index),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    project_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    completed_units INTEGER NOT NULL DEFAULT 0,
                    total_units INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    metadata_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, task_name),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS subtitle_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT,
                    source_edited INTEGER NOT NULL DEFAULT 0,
                    translation_edited INTEGER NOT NULL DEFAULT 0,
                    translation_status TEXT NOT NULL DEFAULT 'pending',
                    source_unit_index INTEGER,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, sequence),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_segments_project_time
                    ON subtitle_segments(project_id, start_ms, sequence);
                CREATE INDEX IF NOT EXISTS idx_segments_translation_status
                    ON subtitle_segments(project_id, translation_status, sequence);
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(projects)").fetchall()}
            if "original_source_path" not in columns:
                db.execute("ALTER TABLE projects ADD COLUMN original_source_path TEXT")
                db.execute(
                    "UPDATE projects SET original_source_path=source_path WHERE original_source_path IS NULL"
                )
            db.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def create_project(
        self,
        source_path: str | Path,
        *,
        name: Optional[str] = None,
        source_language: str = "auto",
        target_language: str = "zh",
        whisper_model: str = "base",
        translation_model: str = "google",
    ) -> Project:
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source media not found: {source}")

        project_id = uuid.uuid4().hex
        project_dir = self.project_dir(project_id)
        media_dir = project_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "exports").mkdir(exist_ok=True)
        (project_dir / "work").mkdir(exist_ok=True)

        managed_source = media_dir / source.name
        try:
            os.link(source, managed_source)
        except OSError:
            import shutil
            shutil.copy2(source, managed_source)

        now = _now()
        record = {
            "id": project_id,
            "name": (name or source.stem).strip() or source.stem,
            "source_path": str(managed_source),
            "original_source_path": str(source),
            "source_hash": file_sha256(managed_source),
            "source_language": source_language,
            "target_language": target_language,
            "whisper_model": whisper_model,
            "translation_model": translation_model,
            "stage": "imported",
            "duration_seconds": None,
            "working_audio_path": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO projects(
                    id, name, source_path, original_source_path, source_hash,
                    source_language, target_language, whisper_model, translation_model,
                    stage, duration_seconds, working_audio_path, created_at, updated_at
                ) VALUES(
                    :id, :name, :source_path, :original_source_path, :source_hash,
                    :source_language, :target_language, :whisper_model, :translation_model,
                    :stage, :duration_seconds, :working_audio_path, :created_at, :updated_at
                )
                """,
                record,
            )
        return Project(**record)

    def get_project(self, project_id: str) -> Project:
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise KeyError(f"Unknown project: {project_id}")
        return Project(**dict(row))

    def list_projects(self, limit: int = 50) -> List[Project]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Project(**dict(row)) for row in rows]

    def update_project(self, project_id: str, **fields: Any) -> Project:
        allowed = {
            "name", "source_language", "target_language", "whisper_model",
            "translation_model", "stage", "duration_seconds", "working_audio_path",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return self.get_project(project_id)
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._connect() as db:
            db.execute(
                f"UPDATE projects SET {assignments} WHERE id=?",
                list(updates.values()) + [project_id],
            )
        return self.get_project(project_id)

    def ensure_task_units(
        self, project_id: str, task_name: str, units: Sequence[Dict[str, Any]]
    ) -> None:
        now = _now()
        with self._connect() as db:
            for unit in units:
                db.execute(
                    """
                    INSERT OR IGNORE INTO task_units(
                        project_id, task_name, unit_index, start_ms, end_ms,
                        status, payload_json, last_error, updated_at
                    ) VALUES(?, ?, ?, ?, ?, 'pending', ?, NULL, ?)
                    """,
                    (
                        project_id,
                        task_name,
                        int(unit["unit_index"]),
                        unit.get("start_ms"),
                        unit.get("end_ms"),
                        json.dumps(unit.get("payload") or {}, ensure_ascii=False),
                        now,
                    ),
                )
        self.refresh_checkpoint(project_id, task_name)

    def replace_task_units(
        self, project_id: str, task_name: str, units: Sequence[Dict[str, Any]]
    ) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM task_units WHERE project_id=? AND task_name=?", (project_id, task_name))
            db.execute("DELETE FROM task_checkpoints WHERE project_id=? AND task_name=?", (project_id, task_name))
        self.ensure_task_units(project_id, task_name, units)

    def _decode_units(self, rows) -> List[Dict[str, Any]]:
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            result.append(item)
        return result

    def task_units(self, project_id: str, task_name: str) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM task_units WHERE project_id=? AND task_name=? ORDER BY unit_index",
                (project_id, task_name),
            ).fetchall()
        return self._decode_units(rows)

    def pending_task_units(self, project_id: str, task_name: str) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT * FROM task_units
                WHERE project_id=? AND task_name=? AND status!='complete'
                ORDER BY unit_index
                """,
                (project_id, task_name),
            ).fetchall()
        return self._decode_units(rows)

    def reset_running_units(self, project_id: str, task_name: str) -> None:
        now = _now()
        with self._connect() as db:
            db.execute(
                """UPDATE task_units SET status='pending', updated_at=?
                   WHERE project_id=? AND task_name=? AND status='running'""",
                (now, project_id, task_name),
            )
            db.execute(
                """UPDATE task_checkpoints SET status='paused', updated_at=?
                   WHERE project_id=? AND task_name=? AND status='running'""",
                (now, project_id, task_name),
            )

    def set_task_unit_status(
        self,
        project_id: str,
        task_name: str,
        unit_index: int,
        status: str,
        *,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE task_units SET status=?, last_error=?, updated_at=?
                   WHERE project_id=? AND task_name=? AND unit_index=?""",
                (status, error, _now(), project_id, task_name, unit_index),
            )
        self.refresh_checkpoint(project_id, task_name, error=error)

    def refresh_checkpoint(
        self,
        project_id: str,
        task_name: str,
        *,
        status: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _now()
        with self._connect() as db:
            existing = db.execute(
                "SELECT metadata_json FROM task_checkpoints WHERE project_id=? AND task_name=?",
                (project_id, task_name),
            ).fetchone()
            metadata_json = (
                json.dumps(metadata, ensure_ascii=False)
                if metadata is not None
                else ((existing["metadata_json"] if existing else "{}") or "{}")
            )
            counts = db.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status='complete' THEN 1 ELSE 0 END) AS done,
                       SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running
                FROM task_units WHERE project_id=? AND task_name=?
                """,
                (project_id, task_name),
            ).fetchone()
            total = int(counts["total"] or 0)
            done = int(counts["done"] or 0)
            running = int(counts["running"] or 0)
            if status is None:
                status = "complete" if total and done == total else ("running" if running else "paused")
            db.execute(
                """
                INSERT INTO task_checkpoints(
                    project_id, task_name, status, completed_units, total_units,
                    last_error, metadata_json, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, task_name) DO UPDATE SET
                    status=excluded.status,
                    completed_units=excluded.completed_units,
                    total_units=excluded.total_units,
                    last_error=excluded.last_error,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (project_id, task_name, status, done, total, error, metadata_json, now),
            )
        return self.get_checkpoint(project_id, task_name)

    def get_checkpoint(self, project_id: str, task_name: str) -> Dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM task_checkpoints WHERE project_id=? AND task_name=?",
                (project_id, task_name),
            ).fetchone()
        if not row:
            return {
                "project_id": project_id,
                "task_name": task_name,
                "status": "pending",
                "completed_units": 0,
                "total_units": 0,
                "last_error": None,
                "metadata": {},
            }
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        return item

    def replace_segments_for_unit(
        self, project_id: str, unit_index: int, segments: Iterable[Dict[str, Any]]
    ) -> None:
        now = _now()
        with self._connect() as db:
            db.execute(
                "DELETE FROM subtitle_segments WHERE project_id=? AND source_unit_index=?",
                (project_id, unit_index),
            )
            for segment in segments:
                db.execute(
                    """
                    INSERT INTO subtitle_segments(
                        project_id, sequence, start_ms, end_ms, source_text,
                        translated_text, source_edited, translation_edited,
                        translation_status, source_unit_index, updated_at
                    ) VALUES(?, ?, ?, ?, ?, NULL, 0, 0, 'pending', ?, ?)
                    """,
                    (
                        project_id,
                        int(segment["sequence"]),
                        int(segment["start_ms"]),
                        int(segment["end_ms"]),
                        str(segment["source_text"]).strip(),
                        unit_index,
                        now,
                    ),
                )

    def renumber_segments(self, project_id: str) -> None:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id FROM subtitle_segments WHERE project_id=? ORDER BY start_ms, end_ms, id",
                (project_id,),
            ).fetchall()
            for index, row in enumerate(rows, 1):
                db.execute("UPDATE subtitle_segments SET sequence=? WHERE id=?", (-index, row["id"]))
            for index, row in enumerate(rows, 1):
                db.execute("UPDATE subtitle_segments SET sequence=? WHERE id=?", (index, row["id"]))

    def list_segments(self, project_id: str) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM subtitle_segments WHERE project_id=? ORDER BY sequence",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_segment(
        self,
        project_id: str,
        sequence: int,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        source_text: Optional[str] = None,
        translated_text: Optional[str] = None,
    ) -> None:
        with self._connect() as db:
            current = db.execute(
                "SELECT * FROM subtitle_segments WHERE project_id=? AND sequence=?",
                (project_id, sequence),
            ).fetchone()
            if not current:
                raise KeyError(f"Unknown subtitle segment {sequence}")
            values: Dict[str, Any] = {}
            if start_ms is not None and start_ms != current["start_ms"]:
                values["start_ms"] = start_ms
            if end_ms is not None and end_ms != current["end_ms"]:
                values["end_ms"] = end_ms
            if source_text is not None and source_text != current["source_text"]:
                values["source_text"] = source_text
                values["source_edited"] = 1
                if current["translation_status"] != "manual":
                    values["translation_status"] = "stale"
            if translated_text is not None and translated_text != (current["translated_text"] or ""):
                values["translated_text"] = translated_text
                values["translation_edited"] = 1
                values["translation_status"] = "manual"
            if not values:
                return
            values["updated_at"] = _now()
            assignments = ", ".join(f"{key}=?" for key in values)
            db.execute(
                f"UPDATE subtitle_segments SET {assignments} WHERE project_id=? AND sequence=?",
                list(values.values()) + [project_id, sequence],
            )
        self.update_project(project_id, stage="editing")

    def segments_needing_translation(
        self, project_id: str, include_manual: bool = False
    ) -> List[Dict[str, Any]]:
        statuses = ["pending", "stale"] + (["manual"] if include_manual else [])
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT * FROM subtitle_segments
                    WHERE project_id=? AND translation_status IN ({placeholders})
                    ORDER BY sequence""",
                [project_id] + statuses,
            ).fetchall()
        return [dict(row) for row in rows]

    def save_translation(self, project_id: str, translations: Dict[int, str]) -> None:
        now = _now()
        with self._connect() as db:
            for sequence, text in translations.items():
                db.execute(
                    """UPDATE subtitle_segments
                       SET translated_text=?, translation_status='complete',
                           translation_edited=0, updated_at=?
                       WHERE project_id=? AND sequence=?""",
                    (text.strip(), now, project_id, int(sequence)),
                )
        self.update_project(project_id, stage="translated")

    def delete_project(self, project_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        project_dir = self.project_dir(project_id)
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir, ignore_errors=True)
