"""Local-only SQLite storage for RewindAI learning workspace state."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class LibraryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS courses (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS video_courses (
                    video_id TEXT PRIMARY KEY,
                    course_id INTEGER NOT NULL REFERENCES courses(id)
                );
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY,
                    video_id TEXT NOT NULL, segment_id TEXT NOT NULL,
                    start_time REAL NOT NULL, end_time REAL NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY, question TEXT NOT NULL,
                    segment_id TEXT, rating TEXT NOT NULL, comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS local_transcript_videos (
                    video_id TEXT PRIMARY KEY, video_path TEXT NOT NULL,
                    video_size INTEGER NOT NULL, video_mtime_ns INTEGER NOT NULL,
                    segments_json TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def courses(self) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT id, name FROM courses ORDER BY id")]

    def local_videos(self) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM local_transcript_videos ORDER BY rowid")]

    def remember_video(self, video_id: str, path: str, size: int, mtime_ns: int, segments_json: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO local_transcript_videos VALUES (?, ?, ?, ?, ?)",
                               (video_id, path, size, mtime_ns, segments_json))

    def forget_video(self, video_id: str) -> bool:
        """Remove only the cached source and transcript, retaining personal notes."""
        with self.connect() as connection:
            return connection.execute("DELETE FROM local_transcript_videos WHERE video_id = ?", (video_id,)).rowcount > 0

    def create_course(self, name: str) -> dict:
        with self.connect() as connection:
            result = connection.execute("INSERT INTO courses (name) VALUES (?)", (name,))
            return {"id": result.lastrowid, "name": name}

    def assignments(self) -> dict[str, int]:
        with self.connect() as connection:
            return {row["video_id"]: row["course_id"] for row in connection.execute(
                "SELECT video_id, course_id FROM video_courses"
            )}

    def assign(self, video_id: str, course_id: int | None) -> bool:
        with self.connect() as connection:
            if course_id is None:
                connection.execute("DELETE FROM video_courses WHERE video_id = ?", (video_id,))
                return True
            if connection.execute("SELECT 1 FROM courses WHERE id = ?", (course_id,)).fetchone() is None:
                return False
            connection.execute(
                "INSERT INTO video_courses (video_id, course_id) VALUES (?, ?) "
                "ON CONFLICT(video_id) DO UPDATE SET course_id = excluded.course_id",
                (video_id, course_id),
            )
            return True

    def bookmarks(self) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id, video_id, segment_id, start_time, end_time, note, created_at "
                "FROM bookmarks ORDER BY id DESC"
            )]

    def create_bookmark(self, video_id: str, segment_id: str, start: float, end: float, note: str) -> dict:
        with self.connect() as connection:
            result = connection.execute(
                "INSERT INTO bookmarks (video_id, segment_id, start_time, end_time, note) "
                "VALUES (?, ?, ?, ?, ?)", (video_id, segment_id, start, end, note),
            )
            return dict(connection.execute("SELECT * FROM bookmarks WHERE id = ?", (result.lastrowid,)).fetchone())

    def update_note(self, bookmark_id: int, note: str) -> dict | None:
        with self.connect() as connection:
            connection.execute("UPDATE bookmarks SET note = ? WHERE id = ?", (note, bookmark_id))
            row = connection.execute("SELECT * FROM bookmarks WHERE id = ?", (bookmark_id,)).fetchone()
            return dict(row) if row else None

    def delete_bookmark(self, bookmark_id: int) -> bool:
        with self.connect() as connection:
            return connection.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,)).rowcount > 0

    def add_feedback(self, question: str, segment_id: str | None, rating: str, comment: str) -> int:
        with self.connect() as connection:
            result = connection.execute(
                "INSERT INTO feedback (question, segment_id, rating, comment) VALUES (?, ?, ?, ?)",
                (question, segment_id, rating, comment),
            )
            return int(result.lastrowid)

    def recent_feedback(self, limit: int = 100) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id, question, segment_id, rating, comment, created_at "
                "FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
            )]
