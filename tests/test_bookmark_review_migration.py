import sqlite3

from video_rag.api.library import LibraryStore


def test_existing_bookmark_and_note_survive_review_status_migration(tmp_path):
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE bookmarks (id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, '
                   'segment_id TEXT NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL, '
                   "note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT 'old')")
        db.execute("INSERT INTO bookmarks (video_id, segment_id, start_time, end_time, note) VALUES ('v', 's', 1, 3, 'keep')")
    store = LibraryStore(path)
    original = store.bookmarks()[0]
    assert (original['note'], original['reviewed']) == ('keep', 0)
    store.set_bookmark_reviewed(original['id'], True)
    reopened = LibraryStore(path).bookmarks()[0]
    assert (reopened['note'], reopened['reviewed']) == ('keep', 1)
