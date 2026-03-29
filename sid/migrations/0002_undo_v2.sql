CREATE TABLE IF NOT EXISTS undo_ops_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    op_type TEXT NOT NULL,
    src_path TEXT NOT NULL,
    dst_path TEXT NOT NULL,
    backup_blob BLOB,
    reversed INTEGER NOT NULL DEFAULT 0
);
