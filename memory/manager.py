"""AirPI Memory Manager — persistente Fakten in SQLite + human-readable memory.md.

Kein Import aus server.py oder model_manager.py (kein Import-Loop).
Thread-safe: check_same_thread=False + try/except um alle DB-Calls.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source TEXT DEFAULT 'auto',
    category TEXT DEFAULT 'fact',
    created_at REAL NOT NULL,
    active INTEGER DEFAULT 1
);
"""

_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_manual_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    source_label TEXT DEFAULT '',
    target_label TEXT DEFAULT '',
    relation_type TEXT DEFAULT 'manual',
    label TEXT DEFAULT '',
    directed INTEGER DEFAULT 1,
    weight INTEGER DEFAULT 1,
    confidence INTEGER DEFAULT 80,
    note TEXT DEFAULT '',
    active INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edge_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_key TEXT NOT NULL,
    action TEXT NOT NULL,
    note TEXT DEFAULT '',
    active INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(edge_key, action)
);

CREATE TABLE IF NOT EXISTS graph_manual_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_key    TEXT    NOT NULL UNIQUE,
    label       TEXT    NOT NULL,
    type        TEXT    DEFAULT 'concept',
    note        TEXT    DEFAULT '',
    confidence  INTEGER DEFAULT 80,
    active      INTEGER DEFAULT 1,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
"""

# QW-5: Rückwärtskompatible Schema-Migrationen (idempotent via OperationalError-Catch)
_MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN confidence INTEGER DEFAULT 80",
    "ALTER TABLE memories ADD COLUMN last_seen_at REAL",
]

# QW-5: Gültige Kategorien
_VALID_CATEGORIES = {"fact", "preference", "correction", "project", "system", "todo"}

_CATEGORY_HEADERS = {
    "fact": "Fakten",
    "preference": "Präferenzen",
    "correction": "Korrekturen",
    "project": "Projekte",
    "system": "System",
    "todo": "Aufgaben",
}

_CATEGORY_ORDER = ["fact", "preference", "correction", "project", "system", "todo"]

# QW-6: Maximale Anzahl Einträge in memory.md
MEMORY_MD_MAX_ENTRIES = int(os.environ.get("MEMORY_MD_MAX_ENTRIES", "50"))

_VALID_GRAPH_OVERRIDE_ACTIONS = {"hide"}

_VALID_NODE_TYPES: frozenset[str] = frozenset({
    "person", "place", "tech", "date", "concept"
})


def _today() -> str:
    return date.today().isoformat()


def _normalize_graph_key(value: str) -> str:
    """Stable, display-independent graph key for persisted overlays."""
    return re.sub(r"\s+", " ", value.strip().lower())


def _edge_row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["directed"] = bool(data.get("directed", 1))
    data["active"] = bool(data.get("active", 1))
    return data


def _node_row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["active"] = bool(data.get("active", 1))
    return data


def _score(entry: dict) -> float:
    """QW-6: Scoring-Funktion für memory.md-Sortierung: confidence * (1.0 / max(1, age_days))."""
    confidence = entry.get("confidence") or 80
    last_seen_at = entry.get("last_seen_at")
    if last_seen_at:
        age_days = (time.time() - last_seen_at) / 86400
    else:
        age_days = 30  # Schlechter Score für ältere Einträge ohne last_seen_at
    return confidence * (1.0 / max(1, age_days))


def _format_md(entries: list[dict]) -> str:
    """Rendert alle aktiven Einträge als memory.md."""
    now = _today()
    lines: list[str] = [
        "# AirPI Memory",
        "",
        f"_Automatisch gepflegt. Letzte Änderung: {now}_",
        "",
    ]

    by_category: dict[str, list[dict]] = {cat: [] for cat in _CATEGORY_ORDER}
    for entry in entries:
        cat = entry.get("category", "fact")
        if cat not in by_category:
            by_category.setdefault(cat, [])
        by_category[cat].append(entry)

    for cat in _CATEGORY_ORDER:
        cat_entries = by_category.get(cat, [])
        if not cat_entries:
            continue
        header = _CATEGORY_HEADERS.get(cat, cat.capitalize())
        lines.append(f"## {header}")
        for e in cat_entries:
            ts = date.fromtimestamp(e["created_at"]).isoformat()
            lines.append(f"- [{ts}] {e['content']}")
        lines.append("")

    return "\n".join(lines)


class MemoryManager:
    def __init__(self, db_path: str, memory_md_path: str) -> None:
        self._db_path = db_path
        self._md_path = Path(memory_md_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        self._run_migrations()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Lazy-connect; reuseverbindung ist thread-safe mit check_same_thread=False."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            # WAL-Mode: Reads blockieren Writes nicht; besser für concurrent FastAPI-Calls
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL ist mit WAL sicher (Daten geschützt außer bei OS-Crash/Power-Loss)
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        try:
            conn = self._connect()
            conn.execute(_SCHEMA)
            conn.executescript(_GRAPH_SCHEMA)
            # Indexes für häufige Queries (idempotent — IF NOT EXISTS)
            # all_active(): WHERE active=1 ORDER BY created_at
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_active_created "
                "ON memories(active, created_at)"
            )
            # _is_duplicate() + _touch(): WHERE active=1 AND LOWER(content)=LOWER(?)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_active_content "
                "ON memories(active, content)"
            )
            conn.commit()
        except Exception:
            logger.exception("memory: DB init failed — memory disabled for this session")
            self._conn = None

    def _run_migrations(self) -> None:
        """QW-5: Führt Schema-Migrationen idempotent durch. Ignoriert 'duplicate column' Fehler."""
        if self._conn is None:
            return
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError as exc:
                # "duplicate column name" → Migration bereits angewendet, ignorieren
                if "duplicate column" in str(exc).lower():
                    pass
                else:
                    logger.warning("memory: migration skipped (%s): %s", exc, sql)
            except Exception:
                logger.exception("memory: migration failed: %s", sql)

    def _execute(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Cursor]:
        try:
            conn = self._connect()
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        except Exception:
            logger.exception("memory: DB execute failed: %s", sql)
            return None

    def _write_md(self) -> None:
        """QW-6: Schreibt memory.md atomisch neu — nur Top-N Einträge nach Score."""
        try:
            all_entries = self.all_active()
            # QW-6: Sortiere nach Score (neuere + konfidentere Einträge zuerst), begrenze auf MAX
            sorted_entries = sorted(all_entries, key=_score, reverse=True)
            top_entries = sorted_entries[:MEMORY_MD_MAX_ENTRIES]
            content = _format_md(top_entries)
            self._md_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._md_path.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(self._md_path)
        except Exception:
            logger.exception("memory: Failed to write %s", self._md_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(self) -> str:
        """Gibt den memory.md-Inhalt zurück — wird in System-Prompt injiziert."""
        try:
            if self._md_path.exists():
                return self._md_path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("memory: Cannot read %s", self._md_path)
        # Fallback: generiere direkt aus DB
        try:
            return _format_md(self.all_active())
        except Exception:
            return ""

    def _is_duplicate(self, content: str) -> bool:
        """Prüft ob ein identischer aktiver Eintrag bereits existiert."""
        try:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE active=1 AND LOWER(content)=LOWER(?)",
                (content.strip(),)
            )
            return cur.fetchone()[0] > 0
        except Exception:
            return False  # Im Zweifel: speichern

    def _touch(self, content: str) -> None:
        """QW-5: Aktualisiert last_seen_at für einen existierenden Eintrag (Touch-Mechanismus)."""
        try:
            self._conn.execute(
                "UPDATE memories SET last_seen_at=? WHERE active=1 AND LOWER(content)=LOWER(?)",
                (time.time(), content.strip()),
            )
            self._conn.commit()
        except Exception:
            logger.debug("memory: touch failed for %r", content)

    def store_fact(self, content: str, source: str = "user", category: str = "fact", confidence: int = 80) -> None:
        """Speichert einen Fakt in SQLite + schreibt memory.md neu."""
        content = content.strip()
        if not content:
            return
        # QW-5: Kategorie validieren
        if category not in _VALID_CATEGORIES:
            category = "fact"
        if self._is_duplicate(content):
            logger.warning("memory: Duplikat ignoriert (bereits aktiv): %r", content)
            # QW-5: Touch-Mechanismus — last_seen_at aktualisieren
            self._touch(content)
            self._write_md()
            return
        # QW-5: confidence + last_seen_at beim INSERT setzen
        self._execute(
            "INSERT INTO memories (content, source, category, created_at, active, confidence, last_seen_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
            (content, source, category, time.time(), confidence, time.time()),
        )
        self._write_md()

    def delete_fact(self, keyword: str) -> int:
        """Markiert alle Einträge die `keyword` enthalten als inactive. Gibt Anzahl zurück."""
        keyword = keyword.strip()
        if not keyword:
            return 0
        cur = self._execute(
            "UPDATE memories SET active = 0 WHERE active = 1 AND content LIKE ?",
            (f"%{keyword}%",),
        )
        count = cur.rowcount if cur else 0
        if count > 0:
            self._write_md()
        return count

    def extract_and_store(self, conversation: list[dict]) -> list[str]:
        """Nicht implementiert — Extraktion via LLM läuft in server.py.
        Signatur bleibt für mögliche zukünftige direkte Nutzung erhalten.
        """
        return []

    def store_extracted(self, facts: list) -> None:
        """Speichert eine Liste von Fakten. Akzeptiert str oder {"content": ..., "category": ..., "confidence": ...}"""
        if not facts:
            return
        changed = False
        for item in facts:
            if isinstance(item, dict):
                content = str(item.get("content", "")).strip()
                category = item.get("category", "fact")
                # QW-5: Kategorie validieren, Fallback auf "fact"
                if category not in _VALID_CATEGORIES:
                    category = "fact"
                confidence = int(item.get("confidence", 80))
            elif isinstance(item, str):
                content = item.strip()
                category = "fact"
                confidence = 80
            else:
                continue

            if not content or self._is_duplicate(content):
                if content:
                    # QW-5: Touch-Mechanismus für Duplikate
                    self._touch(content)
                continue

            self.store_fact(content, source="auto", category=category, confidence=confidence)
            changed = True
        if changed:
            self._write_md()

    def rebuild_md(self) -> None:
        """Schreibt memory.md komplett neu aus SQLite."""
        self._write_md()

    def all_active(self) -> list[dict]:
        """Gibt alle aktiven Einträge zurück."""
        try:
            conn = self._connect()
            cur = conn.execute(
                "SELECT id, content, source, category, created_at, confidence, last_seen_at FROM memories WHERE active = 1 ORDER BY created_at ASC"
            )
            return [dict(row) for row in cur.fetchall()]
        except Exception:
            logger.exception("memory: all_active failed")
            return []

    # ── Graph overlays ───────────────────────────────────────────────────────

    def list_manual_edges(self, include_inactive: bool = False) -> list[dict]:
        """Returns persisted user-managed graph edges."""
        try:
            conn = self._connect()
            where = "" if include_inactive else "WHERE active = 1"
            cur = conn.execute(
                f"SELECT * FROM graph_manual_edges {where} ORDER BY created_at ASC"
            )
            return [_edge_row_to_dict(row) for row in cur.fetchall()]
        except Exception:
            logger.exception("memory: list_manual_edges failed")
            return []

    def add_manual_edge(
        self,
        source_key: str,
        target_key: str,
        source_label: str = "",
        target_label: str = "",
        relation_type: str = "manual",
        label: str = "",
        directed: bool = True,
        confidence: int = 80,
        note: str = "",
    ) -> Optional[dict]:
        """Persists a manual graph edge without mutating auto-generated graph data."""
        source_key = _normalize_graph_key(source_key)
        target_key = _normalize_graph_key(target_key)
        relation_type = _normalize_graph_key(relation_type or "manual") or "manual"
        if not source_key or not target_key or source_key == target_key:
            return None
        confidence = max(0, min(100, int(confidence)))
        now = time.time()
        cur = self._execute(
            """
            INSERT INTO graph_manual_edges (
                source_key, target_key, source_label, target_label,
                relation_type, label, directed, weight, confidence, note,
                active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1, ?, ?)
            """,
            (
                source_key,
                target_key,
                source_label.strip(),
                target_label.strip(),
                relation_type,
                label.strip(),
                1 if directed else 0,
                confidence,
                note.strip(),
                now,
                now,
            ),
        )
        if not cur:
            return None
        return self.get_manual_edge(int(cur.lastrowid))

    def get_manual_edge(self, edge_id: int) -> Optional[dict]:
        try:
            conn = self._connect()
            cur = conn.execute("SELECT * FROM graph_manual_edges WHERE id = ?", (edge_id,))
            row = cur.fetchone()
            return _edge_row_to_dict(row) if row else None
        except Exception:
            logger.exception("memory: get_manual_edge failed")
            return None

    def update_manual_edge(self, edge_id: int, updates: dict) -> Optional[dict]:
        """Updates mutable manual-edge fields and returns the updated edge."""
        allowed = {
            "relation_type",
            "label",
            "directed",
            "confidence",
            "note",
            "source_label",
            "target_label",
        }
        fields: list[str] = []
        values: list = []
        for key, value in updates.items():
            if key not in allowed or value is None:
                continue
            if key == "relation_type":
                value = _normalize_graph_key(str(value) or "manual") or "manual"
            elif key in {"label", "note", "source_label", "target_label"}:
                value = str(value).strip()
            elif key == "directed":
                value = 1 if bool(value) else 0
            elif key == "confidence":
                value = max(0, min(100, int(value)))
            fields.append(f"{key} = ?")
            values.append(value)
        if not fields:
            return self.get_manual_edge(edge_id)
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(edge_id)
        cur = self._execute(
            f"UPDATE graph_manual_edges SET {', '.join(fields)} WHERE id = ?",
            tuple(values),
        )
        if not cur or cur.rowcount == 0:
            return None
        return self.get_manual_edge(edge_id)

    def delete_manual_edge(self, edge_id: int) -> bool:
        """Soft-deletes a manual edge."""
        cur = self._execute(
            "UPDATE graph_manual_edges SET active = 0, updated_at = ? WHERE id = ? AND active = 1",
            (time.time(), edge_id),
        )
        return bool(cur and cur.rowcount > 0)

    def list_edge_overrides(self, include_inactive: bool = False) -> list[dict]:
        """Returns active graph-edge overrides such as hidden auto edges."""
        try:
            conn = self._connect()
            where = "" if include_inactive else "WHERE active = 1"
            cur = conn.execute(
                f"SELECT * FROM graph_edge_overrides {where} ORDER BY created_at ASC"
            )
            return [dict(row) for row in cur.fetchall()]
        except Exception:
            logger.exception("memory: list_edge_overrides failed")
            return []

    def hide_auto_edge(self, edge_key: str, note: str = "") -> Optional[dict]:
        """Marks an auto-generated graph edge as hidden at render time."""
        edge_key = edge_key.strip()
        if not edge_key:
            return None
        now = time.time()
        cur = self._execute(
            """
            INSERT INTO graph_edge_overrides (edge_key, action, note, active, created_at, updated_at)
            VALUES (?, 'hide', ?, 1, ?, ?)
            ON CONFLICT(edge_key, action) DO UPDATE SET
                note = excluded.note,
                active = 1,
                updated_at = excluded.updated_at
            """,
            (edge_key, note.strip(), now, now),
        )
        if not cur:
            return None
        return self.get_edge_override(edge_key, "hide")

    def get_edge_override(self, edge_key: str, action: str) -> Optional[dict]:
        if action not in _VALID_GRAPH_OVERRIDE_ACTIONS:
            return None
        try:
            conn = self._connect()
            cur = conn.execute(
                "SELECT * FROM graph_edge_overrides WHERE edge_key = ? AND action = ?",
                (edge_key, action),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        except Exception:
            logger.exception("memory: get_edge_override failed")
            return None

    def clear_edge_override(self, edge_key: str, action: str = "hide") -> bool:
        if action not in _VALID_GRAPH_OVERRIDE_ACTIONS:
            return False
        cur = self._execute(
            """
            UPDATE graph_edge_overrides
            SET active = 0, updated_at = ?
            WHERE edge_key = ? AND action = ? AND active = 1
            """,
            (time.time(), edge_key.strip(), action),
        )
        return bool(cur and cur.rowcount > 0)

    # ── Manual node CRUD ─────────────────────────────────────────────────────────

    def add_manual_node(
        self,
        label: str,
        type: str = "concept",
        note: str = "",
        confidence: int = 80,
    ) -> Optional[dict]:
        """Persistiert einen manuellen Graph-Knoten.

        Upsert-Verhalten: wenn ein Knoten mit gleichem node_key bereits existiert
        (auch soft-deleted), wird er reaktiviert und aktualisiert.
        Gibt die gespeicherte Row zurück oder None bei Fehler.
        """
        label = label.strip()
        if not label:
            return None
        node_key = _normalize_graph_key(label)
        if not node_key:
            return None
        if type not in _VALID_NODE_TYPES:
            type = "concept"
        confidence = max(0, min(100, int(confidence)))
        now = time.time()
        cur = self._execute(
            """
            INSERT INTO graph_manual_nodes
                (node_key, label, type, note, confidence, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(node_key) DO UPDATE SET
                label      = excluded.label,
                type       = excluded.type,
                note       = excluded.note,
                confidence = excluded.confidence,
                active     = 1,
                updated_at = excluded.updated_at
            """,
            (node_key, label, type, note.strip(), confidence, now, now),
        )
        if not cur:
            return None
        return self.get_manual_node(int(cur.lastrowid))

    def get_manual_node(self, node_id: int) -> Optional[dict]:
        """Gibt einen manuellen Knoten per ID zurück (aktiv oder inaktiv)."""
        try:
            conn = self._connect()
            cur = conn.execute(
                "SELECT * FROM graph_manual_nodes WHERE id = ?", (node_id,)
            )
            row = cur.fetchone()
            return _node_row_to_dict(row) if row else None
        except Exception:
            logger.exception("memory: get_manual_node failed")
            return None

    def list_manual_nodes(self, include_inactive: bool = False) -> list[dict]:
        """Gibt persistierte manuelle Graph-Knoten zurück."""
        try:
            conn = self._connect()
            where = "" if include_inactive else "WHERE active = 1"
            cur = conn.execute(
                f"SELECT * FROM graph_manual_nodes {where} ORDER BY created_at ASC"
            )
            return [_node_row_to_dict(row) for row in cur.fetchall()]
        except Exception:
            logger.exception("memory: list_manual_nodes failed")
            return []

    def update_manual_node(self, node_id: int, updates: dict) -> Optional[dict]:
        """Aktualisiert veränderbare Felder eines manuellen Knotens.

        Erlaubte Keys: label, type, note, confidence.
        Gibt die aktualisierte Row zurück oder None wenn nicht gefunden.
        """
        allowed = {"label", "type", "note", "confidence"}
        fields: list[str] = []
        values: list = []
        for key, value in updates.items():
            if key not in allowed or value is None:
                continue
            if key == "label":
                value = str(value).strip()
                if not value:
                    continue
            elif key == "type":
                value = str(value).strip()
                if value not in _VALID_NODE_TYPES:
                    value = "concept"
            elif key == "note":
                value = str(value).strip()
            elif key == "confidence":
                value = max(0, min(100, int(value)))
            fields.append(f"{key} = ?")
            values.append(value)
        if not fields:
            return self.get_manual_node(node_id)
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(node_id)
        cur = self._execute(
            f"UPDATE graph_manual_nodes SET {', '.join(fields)} WHERE id = ? AND active = 1",
            tuple(values),
        )
        if not cur or cur.rowcount == 0:
            return None
        return self.get_manual_node(node_id)

    def delete_manual_node(self, node_id: int) -> bool:
        """Soft-löscht einen manuellen Knoten (setzt active=0).

        Gibt True zurück wenn ein aktiver Knoten gefunden und deaktiviert wurde.
        """
        cur = self._execute(
            "UPDATE graph_manual_nodes SET active = 0, updated_at = ? WHERE id = ? AND active = 1",
            (time.time(), node_id),
        )
        return bool(cur and cur.rowcount > 0)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Gibt die globale MemoryManager-Instanz zurück (lazy init)."""
    global _instance
    if _instance is None:
        # Import here to avoid circular imports; config is a plain module with no server deps
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        try:
            import config as _cfg
            db_path = _cfg.SESSIONS_DB_PATH
            md_path = _cfg.MEMORY_MD_PATH
        except Exception:
            db_path = "/opt/airpi/sessions.db"
            md_path = "/opt/airpi/memory.md"
        _instance = MemoryManager(db_path=db_path, memory_md_path=md_path)
    return _instance
