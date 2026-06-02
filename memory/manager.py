"""AirPI Memory Manager — persistente Fakten in SQLite + human-readable memory.md.

Kein Import aus server.py oder model_manager.py (kein Import-Loop).
Thread-safe: check_same_thread=False + try/except um alle DB-Calls.
"""

from __future__ import annotations

import logging
import os
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


def _today() -> str:
    return date.today().isoformat()


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
        return self._conn

    def _init_db(self) -> None:
        try:
            conn = self._connect()
            conn.execute(_SCHEMA)
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
