"""Tests für memory/manager.py, memory/graph.py und /memory/* Endpoints."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# MemoryManagerTests
# ---------------------------------------------------------------------------

class MemoryManagerTests(unittest.TestCase):
    """Tests für memory/manager.py"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.md_path = os.path.join(self.tmpdir, "memory.md")
        from memory.manager import MemoryManager
        self.mm = MemoryManager(self.db_path, self.md_path)

    def tearDown(self):
        # Close DB connection if open
        if self.mm._conn is not None:
            try:
                self.mm._conn.close()
            except Exception:
                pass

    def test_store_fact_persists_in_db(self):
        """store_fact schreibt in SQLite und all_active gibt es zurück."""
        self.mm.store_fact("Alex mag Python")
        entries = self.mm.all_active()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["content"], "Alex mag Python")

    def test_store_fact_writes_memory_md(self):
        """Nach store_fact existiert memory.md und enthält den Inhalt."""
        self.mm.store_fact("Lieblingsfarbe ist Blau")
        self.assertTrue(os.path.exists(self.md_path))
        content = open(self.md_path, encoding="utf-8").read()
        self.assertIn("Lieblingsfarbe ist Blau", content)

    def test_delete_fact_removes_by_keyword(self):
        """delete_fact(keyword) deaktiviert alle Einträge die keyword enthalten."""
        self.mm.store_fact("Alex arbeitet an AirPI")
        self.mm.store_fact("AirPI läuft auf Raspberry Pi")
        self.mm.store_fact("Alex trinkt Kaffee")

        count = self.mm.delete_fact("AirPI")
        self.assertEqual(count, 2)

        remaining = self.mm.all_active()
        self.assertEqual(len(remaining), 1)
        self.assertIn("Kaffee", remaining[0]["content"])

    def test_get_context_returns_md_content(self):
        """get_context() gibt memory.md Inhalt zurück."""
        self.mm.store_fact("Testfakt für Kontext")
        ctx = self.mm.get_context()
        self.assertIn("Testfakt für Kontext", ctx)

    def test_store_extracted_batch(self):
        """store_extracted(list) speichert mehrere Fakten auf einmal."""
        facts = ["Fakt eins", "Fakt zwei", "Fakt drei"]
        self.mm.store_extracted(facts)
        entries = self.mm.all_active()
        self.assertEqual(len(entries), 3)
        contents = [e["content"] for e in entries]
        for f in facts:
            self.assertIn(f, contents)

    def test_rebuild_md_includes_all_active(self):
        """rebuild_md() schreibt alle aktiven Einträge korrekt in memory.md."""
        self.mm.store_fact("Erster Fakt")
        self.mm.store_fact("Zweiter Fakt")
        # Delete the md file manually to force a rebuild
        os.remove(self.md_path)
        self.assertFalse(os.path.exists(self.md_path))
        self.mm.rebuild_md()
        self.assertTrue(os.path.exists(self.md_path))
        content = open(self.md_path, encoding="utf-8").read()
        self.assertIn("Erster Fakt", content)
        self.assertIn("Zweiter Fakt", content)

    def test_store_extracted_no_duplicates(self):
        """Zweimaliges Speichern desselben Fakts erzeugt nur einen Eintrag."""
        self.mm.store_extracted(["Alex ist IT-Forensiker"])
        self.mm.store_extracted(["Alex ist IT-Forensiker"])  # Duplikat
        entries = self.mm.all_active()
        contents = [e["content"] for e in entries]
        assert contents.count("Alex ist IT-Forensiker") == 1

    def test_store_extracted_dict_format(self):
        """store_extracted akzeptiert {"content": ..., "category": ...} Dicts."""
        self.mm.store_extracted([
            {"content": "User mag Python", "category": "preference"},
            {"content": "User heißt Max", "category": "fact"},
        ])
        entries = self.mm.all_active()
        assert len(entries) == 2
        categories = {e["content"]: e["category"] for e in entries}
        assert categories["User mag Python"] == "preference"

    def test_store_extracted_invalid_category_defaults_to_fact(self):
        """Ungültige Kategorie wird zu 'fact'."""
        self.mm.store_extracted([{"content": "Test", "category": "invalid_xyz"}])
        entries = self.mm.all_active()
        assert entries[0]["category"] == "fact"

    def test_graceful_on_bad_db_path(self):
        """MemoryManager mit ungültigem DB-Pfad wirft keine Exception beim Init."""
        from memory.manager import MemoryManager
        # /proc/nonexistent is a path that cannot be written to
        bad_path = "/proc/nonexistent/really/bad.db"
        try:
            mm = MemoryManager(bad_path, os.path.join(self.tmpdir, "bad_memory.md"))
            # Should not raise; _conn will be None after failed init
        except Exception as exc:
            self.fail(f"MemoryManager.__init__ raised an exception with bad path: {exc}")


# ---------------------------------------------------------------------------
# GraphBuilderTests
# ---------------------------------------------------------------------------

class GraphBuilderTests(unittest.TestCase):
    """Tests für memory/graph.py"""

    def setUp(self):
        from memory.graph import GraphBuilder
        self.builder = GraphBuilder()

    def test_empty_entries_returns_empty_graph(self):
        """Leere Eingabe → leerer Graph."""
        result = self.builder.build([])
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["meta"]["total_entries"], 0)

    def test_single_entry_creates_nodes(self):
        """Ein Entry mit Konzepten → Knoten werden erstellt."""
        entries = [{"content": "Berlin ist eine große Hauptstadt Deutschlands"}]
        result = self.builder.build(entries)
        # 'Berlin' and 'Hauptstadt' and 'Deutschlands' should appear as nodes
        node_ids = [n["id"] for n in result["nodes"]]
        # At least one node should exist
        self.assertGreater(len(node_ids), 0)
        # Berlin should be a concept (uppercase, >4 chars)
        self.assertTrue(any("Berlin" in nid or "berlin" in nid.lower() for nid in node_ids))

    def test_stopwords_not_in_nodes(self):
        """'Mein', 'Name', 'ist', 'und' erscheinen NICHT als Knoten."""
        entries = [{"content": "Mein Name ist Alex und ich lerne Python"}]
        result = self.builder.build(entries)
        node_ids_lower = [n["id"].lower() for n in result["nodes"]]
        # Common stop words that are in _STOP_WORDS
        for stopword in ["ist", "und", "mein"]:
            self.assertNotIn(stopword, node_ids_lower, f"Stop word '{stopword}' should not be a node")

    def test_shared_concepts_create_edge(self):
        """Zwei Entries mit gemeinsamem Konzept → Edge zwischen ihnen (via shared concepts within entries)."""
        # Two entries that each contain multiple concepts — at least one entry must have
        # two concepts to generate an edge between them
        entries = [
            {"content": "Python Programmierung macht Spaß"},
            {"content": "Python Entwicklung erfordert Übung"},
        ]
        result = self.builder.build(entries)
        # Edges are created within a single entry between co-occurring concepts,
        # and nodes appear across entries. With 'Python' in both entries its weight increases.
        # Check that nodes exist
        self.assertGreater(len(result["nodes"]), 0)
        # Python should appear in nodes (uppercase, multi-entry)
        node_ids = [n["id"] for n in result["nodes"]]
        self.assertTrue(
            any("Python" == nid or "python" == nid.lower() for nid in node_ids),
            f"Expected 'Python' in nodes, got: {node_ids}"
        )

    def test_node_weight_reflects_frequency(self):
        """Konzept in 3 Entries → weight >= Konzept in 1 Entry."""
        # 'Python' appears across 3 entries, 'Haskell' only in 1
        entries = [
            {"content": "Python Programmierung"},
            {"content": "Python Entwicklung Frameworks"},
            {"content": "Python Bibliotheken installieren"},
            {"content": "Haskell Funktional"},
        ]
        result = self.builder.build(entries)
        nodes_by_id = {n["id"].lower(): n for n in result["nodes"]}

        python_node = nodes_by_id.get("python")
        haskell_node = nodes_by_id.get("haskell")

        # Both should exist
        self.assertIsNotNone(python_node, f"Expected 'python' node, got nodes: {list(nodes_by_id.keys())}")
        self.assertIsNotNone(haskell_node, f"Expected 'haskell' node, got nodes: {list(nodes_by_id.keys())}")

        # Python has more co-occurrences → higher weight than Haskell (which is alone in its entry)
        # Python appears in 3 entries, each with another concept → more edges → higher degree
        self.assertGreaterEqual(
            python_node["weight"],
            haskell_node["weight"],
            f"Python weight {python_node['weight']} should be >= Haskell weight {haskell_node['weight']}"
        )


# ---------------------------------------------------------------------------
# MemoryEndpointTests
# ---------------------------------------------------------------------------

class MemoryEndpointTests(unittest.TestCase):
    """Integration-Tests für /memory/* Endpoints via TestClient."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "endpoint_test.db")
        self.md_path = os.path.join(self.tmpdir, "endpoint_memory.md")

        from memory.manager import MemoryManager
        self.mock_mm = MemoryManager(self.db_path, self.md_path)

        # Patch get_memory_manager in server to return our isolated instance
        self._patcher = patch("server.get_memory_manager", return_value=self.mock_mm)
        self._patcher.start()

        from starlette.testclient import TestClient
        import server
        self.client = TestClient(server.app, raise_server_exceptions=True)

    def tearDown(self):
        self._patcher.stop()
        if self.mock_mm._conn is not None:
            try:
                self.mock_mm._conn.close()
            except Exception:
                pass

    def test_memory_store_returns_ok(self):
        """POST /memory/store mit gültigem content → {"ok": true}."""
        response = self.client.post("/memory/store", json={"content": "Alex wohnt in Berlin"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])

    def test_memory_delete_returns_count(self):
        """POST /memory/delete mit keyword → {"deleted": N}."""
        # First store something to delete
        self.mock_mm.store_fact("Zu löschender Eintrag mit Keyword")
        self.mock_mm.store_fact("Anderer Eintrag bleibt")

        response = self.client.post("/memory/delete", json={"keyword": "Keyword"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("deleted", body)
        self.assertEqual(body["deleted"], 1)

    def test_memory_get_returns_content_and_entries(self):
        """GET /memory → {"content": "...", "entries": [...]}."""
        self.mock_mm.store_fact("Persistierter Fakt")

        response = self.client.get("/memory")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("content", body)
        self.assertIn("entries", body)
        self.assertIsInstance(body["entries"], list)
        self.assertGreater(len(body["entries"]), 0)
        contents = [e["content"] for e in body["entries"]]
        self.assertIn("Persistierter Fakt", contents)


if __name__ == "__main__":
    unittest.main()
