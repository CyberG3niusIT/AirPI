"""Tests für manuelle Graph-Knoten (CRUD) und NER-Bug-Fixes."""

from __future__ import annotations

import os
import tempfile
import unittest

from memory.graph import GraphBuilder, merge_graph_overlays
from memory.manager import MemoryManager


class ManualNodeManagerTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.db")
        self.md_path = os.path.join(self.tmpdir.name, "memory.md")
        self.mem = MemoryManager(self.db_path, self.md_path)

    def tearDown(self):
        if self.mem._conn is not None:
            self.mem._conn.close()
        self.tmpdir.cleanup()

    def test_add_manual_node_creates_db_row(self):
        """add_manual_node gibt ein dict mit id, node_key, label, type, active zurück."""
        node = self.mem.add_manual_node("Testknoten", type="concept", confidence=75)
        self.assertIsNotNone(node)
        self.assertEqual(node["label"], "Testknoten")
        self.assertEqual(node["node_key"], "testknoten")
        self.assertEqual(node["type"], "concept")
        self.assertTrue(node["active"])
        self.assertIn("id", node)

    def test_list_manual_nodes_returns_only_active(self):
        """list_manual_nodes() schließt soft-deleted Zeilen aus."""
        n1 = self.mem.add_manual_node("Alpha")
        n2 = self.mem.add_manual_node("Beta")
        self.mem.delete_manual_node(n1["id"])
        nodes = self.mem.list_manual_nodes()
        keys = [n["node_key"] for n in nodes]
        self.assertNotIn("alpha", keys)
        self.assertIn("beta", keys)

    def test_update_manual_node_fields(self):
        """update_manual_node ändert nur die angegebenen Felder."""
        node = self.mem.add_manual_node("Gamma", type="concept", confidence=60)
        updated = self.mem.update_manual_node(node["id"], {"confidence": 90, "type": "tech"})
        self.assertIsNotNone(updated)
        self.assertEqual(updated["confidence"], 90)
        self.assertEqual(updated["type"], "tech")
        self.assertEqual(updated["label"], "Gamma")  # unverändert

    def test_delete_manual_node_soft_delete(self):
        """delete_manual_node setzt active=0, entfernt aber nicht die DB-Zeile."""
        node = self.mem.add_manual_node("Delta")
        result = self.mem.delete_manual_node(node["id"])
        self.assertTrue(result)
        # Sollte in aktiver Liste nicht erscheinen
        active_ids = [n["id"] for n in self.mem.list_manual_nodes()]
        self.assertNotIn(node["id"], active_ids)
        # Sollte mit include_inactive=True erscheinen
        all_ids = [n["id"] for n in self.mem.list_manual_nodes(include_inactive=True)]
        self.assertIn(node["id"], all_ids)

    def test_delete_nonexistent_node_returns_false(self):
        """delete_manual_node gibt False zurück wenn kein aktiver Knoten gefunden."""
        result = self.mem.delete_manual_node(99999)
        self.assertFalse(result)

    def test_upsert_reactivates_deleted_node(self):
        """add_manual_node reaktiviert einen gelöschten Knoten mit gleichem Label."""
        node = self.mem.add_manual_node("ReactivateMe")
        self.mem.delete_manual_node(node["id"])
        self.assertEqual(self.mem.list_manual_nodes(), [])
        # Erneutes Anlegen → reaktiviert
        node2 = self.mem.add_manual_node("ReactivateMe", type="tech")
        self.assertIsNotNone(node2)
        active = self.mem.list_manual_nodes()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["type"], "tech")


class ManualNodeGraphMergeTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.db")
        self.md_path = os.path.join(self.tmpdir.name, "memory.md")
        self.mem = MemoryManager(self.db_path, self.md_path)

    def tearDown(self):
        if self.mem._conn is not None:
            self.mem._conn.close()
        self.tmpdir.cleanup()

    def test_manual_node_injected_into_graph(self):
        """Ein manueller Knoten ohne Auto-Pendant wird mit origin='manual' injiziert."""
        self.mem.add_manual_node("OrphanNode", type="tech", confidence=85)
        graph = GraphBuilder().build([])  # leerer Auto-Graph
        merged = merge_graph_overlays(
            graph,
            manual_nodes=self.mem.list_manual_nodes(),
        )
        keys = [n.get("key") or n.get("id", "").lower() for n in merged["nodes"]]
        self.assertIn("orphannode", keys)
        manual_node = next(
            n for n in merged["nodes"]
            if (n.get("key") or "") == "orphannode"
        )
        self.assertEqual(manual_node["origin"], "manual")
        self.assertEqual(manual_node["type"], "tech")
        self.assertEqual(merged["meta"]["manual_nodes"], 1)

    def test_manual_node_merged_with_auto_node(self):
        """Ein manueller Knoten dessen Key einem Auto-Knoten entspricht → origin='mixed'."""
        entries = [{
            "id": 1,
            "content": "Python ist eine Programmiersprache",
            "category": "fact",
            "confidence": 80,
            "last_seen_at": 1748000000.0,
        }]
        graph = GraphBuilder().build(entries)
        self.assertGreater(len(graph["nodes"]), 0)

        # Den ersten Knoten nehmen der ein "key" Feld hat
        target_node = next(
            (n for n in graph["nodes"] if n.get("key") and n.get("key") != "__user__"),
            None,
        )
        self.assertIsNotNone(target_node, "Kein Auto-Knoten mit key-Feld gefunden")
        auto_key = target_node["key"]
        auto_label = target_node["id"]

        # Manuellen Knoten mit gleichem Label anlegen
        self.mem.add_manual_node(auto_label, type="concept")
        merged = merge_graph_overlays(
            graph,
            manual_nodes=self.mem.list_manual_nodes(),
        )
        # Knoten per key suchen (eindeutig)
        matched = [n for n in merged["nodes"] if n.get("key") == auto_key]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["origin"], "mixed")

    def test_soft_deleted_manual_node_not_injected(self):
        """Inaktive manuelle Knoten erscheinen nicht im gemergten Graphen."""
        node = self.mem.add_manual_node("GhostNode")
        self.mem.delete_manual_node(node["id"])
        graph = GraphBuilder().build([])
        merged = merge_graph_overlays(
            graph,
            manual_nodes=self.mem.list_manual_nodes(),
        )
        keys = [n.get("key") or "" for n in merged["nodes"]]
        self.assertNotIn("ghostnode", keys)
        self.assertEqual(merged["meta"].get("manual_nodes", 0), 0)

    def test_merge_without_manual_nodes_param_backward_compat(self):
        """merge_graph_overlays ohne manual_nodes-Parameter läuft ohne Fehler."""
        graph = GraphBuilder().build([])
        merged = merge_graph_overlays(graph)
        self.assertIn("nodes", merged)
        self.assertIn("edges", merged)


class NERBugTests(unittest.TestCase):

    def setUp(self):
        self.builder = GraphBuilder()

    def test_ner_no_partial_token_leakage(self):
        """Einzeltokens von Personennamen ('Niko', 'Demo') dürfen NICHT als
        separate Konzeptknoten erscheinen wenn der volle Name erkannt wurde."""
        entries = [{
            "id": 1,
            "content": "ich bin Vater von 2 Kindern: Mira Test (8 Jahre) und Niko Demo (6 Jahre)",
            "category": "fact",
            "confidence": 95,
            "last_seen_at": 1748000000.0,
        }]
        graph = self.builder.build(entries)
        node_ids_lower = {n["id"].lower() for n in graph["nodes"]}
        # Vollständige Namen müssen vorhanden sein
        self.assertIn("mira test", node_ids_lower)
        self.assertIn("niko demo", node_ids_lower)
        # Einzeltokens dürfen NICHT als separate Knoten erscheinen
        self.assertNotIn("niko", node_ids_lower)
        self.assertNotIn("demo", node_ids_lower)
        self.assertNotIn("mira", node_ids_lower)
        self.assertNotIn("test", node_ids_lower)

    def test_mein_name_not_a_node(self):
        """'Mein Name' darf nicht als Relationship-Target-Knoten erscheinen."""
        entries = [{
            "id": 2,
            "content": "Mein Name ist Leon",
            "category": "fact",
            "confidence": 90,
            "last_seen_at": 1748000000.0,
        }]
        graph = self.builder.build(entries)
        node_ids_lower = {n["id"].lower() for n in graph["nodes"]}
        self.assertNotIn("mein name", node_ids_lower)
        self.assertNotIn("name", node_ids_lower)

    def test_2_kindern_not_a_node(self):
        """'2 Kindern' darf nicht als vater_von-Relationship-Target-Knoten erscheinen."""
        entries = [{
            "id": 3,
            "content": "ich bin Vater von 2 Kindern: Anna Müller (5 Jahre) und Ben Müller (3 Jahre)",
            "category": "fact",
            "confidence": 95,
            "last_seen_at": 1748000000.0,
        }]
        graph = self.builder.build(entries)
        node_ids_lower = {n["id"].lower() for n in graph["nodes"]}
        self.assertNotIn("2 kindern", node_ids_lower)
        self.assertNotIn("kindern", node_ids_lower)

    def test_leon_maxim_full_name_extracted(self):
        """Vollständige Kindernamen werden korrekt als Person-Knoten extrahiert."""
        entries = [{
            "id": 4,
            "content": (
                "ich bin Vater von 3 Kindern: Leon Maxim (8 Jahre) Geboren am 05.07.2017 in Pforzheim, "
                "Emilia Grace (6 Jahre) Geboren am 17.10.2019 in Pforzheim, "
                "Elsa Sophie (4 Jahre) Geboren am 05.04.2022 in Göppingen"
            ),
            "category": "fact",
            "confidence": 95,
            "last_seen_at": 1748000000.0,
        }]
        graph = self.builder.build(entries)
        node_ids_lower = {n["id"].lower() for n in graph["nodes"]}
        self.assertIn("leon maxim", node_ids_lower)
        self.assertIn("emilia grace", node_ids_lower)
        self.assertIn("elsa sophie", node_ids_lower)
        # Einzeltokens der Kindernamen dürfen nicht separat erscheinen
        for token in ["leon", "maxim", "emilia", "grace", "elsa", "sophie"]:
            self.assertNotIn(token, node_ids_lower, f"Token '{token}' sollte nicht als Node erscheinen")


if __name__ == "__main__":
    unittest.main()
