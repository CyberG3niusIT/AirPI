import os
import tempfile
import unittest

from memory.graph import GraphBuilder, graph_edge_key, merge_graph_overlays
from memory.manager import MemoryManager


class GraphOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.db")
        self.md_path = os.path.join(self.tmpdir.name, "memory.md")
        self.mem = MemoryManager(self.db_path, self.md_path)

    def tearDown(self):
        if self.mem._conn is not None:
            self.mem._conn.close()
        self.tmpdir.cleanup()

    def test_manual_edge_survives_graph_rebuild(self):
        self.mem.store_fact("Python arbeitet mit FastAPI", source="test")
        edge = self.mem.add_manual_edge(
            source_key="python",
            target_key="fastapi",
            source_label="Python",
            target_label="FastAPI",
            relation_type="uses",
            label="nutzt",
            directed=True,
            confidence=91,
        )
        self.assertIsNotNone(edge)

        graph = GraphBuilder().build(self.mem.all_active())
        merged = merge_graph_overlays(graph, self.mem.list_manual_edges())
        manual_edges = [e for e in merged["edges"] if "manual" in e.get("origins", [])]

        self.assertEqual(len(manual_edges), 1)
        self.assertEqual(manual_edges[0]["label"], "nutzt")
        self.assertEqual(manual_edges[0]["confidence"], 91)

        rebuilt = merge_graph_overlays(
            GraphBuilder().build(self.mem.all_active()),
            self.mem.list_manual_edges(),
        )
        rebuilt_manual = [e for e in rebuilt["edges"] if "manual" in e.get("origins", [])]
        self.assertEqual(len(rebuilt_manual), 1)

    def test_soft_deleted_manual_edge_is_not_merged(self):
        edge = self.mem.add_manual_edge("alpha", "beta", relation_type="related")
        self.assertIsNotNone(edge)
        self.assertTrue(self.mem.delete_manual_edge(edge["id"]))

        graph = {"nodes": [], "edges": [], "meta": {"total_entries": 0}}
        merged = merge_graph_overlays(graph, self.mem.list_manual_edges())

        self.assertEqual(merged["edges"], [])
        self.assertEqual(merged["meta"]["manual_edges"], 0)

    def test_hide_auto_edge_overlay_removes_auto_edge_only_at_render_time(self):
        entries = [{"content": "Python FastAPI"}]
        graph = GraphBuilder().build(entries)
        self.assertGreaterEqual(len(graph["edges"]), 1)
        auto_edge_key = graph["edges"][0]["key"]

        override = self.mem.hide_auto_edge(auto_edge_key, note="manuell ausgeblendet")
        self.assertIsNotNone(override)
        merged = merge_graph_overlays(graph, edge_overrides=self.mem.list_edge_overrides())

        self.assertNotIn(auto_edge_key, [e["key"] for e in merged["edges"]])
        self.assertEqual(merged["meta"]["hidden_auto_edges"], 1)

    def test_undirected_manual_edge_key_is_order_independent(self):
        first = graph_edge_key("Python", "FastAPI", "related", False)
        second = graph_edge_key("fastapi", "python", "related", False)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
