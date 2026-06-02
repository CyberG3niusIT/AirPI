"""AirPI Memory Graph Builder — extrahiert Konzepte und Verbindungen aus Memory-Einträgen.

Kein ML, kein NLTK, kein numpy — nur stdlib Python.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


# ── Stop-Words (Deutsch + Englisch) ──────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    # ── Deutsche Personalpronomen ──
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mich", "dich", "sich", "uns", "euch",
    "mir", "dir", "ihm", "ihnen", "ihn",
    # ── Possessivpronomen ──
    "mein", "meine", "meinen", "meinem", "meiner",
    "dein", "deine", "deinen", "deinem", "deiner",
    "sein", "seine", "seinen", "seinem", "seiner",
    "ihr", "ihre", "ihren", "ihrem", "ihrer",
    "unser", "unsere", "unseren", "unserem", "unserer",
    "euer", "eure", "euren", "eurem", "eurer",
    # ── Artikel ──
    "der", "die", "das", "dem", "den", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
    # ── Fragewörter ──
    "was", "wer", "wie", "wo", "wann", "warum", "weshalb",
    "welche", "welcher", "welches", "welchem", "welchen",
    "wen", "wem",
    # ── Hilfsverben / Modalverben ──
    "bin", "bist", "ist", "sind", "seid",
    "war", "waren", "wäre", "wären", "sei",
    "habe", "hast", "hat", "haben", "habt", "hatte", "hatten",
    "werde", "wirst", "wird", "werden", "werdet", "wurde", "wurden",
    "soll", "sollst", "sollen", "sollt", "sollte", "sollten",
    "kann", "kannst", "können", "könnt", "konnte", "konnten", "könnte",
    "muss", "musst", "müssen", "müsst", "musste", "mussten", "müsste",
    "darf", "darfst", "dürfen", "dürft", "durfte", "durften",
    "mag", "magst", "mögen", "mögt", "mochte", "mochten", "möchte",
    "will", "willst", "wollen", "wollt", "wollte", "wollten", "würde",
    # ── Häufige Vollverben ──
    "gibt", "geben", "nehmen", "kommen", "gehen", "sehen",
    "wissen", "denken", "glauben", "meinen", "sagen", "machen",
    "gemacht", "gesagt", "geworden", "worden",
    "läuft", "laufen",
    "wohne", "wohnst", "wohnt", "wohnen", "wohnte", "wohnten",
    "arbeite", "arbeitest", "arbeitet", "arbeiten", "arbeitete",
    "heisse", "heißt", "heissen", "heiße", "heißen",
    "lebe", "lebst", "lebt", "leben", "lebte",
    "nutze", "nutzt", "nutzen", "nutzst", "nutzte",
    "spreche", "sprichst", "spricht", "sprechen",
    "schreibe", "schreibst", "schreibt", "schreiben",
    "installiert", "installiere", "installieren",
    # ── Präpositionen / Konjunktionen ──
    "und", "oder", "aber", "doch", "jedoch",
    "mit", "auf", "für", "von", "aus", "bei", "nach", "seit", "vor",
    "über", "unter", "neben", "zwischen", "durch", "ohne", "gegen",
    "in", "an", "am", "im", "ins", "zum", "zur", "zu", "bis", "als",
    "wenn", "dann", "dass", "damit", "weil", "da", "ob", "obwohl",
    "beim", "vom", "ans", "aufs", "ums", "fürs", "hinter",
    # ── Allgemeine/umgangssprachliche Wörter ──
    "name", "namen", "bitte", "hallo", "tschüss", "danke",
    "okay", "ok", "ja", "nein",
    "mal", "noch", "schon", "auch", "nur", "halt", "eben", "eigentlich",
    "mehr", "sehr", "alle", "alles",
    "dieser", "dieses", "diese", "diesem", "diesen",
    "jeder", "jedes", "jede", "jedem", "jeden",
    "immer", "nie", "bereits", "erst",
    "hier", "dort", "nun", "jetzt", "heute", "morgen", "gestern",
    "so", "gut", "neu", "alt", "groß", "klein",
    "kein", "keine", "keinen", "keinem",
    # ── Englische Pronomen / Possessiva ──
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    # ── Englische Artikel / Konjunktionen / Präpositionen ──
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how",
    # ── Englische Fragewörter / Pronomen ──
    "what", "which", "who", "whom", "this", "that", "these", "those",
    # ── Englische Hilfsverben ──
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "can", "will", "should", "would", "could", "may", "might",
    "shall", "must", "ought",
    # ── Englische allgemeine Wörter ──
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "also", "now", "new", "old", "good",
    "name", "please", "hello", "yes",
    "use", "used", "using", "make", "made", "making",
    "get", "got", "getting", "set", "sets", "setting",
    "run", "runs", "running", "like", "likes", "liked",
    "know", "knows", "known", "need", "needs", "needed",
    "want", "wants", "wanted",
    "true", "false", "none", "null", "auto", "via", "etc",
    "user", "users",
})

# Blacklist für generische Tech-/Gesprächswörter, die trotz Länge keine
# sinnvollen Konzept-Knoten ergeben (verbinden sonst alles mit allem).
_GENERIC_BLACKLIST: frozenset[str] = frozenset({
    "Benutzer", "System", "Server", "Antwort", "Frage", "Hinweis",
    "Schritt", "Punkt", "Liste", "Beispiel", "Bereich", "Methode",
    "Funktion", "Wert", "Datei", "Ordner", "Fehler", "Problem",
    "Lösung", "Ergebnis", "Information", "Version", "Update",
    "Status", "Zustand", "Prozess", "Service",
})
_GENERIC_BLACKLIST_LOWER: frozenset[str] = frozenset(w.lower() for w in _GENERIC_BLACKLIST)

# Tokenisierungs-Regex: Buchstaben (inkl. Umlaute) ODER alphanumerische
# Tokens mit Bindestrich (z.B. "IT-Forensiker", "IPv4", "Pi5")
_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9][A-Za-zÄÖÜäöüß0-9\-]*[A-Za-zÄÖÜäöüß0-9]|[A-Za-zÄÖÜäöüß0-9]")

# Mindestlänge eines Tokens (in Zeichen) um als Konzept in Frage zu kommen
_MIN_LEN = 5


def _score_token(tok: str) -> int:
    """Berechnet einen Relevanz-Score für ein Token.

    Kriterien:
    - CamelCase oder vollständig Großbuchstaben → +2 (Eigenname/Fachbegriff)
    - Enthält Ziffern (z.B. "Pi5", "IPv4") → +1
    - Vorkommen in mehreren Einträgen → +1 pro Eintrag (wird in build() addiert)
    """
    score = 0
    # CamelCase: mind. ein Großbuchstabe nach erstem Zeichen
    if re.search(r"[A-ZÄÖÜ]", tok[1:]):
        score += 2
    # Vollständig groß
    elif tok.isupper() and len(tok) > 1:
        score += 2
    # Enthält Ziffern
    if re.search(r"[0-9]", tok):
        score += 1
    return score


def _extract_concepts(text: str, max_concepts: int = 6) -> list[str]:
    """Extrahiert Konzepte aus einem Text-Eintrag.

    Algorithmus:
    1. Tokenisiere nach Wörtern (Buchstaben + Ziffern + Bindestrich)
    2. Entferne Stop-Words (case-insensitive) und generische Blacklist
    3. Behalte nur Tokens mit Mindestlänge _MIN_LEN
    4. Behalte Tokens mit Großbuchstaben am Anfang ODER Tokens > _MIN_LEN Zeichen
    5. Score: CamelCase/Großschreibung +2, Ziffern +1
    6. Dedupliziere (case-insensitive, behalte erste Schreibweise)
    7. Sortiere nach Score absteigend, dann Länge absteigend
    8. Gib max max_concepts zurück
    """
    tokens = _TOKEN_RE.findall(text)
    seen_lower: dict[str, tuple[str, int]] = {}  # lower → (first_occurrence, score)

    for tok in tokens:
        lower = tok.lower()

        # Stop-Words zuerst prüfen (vor Längencheck, erfasst auch kurze Stop-Words)
        if lower in _STOP_WORDS:
            continue

        # Generische Blacklist
        if lower in _GENERIC_BLACKLIST_LOWER:
            continue

        # Mindestlänge:
        # - Großgeschriebene Tokens (Eigennamen) → mind. 3 Zeichen
        # - Alles andere → mind. _MIN_LEN Zeichen (filtert Rauschen aggressiv)
        is_capitalized = tok[0].isupper()
        min_len = 3 if is_capitalized else _MIN_LEN
        if len(tok) < min_len:
            continue

        score = _score_token(tok)

        if lower not in seen_lower:
            seen_lower[lower] = (tok, score)

    candidates = [(display, score) for display, score in seen_lower.values()]

    # Sortiere: Score absteigend, dann Länge absteigend
    candidates.sort(key=lambda x: (-x[1], -len(x[0])))

    return [display for display, _ in candidates[:max_concepts]]


class GraphBuilder:
    """Baut einen Knoten-Kanten-Graphen aus Memory-Einträgen."""

    def build(self, entries: list[dict]) -> dict:
        """
        Input: Liste von Memory-Einträgen mit 'content' Feld (aus MemoryManager.all_active())
        Output: {"nodes": [...], "edges": [...], "meta": {...}}
        """
        # Konzepte pro Eintrag extrahieren
        entry_concepts: list[list[str]] = []
        now = time.time()
        for entry in entries:
            content = entry.get("content", "")
            concepts = _extract_concepts(content)
            entry_concepts.append(concepts)

        # Knoten-Tracking: concept_lower → {"display": str, "entries": set, "degree": int, "base_score": int, ...meta}
        node_map: dict[str, dict] = {}

        # Kanten-Tracking: frozenset({a_lower, b_lower}) → {"count": int}
        edge_map: defaultdict[frozenset, int] = defaultdict(int)

        for idx, concepts in enumerate(entry_concepts):
            entry = entries[idx]
            entry_id = entry.get("id")
            entry_category = entry.get("category", "fact") or "fact"
            entry_source = entry.get("source", "unknown") or "unknown"
            # last_seen_at: use COALESCE-style fallback (field may not exist yet)
            raw_last_seen = entry.get("last_seen_at") or entry.get("created_at") or 0
            try:
                raw_last_seen = float(raw_last_seen)
            except (TypeError, ValueError):
                raw_last_seen = 0.0
            # confidence: COALESCE(confidence, 80)
            raw_confidence = entry.get("confidence")
            try:
                entry_confidence = int(raw_confidence) if raw_confidence is not None else 80
            except (TypeError, ValueError):
                entry_confidence = 80

            for concept in concepts:
                lower = concept.lower()
                if lower not in node_map:
                    node_map[lower] = {
                        "display": concept,
                        "entries": set(),      # set of entry indices
                        "memory_ids": [],      # list of DB ids
                        "degree": 0,
                        "base_score": _score_token(concept),
                        # Aggregated meta — taken from first / majority entry
                        "category": entry_category,
                        "source": entry_source,
                        "last_seen_at": raw_last_seen,
                        "confidence_sum": entry_confidence,
                        "confidence_count": 1,
                    }
                else:
                    # Update aggregated stats
                    if raw_last_seen > node_map[lower]["last_seen_at"]:
                        node_map[lower]["last_seen_at"] = raw_last_seen
                    node_map[lower]["confidence_sum"] += entry_confidence
                    node_map[lower]["confidence_count"] += 1
                node_map[lower]["entries"].add(idx)
                if entry_id is not None and entry_id not in node_map[lower]["memory_ids"]:
                    node_map[lower]["memory_ids"].append(entry_id)

            # Kanten zwischen allen Konzepten desselben Eintrags
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    a = concepts[i].lower()
                    b = concepts[j].lower()
                    if a != b:
                        edge_map[frozenset({a, b})] += 1

        # Degree (Anzahl der Kanten) pro Knoten berechnen
        for pair in edge_map:
            for node_lower in pair:
                if node_lower in node_map:
                    node_map[node_lower]["degree"] += 1

        # Multi-Entry-Bonus: +1 pro Eintrag in dem das Konzept vorkommt
        # (wird auf base_score addiert, beeinflusst weight im Output)
        for lower, data in node_map.items():
            data["entry_bonus"] = len(data["entries"])

        # Isolierte Knoten filtern wenn mehr als 20 Knoten vorhanden
        total_nodes = len(node_map)
        if total_nodes > 20:
            node_map = {k: v for k, v in node_map.items() if v["degree"] >= 1}

        # Kanten auf bekannte Knoten beschränken
        valid_nodes = set(node_map.keys())

        # Output aufbauen
        nodes = []
        for lower, data in node_map.items():
            last_seen = data.get("last_seen_at", 0) or 0
            if last_seen > 0:
                age_days = round((now - last_seen) / 86400, 1)
            else:
                age_days = 30.0
            # Average confidence across entries that mention this concept
            conf_count = data.get("confidence_count", 1) or 1
            confidence = round(data.get("confidence_sum", 80) / conf_count)

            nodes.append({
                "id": data["display"],
                # Gewicht = Kanten-Degree + base_score + entry_bonus
                "weight": (data["degree"] + data["base_score"] + data.get("entry_bonus", 0))
                          if data["degree"] > 0
                          else (1 + data["base_score"] + data.get("entry_bonus", 0)),
                "entries": sorted(data["entries"]),
                # QW-7: extended node meta
                "category": data.get("category", "fact"),
                "source": data.get("source", "unknown"),
                "age_days": age_days,
                "confidence": confidence,
                "memory_ids": sorted(data.get("memory_ids", [])),
            })

        edges = []
        for pair, count in edge_map.items():
            pair_list = list(pair)
            if len(pair_list) != 2:
                continue
            a_lower, b_lower = pair_list[0], pair_list[1]
            if a_lower not in valid_nodes or b_lower not in valid_nodes:
                continue
            edges.append({
                "source": node_map[a_lower]["display"],
                "target": node_map[b_lower]["display"],
                "weight": count,
            })

        # Nach Gewicht sortieren (stärkste Verbindungen zuerst)
        nodes.sort(key=lambda n: -n["weight"])
        edges.sort(key=lambda e: -e["weight"])

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "total_entries": len(entries),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }


if __name__ == "__main__":
    entries = [
        {"content": "Mein Name ist Alex, ich bin IT-Forensiker und arbeite mit Raspberry Pi 5", "source": "user"},
        {"content": "AirPI läuft auf dem Raspberry Pi 5 mit 8GB RAM", "source": "user"},
        {"content": "Home Assistant ist als Docker Container installiert", "source": "auto"},
    ]
    g = GraphBuilder().build(entries)
    print("Nodes:", [n["id"] for n in g["nodes"]])
    print("Edges:", len(g["edges"]))
