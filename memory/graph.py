"""AirPI Memory Graph Builder v2 — semantische NER + Relationship Extraction.

Kein ML, kein NLTK, kein numpy — nur stdlib Python.

v2 Verbesserungen:
- Named Entity Recognition (NER): Personen, Orte, Daten, Konzepte trennen
- Relationship Extraction: typisierte, gerichtete Kanten (vater_von, geboren_in, etc.)
- Stopwort-Filter für generische Wörter ("Geboren", "Kindern")
- Voller Kontext-Erhalt: "Leon Maxim" bleibt als ganzer Name erhalten
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


def normalize_node_key(value: str) -> str:
    """Stable, display-independent key used by graph overlays."""
    return re.sub(r"\s+", " ", value.strip().lower())


def graph_edge_key(source_key: str, target_key: str, relation_type: str, directed: bool) -> str:
    """Build a stable edge key. Undirected edges use sorted node keys."""
    source = normalize_node_key(source_key)
    target = normalize_node_key(target_key)
    rel = normalize_node_key(relation_type or "co_occurrence")
    if not directed:
        source, target = sorted((source, target))
    return f"{rel}|{source}|{target}|{'1' if directed else '0'}"


# ── Bekannte Städte / Orte (schnelle Lookups) ────────────────────────────────

_KNOWN_PLACES: frozenset[str] = frozenset({
    # Deutschland
    "Berlin", "Hamburg", "München", "Köln", "Frankfurt", "Stuttgart",
    "Düsseldorf", "Dortmund", "Essen", "Leipzig", "Bremen", "Dresden",
    "Hannover", "Nürnberg", "Duisburg", "Bochum", "Wuppertal", "Bielefeld",
    "Bonn", "Münster", "Karlsruhe", "Mannheim", "Augsburg", "Wiesbaden",
    "Gelsenkirchen", "Mönchengladbach", "Braunschweig", "Kiel", "Aachen",
    "Halle", "Magdeburg", "Freiburg", "Krefeld", "Lübeck", "Mainz",
    "Erfurt", "Rostock", "Kassel", "Hagen", "Hamm", "Saarbrücken",
    "Mülheim", "Potsdam", "Ludwigshafen", "Leverkusen", "Osnabrück",
    "Solingen", "Heidelberg", "Herne", "Neuss", "Regensburg", "Paderborn",
    "Ingolstadt", "Würzburg", "Fürth", "Wolfsburg", "Offenbach",
    "Ulm", "Heilbronn", "Pforzheim", "Göppingen", "Esslingen", "Böblingen",
    "Tübingen", "Reutlingen", "Konstanz", "Ravensburg", "Friedrichshafen",
    "Landsberg", "Hochdorf", "Kirchheim", "Wendlingen",
    # Österreich
    "Wien", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt",
    # Schweiz
    "Zürich", "Bern", "Basel", "Lausanne", "Genf", "Winterthur",
    # Landkreise / Regionen
    "Landkreis", "Kreis", "Bezirk", "Baden-Württemberg", "Bayern",
    "Nordrhein-Westfalen", "Niedersachsen", "Sachsen", "Hessen",
})

_KNOWN_PLACES_LOWER: dict[str, str] = {p.lower(): p for p in _KNOWN_PLACES}


# ── Beziehungs-Patterns ──────────────────────────────────────────────────────

# Jeder Eintrag: (regex_pattern, relationship_type, group_subject, group_object)
# group=0 → ganzer Match, group=1..N → Capture-Group
_RELATION_PATTERNS: list[tuple[re.Pattern, str, int, int]] = [
    # "ich bin Vater von X" / "bin Vater von"
    (re.compile(r"(?:ich\s+)?bin\s+(?:der\s+)?Vater\s+von\s+(.+?)(?:\s*[:,]|$)", re.I), "vater_von", -1, 1),
    # "ich bin Mutter von X"
    (re.compile(r"(?:ich\s+)?bin\s+(?:die\s+)?Mutter\s+von\s+(.+?)(?:\s*[:,]|$)", re.I), "mutter_von", -1, 1),
    # "geboren am ... in STADT" oder "geboren in STADT"
    (re.compile(r"Geboren\s+(?:am\s+[\d\.]+\s+)?in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)", re.I), "geboren_in", -1, 1),
    # "wohne / lebe / wohnhaft in STADT"
    (re.compile(r"(?:wohne|wohnhaft|lebe|wohnt|lebt)\s+in\s+([A-ZÄÖÜ][a-zäöüß\-]+)", re.I), "wohnt_in", -1, 1),
    # "arbeite / arbeitet als BERUF"
    (re.compile(r"(?:arbeite|arbeitet|bin|ist)\s+als\s+([A-ZÄÖÜ][a-zäöüß\-]+)", re.I), "arbeitet_als", -1, 1),
    # "bin BERUF" (bekannte Berufe)
    (re.compile(r"(?:ich\s+)?bin\s+(IT-Forensiker|Entwickler|Ingenieur|Arzt|Lehrer|Programmierer|Elektriker|Designer|Manager|Berater|Analyst|Buchhalter|Pilot|Koch|Pfleger)\b", re.I), "ist_beruf", -1, 1),
    # "heisse / heiße / mein Name ist X"
    (re.compile(r"(?:hei[sß]e|mein\s+Name\s+ist)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)", re.I), "hat_namen", -1, 1),
    # "nutze / benutze / verwende X" (Werkzeuge/Tech)
    (re.compile(r"(?:nutze|benutze|verwende|nutzt|benutzt)\s+([A-Za-z0-9][A-Za-zÄÖÜäöüß0-9\-\.]+)", re.I), "nutzt", -1, 1),
    # "läuft auf / installiert auf X"
    (re.compile(r"(?:läuft|laufen|installiert)\s+auf\s+(?:dem\s+|einem\s+)?([A-Za-z][A-Za-z0-9\-\s]+?)(?:\s+mit|\s*$|,)", re.I), "laeuft_auf", -1, 1),
]

# Relationship labels für Anzeige (Deutsch)
_RELATION_LABELS: dict[str, str] = {
    "vater_von": "Vater von",
    "mutter_von": "Mutter von",
    "geboren_in": "geboren in",
    "wohnt_in": "wohnt in",
    "arbeitet_als": "arbeitet als",
    "ist_beruf": "ist",
    "hat_namen": "heißt",
    "nutzt": "nutzt",
    "laeuft_auf": "läuft auf",
    "co_occurrence": "verwandt mit",
    "mentioned_with": "erwähnt mit",
}

# ── Stop-Words (Deutsch + Englisch) ──────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    # Deutsche Personalpronomen
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mich", "dich", "sich", "uns", "euch",
    "mir", "dir", "ihm", "ihnen", "ihn",
    # Possessivpronomen
    "mein", "meine", "meinen", "meinem", "meiner",
    "dein", "deine", "deinen", "deinem", "deiner",
    "sein", "seine", "seinen", "seinem", "seiner",
    "unser", "unsere", "unseren", "unserem", "unserer",
    # Artikel
    "der", "die", "das", "dem", "den", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
    # Fragewörter
    "was", "wer", "wie", "wo", "wann", "warum", "weshalb",
    "welche", "welcher", "welches", "welchem", "welchen",
    # Hilfsverben
    "bin", "bist", "ist", "sind", "seid",
    "war", "waren", "wäre", "wären", "sei",
    "habe", "hast", "hat", "haben", "habt", "hatte", "hatten",
    "werde", "wirst", "wird", "werden", "werdet", "wurde", "wurden",
    "soll", "sollst", "sollen", "sollt", "sollte", "sollten",
    "kann", "kannst", "können", "könnt", "konnte", "konnten",
    "muss", "musst", "müssen", "müsst", "musste", "mussten",
    "darf", "darfst", "dürfen", "dürft", "durfte", "durften",
    "mag", "magst", "mögen", "mögt", "mochte", "mochten", "möchte",
    "will", "willst", "wollen", "wollt", "wollte", "wollten", "würde",
    # Verben
    "gibt", "geben", "nehmen", "kommen", "gehen", "sehen",
    "wissen", "denken", "glauben", "meinen", "sagen", "machen",
    "gemacht", "gesagt", "geworden", "worden",
    "läuft", "laufen",
    "wohne", "wohnst", "wohnt", "wohnen", "wohnhaft",
    "arbeite", "arbeitest", "arbeitet", "arbeiten",
    "heisse", "heißt", "heissen", "heiße", "heißen",
    "lebe", "lebst", "lebt", "leben",
    "nutze", "nutzt", "nutzen",
    "spreche", "sprichst", "spricht", "sprechen",
    "installiert", "installiere", "installieren",
    # Präpositionen / Konjunktionen
    "und", "oder", "aber", "doch", "jedoch",
    "mit", "auf", "für", "von", "aus", "bei", "nach", "seit", "vor",
    "über", "unter", "neben", "zwischen", "durch", "ohne", "gegen",
    "in", "an", "am", "im", "ins", "zum", "zur", "zu", "bis", "als",
    "wenn", "dann", "dass", "damit", "weil", "da", "ob", "obwohl",
    "beim", "vom", "ans", "aufs", "ums", "fürs", "hinter",
    # Allgemein
    "bitte", "hallo", "danke", "okay", "ok", "ja", "nein",
    "mal", "noch", "schon", "auch", "nur", "halt", "eben",
    "mehr", "sehr", "alle", "alles",
    "immer", "nie", "bereits", "erst",
    "hier", "dort", "nun", "jetzt", "heute", "morgen", "gestern",
    "so", "gut", "neu", "alt", "groß", "klein",
    "kein", "keine", "keinen", "keinem",
    # Englisch
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "of", "at", "by", "for", "with", "about", "to", "from",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "will", "should", "would", "could", "may", "might",
    "all", "both", "each", "more", "most", "other", "some", "no",
    "not", "only", "same", "so", "than", "too", "very", "just",
    "user", "users", "true", "false", "none", "null",
})

# Wörter die KEINE sinnvollen Knoten sind, obwohl sie die Stop-Word-Prüfung passieren
_SEMANTIC_BLACKLIST: frozenset[str] = frozenset({
    # Generische Relationswörter die als Knoten auftauchen würden
    "geboren", "geburt", "geburtstag", "alter", "jahre", "jahren", "jährig",
    "kinder", "kindern", "kind", "söhne", "töchter", "sohn", "tochter",
    "vater", "mutter", "eltern", "geschwister",
    "name", "namen", "vorname", "nachname",
    # Tech-Generika
    "system", "server", "antwort", "frage", "hinweis", "fehler",
    "schritt", "punkt", "liste", "beispiel", "bereich", "methode",
    "funktion", "wert", "datei", "ordner", "problem", "lösung",
    "ergebnis", "information", "version", "update", "status", "zustand",
    "prozess", "service", "benutzer", "nutzer",
    # Zeitwörter die nicht Entitäten sind
    "uhrzeit", "datum", "zeitpunkt", "zeitraum", "dauer",
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag",
    "samstag", "sonntag", "januar", "februar", "märz", "april",
    "mai", "juni", "juli", "august", "september", "oktober",
    "november", "dezember",
})

_SEMANTIC_BLACKLIST_LOWER: frozenset[str] = frozenset(w.lower() for w in _SEMANTIC_BLACKLIST)

# ── NER: Personen-Namen extrahieren ─────────────────────────────────────────

# Pattern für Personen-Erwähnungen: "Leon Maxim (8 Jahre)", "Emilia Grace (6 J)"
_PERSON_WITH_AGE = re.compile(
    r"([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+)"  # Voller Name (mind. 2 Wörter)
    r"\s*\(\s*(\d+)\s*(?:Jahre?|J\.?|Monate?)\s*\)",       # (8 Jahre) oder (6 J)
    re.UNICODE
)

# Einfacher Name: zwei aufeinanderfolgende großgeschriebene Wörter (kein Satzanfang)
_FULL_NAME = re.compile(
    r"(?<!\.\s)(?<![:\n])\b([A-ZÄÖÜ][a-zäöüß]{2,})\s+([A-ZÄÖÜ][a-zäöüß]{2,})\b",
    re.UNICODE
)

# Geburtsdatum-Pattern
_BIRTH_DATE = re.compile(
    r"(?:geboren|geb\.?)\s+am\s+(\d{1,2}\.\d{1,2}\.\d{4})",
    re.I
)


def _extract_persons(text: str) -> list[dict]:
    """Extrahiert Personen mit optionalem Alter aus Text."""
    persons = []
    seen = set()

    # Pattern 1: Name mit Alter "Leon Maxim (8 Jahre)"
    for m in _PERSON_WITH_AGE.finditer(text):
        full_name = m.group(1).strip()
        age = int(m.group(2))
        key = full_name.lower()
        if key not in seen:
            seen.add(key)
            persons.append({
                "name": full_name,
                "type": "person",
                "age": age,
                "mention_span": (m.start(), m.end()),
            })

    # Pattern 2: Vollständige Namen ohne Alter (wenn nicht schon gefunden)
    for m in _FULL_NAME.finditer(text):
        first = m.group(1)
        last = m.group(2)
        full_name = f"{first} {last}"
        key = full_name.lower()
        # Nicht als Person zählen wenn der gesamte Ausdruck ein Stopwort ist
        if key in _STOP_WORDS or key in _SEMANTIC_BLACKLIST_LOWER:
            continue
        # Bug 2 Fix: Einzeltokens prüfen — "Mein Name" enthält "mein" (Stopwort)
        # und "name" (Blacklist) → verwerfen
        _part_tokens = {first.lower(), last.lower()}
        if _part_tokens & (_STOP_WORDS | _SEMANTIC_BLACKLIST_LOWER):
            continue
        if last.lower() in _KNOWN_PLACES_LOWER:
            continue
        if key not in seen:
            seen.add(key)
            persons.append({
                "name": full_name,
                "type": "person",
                "age": None,
                "mention_span": (m.start(), m.end()),
            })

    return persons


def _extract_places(text: str) -> list[dict]:
    """Extrahiert Orte/Städte aus Text."""
    places = []
    seen = set()

    # Bekannte Orte aus Lookup-Tabelle
    for lower, canonical in _KNOWN_PLACES_LOWER.items():
        if lower in text.lower():
            # Wortgrenze sicherstellen
            if re.search(r'\b' + re.escape(canonical) + r'\b', text, re.I):
                if lower not in seen:
                    seen.add(lower)
                    places.append({
                        "name": canonical,
                        "type": "place",
                    })

    # "in STADTNAME" Pattern für unbekannte Orte (nur wenn Großschreibung + mind. 4 Zeichen)
    for m in re.finditer(r"\bin\s+([A-ZÄÖÜ][a-zäöüß\-]{3,}(?:\s+[A-ZÄÖÜ][a-zäöüß\-]+)?)\b", text):
        candidate = m.group(1).strip()
        key = candidate.lower()
        if key in _STOP_WORDS or key in _SEMANTIC_BLACKLIST_LOWER:
            continue
        # Keine Orte die nur Verben/Adjektive sind
        if re.match(r"^[A-ZÄÖÜ][a-z]+(?:en|te|te|ung|heit|keit|lich|isch)$", candidate):
            continue
        if key not in seen:
            seen.add(key)
            places.append({
                "name": candidate,
                "type": "place",
            })

    return places


def _extract_dates(text: str) -> list[dict]:
    """Extrahiert Datumsangaben."""
    dates = []
    for m in re.finditer(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text):
        dates.append({
            "name": m.group(0),
            "type": "date",
            "day": int(m.group(1)),
            "month": int(m.group(2)),
            "year": int(m.group(3)),
        })
    return dates


def _extract_tech_concepts(
    text: str,
    person_name_tokens: frozenset[str] | None = None,
) -> list[dict]:
    """Extrahiert technische Konzepte (Produkte, Dienste, Technologien).

    person_name_tokens: frozenset von Einzel-Tokens aus extrahierten Personennamen
    (z.B. {"Leon", "Maxim", "Emilia"}) — diese werden aus Tier 2+3 herausgefiltert,
    damit Vorname/Nachname nicht als separate Konzept-Knoten erscheinen.
    """
    concepts = []
    seen = set()

    # Bekannte Tech-Patterns
    tech_patterns = [
        r'\b(Raspberry\s+Pi\s*\d*)\b',
        r'\b(Home\s+Assistant)\b',
        r'\b(Docker(?:\s+Container)?)\b',
        r'\b(Python\s*\d*(?:\.\d+)?)\b',
        r'\b(AirPI)\b',
        r'\b(GitHub)\b',
        r'\b(Linux)\b',
        r'\b(Ubuntu|Debian|Arch)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+(?:B|GB|MB|TB))\b',  # "8GB RAM"
        r'\b(\w+(?:-\w+)+\.gguf)\b',  # GGUF Modelle
        r'\b(llama\.cpp|llama-cpp)\b',
        r'\b(FastAPI|uvicorn|SQLite|PostgreSQL|MySQL)\b',
    ]
    for pat in tech_patterns:
        for m in re.finditer(pat, text, re.I):
            name = m.group(1).strip()
            key = name.lower()
            if key not in seen and len(name) >= 3:
                seen.add(key)
                concepts.append({"name": name, "type": "tech"})

    # CamelCase / Großbuchstaben-Tokens als mögliche Konzepte
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}[a-z]*[A-Z][a-z]*)\b', text):
        name = m.group(1)
        key = name.lower()
        if key in _STOP_WORDS or key in _SEMANTIC_BLACKLIST_LOWER:
            continue
        # Bug 1 Fix: Token das Teil eines Personennamens ist → überspringen
        if person_name_tokens and name in person_name_tokens:
            continue
        if key not in seen and len(name) >= 4:
            seen.add(key)
            concepts.append({"name": name, "type": "concept"})

    # General capitalized concepts. Keeps backwards-compatible nodes such as
    # "Haskell" while the specialised extractors handle richer entities first.
    for m in re.finditer(r'\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9\-]{3,})\b', text):
        name = m.group(1)
        key = name.lower()
        if key in _STOP_WORDS or key in _SEMANTIC_BLACKLIST_LOWER:
            continue
        if key in _KNOWN_PLACES_LOWER:
            continue
        # Bug 1 Fix: Token das Teil eines Personennamens ist → überspringen
        if person_name_tokens and name in person_name_tokens:
            continue
        if key not in seen:
            seen.add(key)
            concepts.append({"name": name, "type": "concept"})

    return concepts


# ── Relationship Extraction ──────────────────────────────────────────────────

def _extract_relationships(text: str, entities: dict[str, dict]) -> list[dict]:
    """Extrahiert semantische Beziehungen als gerichtete Kanten."""
    relationships = []
    text_lower = text.lower()

    # Pattern-basierte Extraction
    for pattern, rel_type, subj_grp, obj_grp in _RELATION_PATTERNS:
        for m in pattern.finditer(text):
            obj_text = m.group(obj_grp).strip() if obj_grp > 0 else m.group(0).strip()

            # Subjekt: entweder explizit oder implizit "user"
            subject = "__user__"

            # Objekt aus extrahierten Entities suchen
            obj_lower = obj_text.lower()
            matched_entity = None
            for ent_key, ent_data in entities.items():
                if ent_key in obj_lower or obj_lower in ent_key:
                    matched_entity = ent_data["name"]
                    break

            if matched_entity is None:
                # Trotzdem als Relationship speichern mit dem rohen Text
                obj_text_clean = obj_text.split(",")[0].strip()  # Nur erstes Element
                # Bug 2 Fix: Phrasen die Stopwörter oder Blacklist-Wörter enthalten ablehnen
                # (z.B. "Mein Name" → "mein" in _STOP_WORDS → wird verworfen)
                # Bug 3 Fix: "2 Kindern" → "kindern" in _SEMANTIC_BLACKLIST_LOWER → verworfen
                _tokens_lower = set(obj_text_clean.lower().split())
                _has_stopword = bool(_tokens_lower & (_STOP_WORDS | _SEMANTIC_BLACKLIST_LOWER))
                if len(obj_text_clean) >= 2 and not _has_stopword:
                    matched_entity = obj_text_clean

            if matched_entity:
                relationships.append({
                    "source": subject,
                    "target": matched_entity,
                    "type": rel_type,
                    "label": _RELATION_LABELS.get(rel_type, rel_type),
                })

    # Personen in "Vater von X, Y, Z" Pattern
    vater_m = re.search(r"(?:ich\s+)?bin\s+Vater\s+von\s+(\d+)\s+Kindern?\s*:\s*(.+)", text, re.I)
    if vater_m:
        children_text = vater_m.group(2)
        # Kinder-Namen extrahieren
        persons = _extract_persons(children_text)
        for person in persons:
            relationships.append({
                "source": "__user__",
                "target": person["name"],
                "type": "vater_von",
                "label": "Vater von",
            })

    # Orte zu Personen (geboren in)
    born_pattern = re.compile(
        r"([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+)"  # Name
        r".*?[Gg]eboren\s+am\s+[\d\.]+\s+in\s+"              # geboren am ... in
        r"([A-ZÄÖÜ][a-zäöüß]+)",                               # Stadt
        re.DOTALL
    )
    for m in born_pattern.finditer(text):
        person_name = m.group(1).strip()
        place_name = m.group(2).strip()
        if person_name.lower() not in _SEMANTIC_BLACKLIST_LOWER:
            relationships.append({
                "source": person_name,
                "target": place_name,
                "type": "geboren_in",
                "label": "geboren in",
            })

    # Deduplizierung
    seen_rels = set()
    unique_rels = []
    for rel in relationships:
        key = (rel["source"], rel["target"], rel["type"])
        if key not in seen_rels:
            seen_rels.add(key)
            unique_rels.append(rel)

    return unique_rels


# ── Hauptklasse GraphBuilder ─────────────────────────────────────────────────

class GraphBuilder:
    """Baut einen semantischen Knoten-Kanten-Graphen aus Memory-Einträgen."""

    def build(self, entries: list[dict]) -> dict:
        """
        Input: Liste von Memory-Einträgen mit 'content' Feld
        Output: {"nodes": [...], "edges": [...], "meta": {...}}

        v2: Typisierte Knoten, gerichtete Kanten mit Labels, NER
        """
        now = time.time()

        # Bei leeren Einträgen: leeren Graph zurückgeben
        if not entries:
            return {
                "nodes": [],
                "edges": [],
                "meta": {
                    "total_entries": 0,
                    "node_types": {},
                    "relation_types": [],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            }

        # ── Phase 1: Alle Entities aus allen Einträgen extrahieren ──────────
        # entity_key (lower) → {name, type, entries: set, memory_ids: list, meta...}
        entity_map: dict[str, dict] = {}

        # Immer: __user__ als zentraler Knoten
        entity_map["__user__"] = {
            "name": "Ich",
            "type": "user",
            "entries": set(),
            "memory_ids": [],
            "age_days": 0.0,
            "confidence": 100,
            "confidence_sum": 100,
            "confidence_count": 1,
            "last_seen_at": now,
            "category": "system",
            "source": "system",
        }

        # Gesammelte Relationships aus allen Einträgen
        all_relationships: list[dict] = []

        for idx, entry in enumerate(entries):
            content = entry.get("content", "")
            entry_id = entry.get("id")
            entry_category = entry.get("category", "fact") or "fact"
            entry_source = entry.get("source", "auto") or "auto"
            raw_last_seen = entry.get("last_seen_at") or entry.get("created_at") or 0
            try:
                raw_last_seen = float(raw_last_seen)
            except (TypeError, ValueError):
                raw_last_seen = 0.0
            raw_confidence = entry.get("confidence")
            try:
                entry_confidence = int(raw_confidence) if raw_confidence is not None else 80
            except (TypeError, ValueError):
                entry_confidence = 80

            # NER: Alle Entity-Typen extrahieren
            persons = _extract_persons(content)
            # Bug 1 Fix: Nur Tokens aus HOCHKONFIDENTEN Personennamen (mit explizitem Alter)
            # blockieren, damit "Leon" und "Maxim" nicht als separate Konzeptknoten erscheinen.
            # _FULL_NAME-Matches (ohne Alter) sind zu unsicher (z.B. "Haskell Funktional")
            # und würden legitime Konzepte blockieren.
            _person_tokens: frozenset[str] = frozenset(
                token
                for p in persons
                if p.get("age") is not None   # Nur Personen mit explizitem Alter
                for token in p["name"].split()
            )
            places = _extract_places(content)
            dates = _extract_dates(content)
            techs = _extract_tech_concepts(content, person_name_tokens=_person_tokens)

            entry_entities: list[dict] = persons + places + dates + techs

            # Entities in entity_map eintragen
            for ent in entry_entities:
                key = ent["name"].lower()
                if key in _STOP_WORDS or key in _SEMANTIC_BLACKLIST_LOWER:
                    continue
                if len(ent["name"]) < 2:
                    continue

                if key not in entity_map:
                    entity_map[key] = {
                        "name": ent["name"],
                        "type": ent.get("type", "concept"),
                        "entries": set(),
                        "memory_ids": [],
                        "last_seen_at": raw_last_seen,
                        "confidence_sum": entry_confidence,
                        "confidence_count": 1,
                        "category": entry_category,
                        "source": entry_source,
                        # Extra-Felder je nach Typ
                        "age": ent.get("age"),
                        "birth_year": ent.get("year"),
                    }
                else:
                    if raw_last_seen > entity_map[key]["last_seen_at"]:
                        entity_map[key]["last_seen_at"] = raw_last_seen
                    entity_map[key]["confidence_sum"] += entry_confidence
                    entity_map[key]["confidence_count"] += 1
                    # Alter übernehmen wenn noch nicht gesetzt
                    if entity_map[key].get("age") is None and ent.get("age") is not None:
                        entity_map[key]["age"] = ent["age"]

                entity_map[key]["entries"].add(idx)
                if entry_id is not None and entry_id not in entity_map[key]["memory_ids"]:
                    entity_map[key]["memory_ids"].append(entry_id)

            # Relationships extrahieren
            local_entities = {e["name"].lower(): e for e in entry_entities}
            rels = _extract_relationships(content, local_entities)
            for rel in rels:
                rel["entry_idx"] = idx
            all_relationships.extend(rels)

            # __user__ last_seen_at aktualisieren
            if raw_last_seen > entity_map["__user__"]["last_seen_at"]:
                entity_map["__user__"]["last_seen_at"] = raw_last_seen
            entity_map["__user__"]["entries"].add(idx)

        # ── Phase 2: Kanten aufbauen ─────────────────────────────────────────

        # edge_key → {source, target, type, label, weight}
        edge_map: dict[tuple, dict] = {}

        for rel in all_relationships:
            src_raw = rel["source"]
            tgt_raw = rel["target"]

            # __user__ → Ich
            if src_raw == "__user__":
                src_name = "Ich"
                src_key = "__user__"
            else:
                src_key = src_raw.lower()
                src_name = entity_map.get(src_key, {}).get("name", src_raw)
                # Wenn Entity nicht bekannt: hinzufügen
                if src_key not in entity_map:
                    entity_map[src_key] = {
                        "name": src_raw,
                        "type": "concept",
                        "entries": set(),
                        "memory_ids": [],
                        "last_seen_at": now,
                        "confidence_sum": 80,
                        "confidence_count": 1,
                        "category": "fact",
                        "source": "auto",
                    }

            tgt_key = tgt_raw.lower()
            tgt_name = entity_map.get(tgt_key, {}).get("name", tgt_raw)
            if tgt_key not in entity_map:
                entity_map[tgt_key] = {
                    "name": tgt_raw,
                    "type": "concept",
                    "entries": set(),
                    "memory_ids": [],
                    "last_seen_at": now,
                    "confidence_sum": 80,
                    "confidence_count": 1,
                    "category": "fact",
                    "source": "auto",
                }

            edge_key = (src_key, tgt_key, rel["type"])
            if edge_key not in edge_map:
                edge_map[edge_key] = {
                    "key": graph_edge_key(src_key, tgt_key, rel["type"], True),
                    "source_key": src_key,
                    "target_key": tgt_key,
                    "source": src_name,
                    "target": tgt_name,
                    "type": rel["type"],
                    "label": rel.get("label", rel["type"]),
                    "weight": 1,
                    "directed": True,
                    "origin": "auto",
                    "origins": ["auto"],
                }
            else:
                edge_map[edge_key]["weight"] += 1

        # Co-Occurrence Kanten für Entities ohne explizite Relationship im selben Eintrag
        entry_entity_lists: dict[int, list[str]] = defaultdict(list)
        for ent_key, ent_data in entity_map.items():
            if ent_key == "__user__":
                continue
            for idx in ent_data.get("entries", set()):
                entry_entity_lists[idx].append(ent_key)

        for idx, ent_keys in entry_entity_lists.items():
            # Nur wenn keine explizite Relationship bereits existiert
            for i in range(len(ent_keys)):
                for j in range(i + 1, len(ent_keys)):
                    a = ent_keys[i]
                    b = ent_keys[j]
                    # Prüfen ob schon gerichtete Kante vorhanden
                    has_directed = (
                        any(k[:2] == (a, b) for k in edge_map) or
                        any(k[:2] == (b, a) for k in edge_map)
                    )
                    if not has_directed:
                        edge_key = (a, b, "co_occurrence")
                        if edge_key not in edge_map:
                            source_key, target_key = sorted((a, b))
                            edge_map[edge_key] = {
                                "key": graph_edge_key(source_key, target_key, "co_occurrence", False),
                                "source_key": source_key,
                                "target_key": target_key,
                                "source": entity_map[a]["name"],
                                "target": entity_map[b]["name"],
                                "type": "co_occurrence",
                                "label": "",
                                "weight": 1,
                                "directed": False,
                                "origin": "auto",
                                "origins": ["auto"],
                            }
                        else:
                            edge_map[edge_key]["weight"] += 1

        # Degree berechnen
        degree_map: dict[str, int] = defaultdict(int)
        for (src, tgt, _), edge_data in edge_map.items():
            degree_map[src] += 1
            degree_map[tgt] += 1

        # ── Phase 3: Output aufbauen ─────────────────────────────────────────

        # Knoten filtern: mind. 1 Kante ODER __user__
        valid_keys = {k for k in entity_map if degree_map.get(k, 0) >= 1 or k == "__user__"}

        # Wenn weniger als 3 Knoten: alle anzeigen
        if len(valid_keys) < 3:
            valid_keys = set(entity_map.keys())

        nodes = []
        for key in valid_keys:
            data = entity_map[key]
            last_seen = data.get("last_seen_at", 0) or 0
            if last_seen > 0:
                age_days = round((now - last_seen) / 86400, 1)
            else:
                age_days = 30.0

            conf_count = data.get("confidence_count", 1) or 1
            confidence = round(data.get("confidence_sum", 80) / conf_count)

            node = {
                "key": key,
                "id": data["name"],
                "type": data.get("type", "concept"),
                "weight": degree_map.get(key, 1),
                "entries": sorted(data.get("entries", set())),
                "category": data.get("category", "fact"),
                "source": data.get("source", "auto"),
                "age_days": age_days,
                "confidence": confidence,
                "memory_ids": sorted(data.get("memory_ids", [])),
            }
            # Extra-Metadaten je nach Typ
            if data.get("type") == "person" and data.get("age") is not None:
                node["person_age"] = data["age"]
            nodes.append(node)

        # Kanten filtern auf valide Knoten
        valid_names = {entity_map[k]["name"] for k in valid_keys}
        edges = []
        for edge_data in edge_map.values():
            if edge_data["source"] in valid_names and edge_data["target"] in valid_names:
                edges.append(edge_data)

        # Sortierung
        nodes.sort(key=lambda n: -n["weight"])
        edges.sort(key=lambda e: (-e["weight"], e["type"] != "co_occurrence"))

        # Entity-Typ Zusammenfassung für Meta
        type_counts: dict[str, int] = defaultdict(int)
        for n in nodes:
            type_counts[n["type"]] += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "total_entries": len(entries),
                "node_types": dict(type_counts),
                "relation_types": list({e["type"] for e in edges}),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }


def merge_graph_overlays(
    graph: dict,
    manual_edges: list[dict] | None = None,
    edge_overrides: list[dict] | None = None,
    manual_nodes: list[dict] | None = None,
) -> dict:
    """Merge user-managed graph overlays into an auto-generated graph.

    Automatic graph output remains disposable. Manual edges and overrides are
    persisted elsewhere and only merged at the response boundary.
    """
    result = {
        "nodes": [dict(node) for node in graph.get("nodes", [])],
        "edges": [dict(edge) for edge in graph.get("edges", [])],
        "meta": dict(graph.get("meta", {})),
    }
    manual_edges = manual_edges or []
    edge_overrides = edge_overrides or []

    hidden_auto_keys = {
        item.get("edge_key")
        for item in edge_overrides
        if item.get("active", 1) and item.get("action") == "hide"
    }
    result["edges"] = [
        edge
        for edge in result["edges"]
        if edge.get("origin") != "auto" or edge.get("key") not in hidden_auto_keys
    ]

    nodes_by_key = {node.get("key"): node for node in result["nodes"] if node.get("key")}
    edge_by_key = {edge.get("key"): edge for edge in result["edges"] if edge.get("key")}

    def ensure_node(node_key: str, display: str) -> None:
        if node_key in nodes_by_key:
            return
        node = {
            "key": node_key,
            "id": display,
            "type": "concept",
            "weight": 1,
            "entries": [],
            "category": "manual",
            "source": "manual",
            "age_days": 0,
            "confidence": 80,
            "memory_ids": [],
            "origin": "manual",
        }
        nodes_by_key[node_key] = node
        result["nodes"].append(node)

    for manual in manual_edges:
        if not manual.get("active", 1):
            continue
        source_key = normalize_node_key(str(manual.get("source_key", "")))
        target_key = normalize_node_key(str(manual.get("target_key", "")))
        if not source_key or not target_key:
            continue
        directed = bool(manual.get("directed", 1))
        relation_type = str(manual.get("relation_type") or "manual")
        key = graph_edge_key(source_key, target_key, relation_type, directed)
        source_name = nodes_by_key.get(source_key, {}).get("id") or manual.get("source_label") or source_key
        target_name = nodes_by_key.get(target_key, {}).get("id") or manual.get("target_label") or target_key
        ensure_node(source_key, str(source_name))
        ensure_node(target_key, str(target_name))

        if key in edge_by_key:
            existing = edge_by_key[key]
            origins = set(existing.get("origins") or [existing.get("origin", "auto")])
            origins.add("manual")
            existing["origin"] = "mixed" if "auto" in origins else "manual"
            existing["origins"] = sorted(origins)
            existing["manual_edge_id"] = manual.get("id")
            if manual.get("label"):
                existing["label"] = manual["label"]
            continue

        edge = {
            "key": key,
            "id": manual.get("id"),
            "source_key": source_key,
            "target_key": target_key,
            "source": nodes_by_key[source_key]["id"],
            "target": nodes_by_key[target_key]["id"],
            "type": relation_type,
            "label": manual.get("label") or relation_type,
            "weight": int(manual.get("weight") or 1),
            "directed": directed,
            "origin": "manual",
            "origins": ["manual"],
            "confidence": int(manual.get("confidence") or 80),
            "note": manual.get("note") or "",
        }
        result["edges"].append(edge)
        edge_by_key[key] = edge

    result["meta"]["manual_edges"] = len([e for e in manual_edges if e.get("active", 1)])
    result["meta"]["hidden_auto_edges"] = len(hidden_auto_keys)

    # ── Manuelle Knoten injizieren ────────────────────────────────────────────
    manual_nodes = manual_nodes or []
    for mn in manual_nodes:
        if not mn.get("active", True):
            continue
        node_key = normalize_node_key(str(mn.get("node_key") or mn.get("label") or ""))
        if not node_key:
            continue
        label = str(mn.get("label") or node_key)

        if node_key in nodes_by_key:
            # Existierender Auto-Knoten → als "mixed" markieren
            existing = nodes_by_key[node_key]
            existing["origin"] = "mixed"
            existing["manual_node_id"] = mn.get("id")
            if mn.get("label"):
                existing["id"] = mn["label"]   # Display-Label überschreiben
            if mn.get("note"):
                existing["manual_note"] = mn["note"]
        else:
            # Neuer rein-manueller Knoten → in Graph injizieren
            node = {
                "key":            node_key,
                "id":             label,
                "type":           mn.get("type") or "concept",
                "weight":         1,
                "entries":        [],
                "category":       "manual",
                "source":         "manual",
                "age_days":       0.0,
                "confidence":     int(mn.get("confidence") or 80),
                "memory_ids":     [],
                "origin":         "manual",
                "manual_node_id": mn.get("id"),
                "note":           mn.get("note") or "",
            }
            nodes_by_key[node_key] = node
            result["nodes"].append(node)

    result["meta"]["manual_nodes"] = len([n for n in manual_nodes if n.get("active", True)])
    return result


if __name__ == "__main__":
    # Synthetic smoke data. Keep real personal data out of repo examples.
    entries = [
        {
            "id": 1,
            "content": "ich bin Vater von 2 Kindern: Mira Test (8 Jahre) Geboren am 05.07.2017 in Berlin, Niko Demo (6 Jahre) Geboren am 17.10.2019 in Hamburg",
            "source": "user",
            "category": "fact",
            "confidence": 95,
            "last_seen_at": 1748000000.0,
        },
        {
            "id": 2,
            "content": "Mein Name ist Ada Demo, ich bin Entwickler in Berlin und nutze Python",
            "source": "user",
            "category": "fact",
            "confidence": 95,
            "last_seen_at": 1748000000.0,
        },
        {
            "id": 3,
            "content": "AirPI läuft auf dem Raspberry Pi 5 mit 8GB RAM und NVMe SSD",
            "source": "user",
            "category": "project",
            "confidence": 90,
            "last_seen_at": 1748000000.0,
        },
    ]
    g = GraphBuilder().build(entries)
    print("=== Nodes ===")
    for n in g["nodes"]:
        print(f"  [{n['type']:8}] {n['id']} (weight={n['weight']})")
    print("\n=== Edges ===")
    for e in g["edges"]:
        arrow = "→" if e.get("directed") else "—"
        label = f" [{e['label']}]" if e.get("label") else ""
        print(f"  {e['source']} {arrow}{label} {e['target']} (w={e['weight']})")
    print(f"\n=== Meta ===")
    print(f"  Nodes: {len(g['nodes'])}, Edges: {len(g['edges'])}")
    print(f"  Types: {g['meta']['node_types']}")
