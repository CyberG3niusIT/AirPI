# AirPI Test Report — 2026-06-01

## Umgebung

| Eigenschaft | Wert |
|---|---|
| Plattform | Linux 6.18 · Raspberry Pi 5 |
| Python | 3.13.5 |
| pytest | 9.0.3 |
| Server-Endpunkt | http://localhost:11435 |
| Geladenes Modell | qwen2.5-coder-1.5b-instruct-q4_k_m.gguf |
| Gesamtlaufzeit Unit-Tests | 5.49 s |
| Gesamtlaufzeit Integrationstests | 11.34 s |

---

## Testergebnisse

### Unit Tests (43 Tests)

Ausgeführt mit:
```
.venv/bin/python -m pytest tests/test_server.py tests/test_model_manager.py tests/test_chat.py -v --timeout=30
```

- 43 passed, 0 failed

| Datei | Tests | Status |
|---|---|---|
| `tests/test_server.py` | 18 | alle bestanden |
| `tests/test_model_manager.py` | 12 | alle bestanden |
| `tests/test_chat.py` | 13 | alle bestanden |

<details>
<summary>Details (alle Tests)</summary>

**test_server.py**
- test_generate_json_mode_fails_after_invalid_repair — PASSED
- test_generate_json_mode_normalizes_json_response — PASSED
- test_generate_json_mode_repairs_once — PASSED
- test_generate_model_not_found_error_contract — PASSED
- test_generate_rejects_path_like_model_name — PASSED
- test_generate_rejects_path_like_preferred_model_name — PASSED
- test_generate_request_accepts_safe_session_id — PASSED
- test_generate_request_rejects_empty_or_too_large_prompt — PASSED
- test_generate_request_rejects_invalid_session_id — PASSED
- test_generate_resolves_fast_lane_alias — PASSED
- test_generate_resolves_preferred_fast_lane_alias — PASSED
- test_generate_success_records_metrics — PASSED
- test_live_health_and_metrics_contract — PASSED
- test_queue_full_error_contract — PASSED
- test_ready_reports_not_ready_without_default_model — PASSED
- test_ready_reports_not_ready_without_fast_model — PASSED
- test_stream_model_not_found_uses_final_error_frame — PASSED
- test_validation_payload_shape — PASSED

**test_model_manager.py**
- test_generate_retries_after_recoverable_failure — PASSED
- test_invalidate_model_state_removes_only_target_model — PASSED
- test_load_passes_pi_tuning_options_to_llama — PASSED
- test_recoverable_generation_error_detection — PASSED
- test_resolve_model_path_rejects_symlink_escape — PASSED
- test_select_model_keeps_explicit_model_name — PASSED
- test_select_model_rejects_path_like_or_non_gguf_names — PASSED
- test_select_model_resolves_builtin_aliases — PASSED
- test_select_model_resolves_fast_lane_alias — PASSED
- test_stream_generate_retries_after_recoverable_failure — PASSED
- test_validate_model_name_accepts_plain_gguf_basename — PASSED

**test_chat.py**
- test_chat_applies_chatml_fallback_when_no_template — PASSED
- test_chat_model_not_found_returns_404 — PASSED
- test_chat_queue_full_returns_503 — PASSED
- test_chat_rejects_invalid_model_name — PASSED
- test_chat_returns_assistant_message — PASSED
- test_chat_uses_jinja2_template_when_available — PASSED
- test_openai_chat_completions_maps_to_chat — PASSED
- test_openai_chat_handles_string_stop — PASSED
- test_api_ps_returns_empty_when_nothing_loaded — PASSED
- test_api_ps_returns_loaded_models — PASSED
- test_api_show_rejects_invalid_model_name — PASSED
- test_api_show_returns_model_details — PASSED
- test_chatml_fallback_includes_all_roles — PASSED
- test_chatml_fallback_used_when_jinja2_fails — PASSED

</details>

---

### Integration Tests — gegen localhost:11435 (20 Tests)

Ausgeführt mit:
```
.venv/bin/python -m pytest tests/test_integration.py -v --timeout=120 -s
```

- 20 passed, 0 failed

| Klasse | Tests | Status |
|---|---|---|
| `TestLiveness` | 4 | alle bestanden |
| `TestApiTags` | 2 | alle bestanden |
| `TestMemorySystem` | 4 | alle bestanden |
| `TestGraphData` | 3 | alle bestanden |
| `TestStaticUI` | 2 | alle bestanden |
| `TestApiGenerate` _(slow)_ | 3 | alle bestanden |
| `TestApiChat` _(slow)_ | 2 | alle bestanden |

<details>
<summary>Details (alle Tests)</summary>

**TestLiveness**
- test_live_returns_ok — PASSED (`GET /live` → `{"status":"ok"}`)
- test_health_has_required_fields — PASSED (status, queue_depth, active_sessions, loaded_models vorhanden)
- test_ready_endpoint — PASSED (200 oder 503)
- test_metrics_prometheus_format — PASSED (airpi_ prefix in HELP/TYPE-Zeilen)

**TestApiTags**
- test_api_tags_returns_models_list — PASSED (6 Modelle zurückgegeben)
- test_models_have_required_fields — PASSED (name, size, modified_at alle vorhanden)

**TestMemorySystem**
- test_store_and_retrieve_fact — PASSED (Fakt gespeichert und in /memory entries sichtbar)
- test_delete_fact_by_keyword — PASSED (delete by keyword entfernt Eintrag korrekt)
- test_memory_content_is_valid_markdown — PASSED (content beginnt mit `# AirPI Memory`)
- test_store_requires_content_field — PASSED (422 bei fehlendem content-Feld)

**TestGraphData**
- test_graph_data_structure — PASSED (nodes, edges, meta vorhanden)
- test_graph_reflects_stored_memory — PASSED (gespeichertes Konzept erscheint als Graph-Knoten)
- test_graph_redirect — PASSED (307 → /ui/graph.html)

**TestStaticUI**
- test_ui_index_html_serves — PASSED (GET /ui/ → 200 HTML)
- test_graph_html_serves — PASSED (GET /ui/graph.html → 200 HTML)

**TestApiGenerate** _(slow — echte LLM-Inferenz)_
- test_generate_non_streaming — PASSED (response, done, eval_count im Body)
- test_generate_invalid_model_returns_error — PASSED (HTTP 4xx/5xx bei unbekanntem Modell)
- test_generate_session_id_accepted — PASSED (session_id='integration-test' → kein Fehler)

**TestApiChat** _(slow — echte LLM-Inferenz)_
- test_chat_non_streaming — PASSED (message.role='assistant', message.content vorhanden, done=true)
- test_chat_system_message_respected — PASSED ('JAWOHL' in Antwort bei entsprechendem System-Prompt)

</details>

---

## Gefundene Probleme

### Problem 1 — Graph-Tokenizer trennt bei Unterstrichen (kein Bug, aber dokumentationswürdig)

Der `GraphBuilder` (memory/graph.py) tokenisiert mit `[A-Za-z0-9\-]+` — Unterstriche gelten als Wortgrenzer. Ein Token wie `INTEGRATION_TEST_abc123` wird in `INTEGRATION`, `TEST` und `abc123` aufgespalten. Das ist konzeptuell korrekt (Unterstriche sind keine natürlichen Wortbestandteile im Deutschen/Englischen), muss aber beim Schreiben von Tests berücksichtigt werden: Test-IDs mit Unterstrichen erscheinen nicht als einzelner Knoten im Graphen.

**Behebung:** Im Integrationstest wurde der eindeutige Ankerpunkt auf ein einzelnes Wort ohne Unterstriche (`XGRAPHNODE`) angepasst.

### Problem 2 — `requests`-Paket fehlte in der venv (kein Produktionsproblem)

Das `requests`-Paket ist als Abhängigkeit in `pyproject.toml` unter `[project] / dependencies` gelistet, war aber in der `.venv` nicht installiert (offenbar wurde `pip install -e .` nach einem venv-Neuaufbau nicht ausgeführt). Der Integrationstest schlug beim Import fehl, bis `pip install requests` manuell ausgeführt wurde.

**Empfehlung:** Siehe Empfehlung 1.

---

## Empfehlungen

1. **venv-Konsistenz sicherstellen:** Ein `make install` oder `pip install -e .[dev]` als Teil des CI-Setups verhindert fehlende Abhängigkeiten. Die Abhängigkeit `requests` ist bereits in `pyproject.toml` deklariert — ein `pip install -e .` im venv genügt.

2. **`pytest.mark.slow` bereits in pyproject.toml registriert:** Während der Testentwicklung wurde festgestellt, dass die `slow`-Marke nicht in `pyproject.toml` registriert war und Warnungen erzeugte. Das wurde behoben (`[tool.pytest.ini_options] markers`).

3. **TestApiChat::test_chat_system_message_respected ist LLM-abhängig:** Der Test erwartet `JAWOHL` in der Antwort. Bei kleineren oder anderen Modellen kann das Instruction-Following variieren. Sollte das Modell gewechselt werden, muss diese Prüfung ggf. angepasst oder per `@pytest.mark.xfail` markiert werden.

4. **Integrationstest-Cleanup ist best-effort:** `teardown_method` ruft `DELETE /memory/delete` mit dem Testtag auf. Falls der Test-Prozess abrupt beendet wird (Ctrl+C mid-test), bleiben Testartefakte in der Datenbank. Da die Einträge das Präfix `INTEGRATION_TEST_` + zufällige UUID tragen, ist manuelles Cleanup einfach: `POST /memory/delete {"keyword": "INTEGRATION_TEST_"}`.

5. **Kein Test für `/v1/chat/completions` (OpenAI-Kompatibilität):** Dieser Endpunkt existiert in `server.py` und delegiert an `/api/chat`. Er ist durch die Unit-Tests in `test_chat.py` (mock-basiert) abgedeckt, hat aber keinen Integrationstest gegen den Live-Server. Bei Bedarf kann `TestApiChat` um einen entsprechenden Test erweitert werden.
