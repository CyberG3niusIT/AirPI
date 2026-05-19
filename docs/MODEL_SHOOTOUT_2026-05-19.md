# AirPI Model Shootout 2026-05-19

## Context

| Field | Value |
| --- | --- |
| Host | Raspberry Pi |
| Date | 2026-05-19 |
| Runtime | Direct llama-cpp-python benchmark |
| Python | AirPI project virtualenv |
| Prompt | `Write one Python function that returns 42.` |
| Tokens per model | 32 |
| Context | `2048` |
| Decode threads | `3` |
| Batch threads | `4` |
| Flash attention | `true` |

## Results

### Existing Local Models

| Model | Size | Seconds | Tokens | tok/sec |
| --- | ---: | ---: | ---: | ---: |
| `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | 1.04 GB | 2.915 | 32 | 10.98 |
| `gemma-3-4b-it-Q4_K_M.gguf` | 2.32 GB | 7.375 | 32 | 4.34 |
| `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | 4.36 GB | 12.747 | 32 | 2.51 |

### Added Speed Candidates

The following files were added to `/data/models`:

| Source | File | Size |
| --- | --- | ---: |
| `bartowski/Qwen2.5-0.5B-Instruct-GGUF` | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 380 MB |
| `mradermacher/aquif-3.6-1B-i1-GGUF` | `aquif-3.6-1B.i1-Q4_K_M.gguf` | 1.2 GB |

Direct llama-cpp-python benchmark:

| Model | Size | Load seconds | Decode seconds | Tokens | tok/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 0.37 GB | 0.781 | 1.143 | 32 | 27.99 |
| `aquif-3.6-1B.i1-Q4_K_M.gguf` | 1.19 GB | 2.730 | 3.022 | 32 | 10.59 |
| `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | 1.04 GB | 4.857 | 2.703 | 32 | 11.84 |

AirPI HTTP benchmark for the best speed candidate:

| Case | OK | Seconds | Tokens | tok/sec |
| --- | --- | ---: | ---: | ---: |
| cold_or_first | yes | 3.683 | 64 | 17.38 |
| warm | yes | 2.266 | 64 | 28.24 |
| session_kv_reuse | yes | 2.295 | 64 | 27.89 |

## Conclusion

The newly added `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` model reaches the 15 tok/sec target in the real AirPI HTTP path.

`aquif-3.6-1B.i1-Q4_K_M.gguf` is not a speed upgrade over the existing 1.5B Qwen Coder model on this Pi.

The installed 4B and 7B models remain quality candidates, not speed candidates.

## Next Performance Step

Use `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` as the fast-lane model and keep `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` as the higher-quality coding default until a quality smoke test confirms the smaller model is good enough for PI Guardian tasks.

## Operational Note

The live AirPI service remained healthy after the direct shootout:

| Check | Result |
| --- | --- |
| `/health` | `ok` |
| Queue depth | `0` |
| Loaded models | `0` |
| Recoveries | `0` |
