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

| Model | Size | Seconds | Tokens | tok/sec |
| --- | ---: | ---: | ---: | ---: |
| `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | 1.04 GB | 2.915 | 32 | 10.98 |
| `gemma-3-4b-it-Q4_K_M.gguf` | 2.32 GB | 7.375 | 32 | 4.34 |
| `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | 4.36 GB | 12.747 | 32 | 2.51 |

## Conclusion

The currently installed 1.5B Qwen model is the fastest available local model by a wide margin.

The 15 tok/sec target is not reachable by switching to the installed 4B or 7B models. Those models are better quality candidates, not speed candidates.

## Next Performance Candidates

To move toward 15 tok/sec, AirPI needs one of these:

1. A smaller GGUF model around 0.5B to 1B parameters.
2. A more aggressive quantization of the current 1.5B model, for example Q3 or Q2 class, if quality remains acceptable.
3. A llama.cpp build optimization pass for this exact Pi, then repeat the same benchmark.

## Operational Note

The live AirPI service remained healthy after the direct shootout:

| Check | Result |
| --- | --- |
| `/health` | `ok` |
| Queue depth | `0` |
| Loaded models | `0` |
| Recoveries | `0` |

