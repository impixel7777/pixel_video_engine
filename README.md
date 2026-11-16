# PIXEL AI

A local, modular YouTube production engine. It turns one topic and one channel
profile into a researched brief, a long-form script, Shorts, visual and audio
direction, a quality report, and FFmpeg assembly instructions — driving a local
DeepSeek model through `llama.cpp`.

Nothing here publishes to YouTube, invents research, or fabricates assets. When
a capability is not configured, the output says so.

---

## 1. What actually works today

| Capability | Status |
|---|---|
| Channel profiles (3 included, JSON-only to add) | Working |
| Topic analysis, brief, outline | Working (local model) |
| Research | Working **when a provider is configured**; otherwise `research_status: "unavailable"` and the script is hedged |
| Long script (`Long.txt`) | Working (local model, section by section) |
| Shorts (`Short.txt`) | Working, in the exact format the legacy Kokoro parser expects |
| Visual plan / production storyboard | Working (direction only) |
| Audio plan | Working (direction only) |
| Quality report (deterministic + optional AI review) | Working |
| Project validation, resumable pipeline, approval gate | Working |
| Image / video / music / SFX generation | **Pending** — provider slots exist, none configured |
| Video rendering | Implemented and smoke-tested with FFmpeg test patterns; refuses to render when real assets are missing |
| Kokoro narration | Not reimplemented; the untouched legacy script handles it via a folder junction |
| YouTube upload | Not implemented, by design |

---

## 2. Requirements

- Windows (the code is pure Python and mostly portable; the TTS bridge uses
  directory junctions).
- The repository's existing `.venv` with `jsonschema`, `requests`, `numpy`,
  `soundfile`, `kokoro`.
- `llama-server` from `llama.cpp` plus the DeepSeek GGUF model used by this
  machine.
- FFmpeg + ffprobe (`C:\ffmpeg\bin` by default).
- The Python standard library. **No new dependencies** were added.

---

## 3. Quick start

```powershell
cd C:\Users\pixel\Desktop\pixel-tts\pixel-ai

# 1. Check that the environment, profiles, prompts, schemas and server are OK
..\.venv\Scripts\python.exe main.py doctor

# 2. Produce the content (server must already be running)
..\.venv\Scripts\python.exe main.py produce `
    --profile pixel_tech `
    --topic "Why solid-state drives slow down when they fill up" `
    --auto-approve

# 3. Inspect the result
..\.venv\Scripts\python.exe main.py review   --project pixel_tech/why-solid-state-drives-slow-down-when-they-fill-up
..\.venv\Scripts\python.exe main.py validate --project pixel_tech/why-solid-state-drives-slow-down-when-they-fill-up

# 4. Prove the untouched Kokoro parser accepts the files
..\.venv\Scripts\python.exe main.py verify-tts --project pixel_tech/why-solid-state-drives-slow-down-when-they-fill-up

# 5. Optional: expose the project to the legacy TTS folder, then synthesise
..\.venv\Scripts\python.exe main.py link-tts
..\.venv\Scripts\python.exe ..\old\pixel_tts.py pixel_tech/why-solid-state-drives-slow-down-when-they-fill-up
```

The app **never launches `llama-server`**. It probes the configured endpoint
first and fails with an actionable message if the server is unreachable.

---

## 4. llama-server requirements (important)

The server must be started by you, before the pipeline:

```powershell
llama-server.exe ^
  -m C:\DeepSeek-R1-Distill-Qwen-14B-Q8_0\DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf ^
  -ngl 16 -t 6 -b 128 -ub 64 -c 8192 -np 1 -fa auto ^
  --reasoning-budget 0 ^
  --host 127.0.0.1 --port 8080
```

Two flags are not optional on this hardware:

- **`--reasoning-budget 0`** — DeepSeek-R1 is a reasoning model. Per-request
  `reasoning_budget`, `chat_template_kwargs.enable_thinking` and
  `reasoning_effort` are ignored by this `llama.cpp` build. Without the
  server-side flag, the model spends the whole completion budget in hidden
  reasoning and returns empty `content` (`finish_reason: "length"`). With it,
  reasoning is empty and only the answer is returned.
- **`-c 8192`** — large enough for the section prompts, small enough to fit in
  16 GB of RAM alongside the 15.7 GB Q8 model.

Measured throughput on this machine is roughly **2.3 tokens/second**. Plan
accordingly: a full 10-minute production is a long background job, not an
interactive command. Use `resume` freely — every completed stage is persisted.

The client auto-detects the context window from `/props`, so changing `-c`
does not require editing the config.

---

## 5. Commands

| Command | Purpose |
|---|---|
| `doctor` | Check Python, server, FFmpeg, profiles, prompts, schemas, providers |
| `config` | Print the resolved configuration (`--json` for machines) |
| `list-profiles` | List channel profiles |
| `list-projects` | List generated projects (`--profile` to filter) |
| `produce --profile ID --topic "..."` | Run the full pipeline |
| `resume --project ID/SLUG [--approve]` | Continue an interrupted or awaiting project |
| `status --project ID/SLUG [--json]` | Per-stage state and artefacts |
| `run-stage --project ID/SLUG --stage STAGE [--force]` | Re-run one stage (repeatable) |
| `review --project ID/SLUG` | Human-readable summary of every artefact |
| `validate --project ID/SLUG` | Schema + narration-format validation |
| `verify-tts --project ID/SLUG [--generate-audio]` | Run the real legacy parser/cleaner on the files |
| `link-tts [--profile ID]` | Create `old/scripts/<profile>` junctions |

Useful `produce` flags: `--auto-approve`, `--fresh`, `--yes`, `--assets`,
`--render`, `--asset-dry-run`, `--no-asset-dry-run`.

Exit codes: `0` success, `1` failure, `2` usage error, `3` awaiting script
approval.

---

## 6. Pipeline stages

```
CONNECTION -> TOPIC -> RESEARCH -> PLAN -> SCRIPT -> SHORTS
           -> VISUALS -> AUDIO -> QUALITY -> SAVE -> VALIDATE
           -> ASSETS -> VIDEO
```

- **CONNECTION** probes the server. It never starts it.
- **TOPIC** writes `ContentBrief.json`.
- **RESEARCH** collects provider sources, then asks the model to analyse them.
  The provider result is authoritative and overwritten in Python, so the model
  can never upgrade "unavailable" to "available".
- **PLAN** writes `Outline.json` (structure, no prose).
- **SCRIPT** writes `Long.txt` section by section (`hook`, `intro`, `S1..Sn`,
  `conclusion`) with rolling summaries for continuity. The model returns JSON;
  Python owns block IDs, character offsets, durations and the final narration.
- **SHORTS** writes `Short.txt` in the legacy heading format.
- **VISUALS** writes `Long_Production.json` and `Shorts_Production.json`.
  Every scene is anchored to an exact character span of narration. Coverage is
  checked and repaired; duplicate visual concepts get a variety pass.
- **AUDIO** writes `AudioPlan.json` (music segments, SFX cues, pauses, mix).
- **QUALITY** runs deterministic checks and an optional AI editorial review,
  then writes `QualityReport.json`. It runs **after** visuals and audio so it
  can actually validate them.
- **SAVE** writes `project.json` and updates `production_state.json`.
- **VALIDATE** re-validates every artefact and both narration files.
- **ASSETS** (only with `--assets`) plans and requests assets. Unconfigured
  providers produce `pending` entries, never fake files.
- **VIDEO** (only with `--render`) refuses to run unless every asset is ready.

Every stage has configurable retries, and a failed stage is recorded as failed —
never silently skipped. Regenerating an upstream stage invalidates its downstream
content stages.

There is a hard approval checkpoint between `SHORTS` and `VISUALS`;
`--auto-approve` records that the checkpoint was bypassed.

---

## 7. Output contract

Each project lives in `scripts/<profile_id>/<topic-slug>/`:

```
Long.txt                  narration only - no headings, metadata or directions
Short.txt                 SHORT 01 — Title, blank line, narration only
ContentBrief.json
Research.json
Outline.json
Long_Sections.json
Long_Production.json
Shorts_Production.json
AudioPlan.json
QualityReport.json
production_state.json
project.json
assets/{images,videos,sounds,music}/
renders/
```

`Long.txt` and `Short.txt` are the only files Kokoro reads, and they contain
spoken words only. The exact heading format is:

```
SHORT 01 — Title
```

An em dash, zero-padded number, no duration metadata. This is not a convention:
the untouched `old/pixel_tts.py` parser is used as the oracle in the test suite,
and `main.py verify-tts` runs the real parser against the real files.

---

## 8. Kokoro bridge (without editing the legacy script)

The legacy script resolves `old/scripts/<folder_name>`. Instead of modifying it,
`link-tts` creates one Windows directory junction per profile:

```
old/scripts/pixel_tech  ->  pixel-ai/scripts/pixel_tech
```

Then:

```powershell
..\.venv\Scripts\python.exe ..\old\pixel_tts.py pixel_tech/<topic-slug>
```

The TTS configuration (`am_liam`, speed `1.25`, `24000 Hz`, CPU, `lang_code="a"`)
is untouched. Audio lands in the project's `output/` folder:
`output/Long/narration.wav` and `output/Shorts/Short_NN/narration.wav`.

Remove junctions with `python tools/link_tts_bridge.py --remove`.

---

## 9. Configuration

Everything tunable lives in two files; engine code reads it through
`config/settings.py` and never hardcodes model settings.

`config/model_settings.json`
- `base_url`, `model`
- `server.context_size` (auto-detected from `/props` when the server is up)
- `server.reasoning_budget` (must stay `0`; documented above)
- request timeouts, retries, JSON strictness, narration sanitisation

`config/pipeline_settings.json`
- script word targets and Shorts heading characters
- research provider and limits
- visual scene sizing, variety repair
- audio mix defaults
- asset provider names
- FFmpeg paths, resolutions, frame rates, `video.render`
- quality threshold (`error` / `warning` / `never`)

Prompts are external templates in `prompts/*.txt` using `{{PLACEHOLDER}}`
syntax. Every declared placeholder must be supplied; rendering fails loudly
otherwise. Schemas are in `schemas/*.json` and are the contract for each AI
call.

### Adding a channel

Copy a file in `profiles/`, change `channel.id`, and run `doctor`. No code
changes. `pixel_tech`, `pixel_history` and `pixel_science` are included.

---

## 10. Providers

### Research

Set `research.provider` to one of `tavily`, `serper`, `custom_http`, and provide:

| Provider | Environment |
|---|---|
| `tavily` | `TAVILY_API_KEY` |
| `serper` | `SERPER_API_KEY` |
| `custom_http` | `PIXEL_SEARCH_ENDPOINT`, optional `PIXEL_SEARCH_API_KEY` |

`PIXEL_RESEARCH_PROVIDER` can override the configured name. With `none` (the
default), no network call is made and `Research.json` records
`research_status: "unavailable"`.

### Images, video, music, SFX

`assets.image_provider`, `video_provider`, `sfx_provider`, `music_provider` are
`none` by default. The provider registry is intentionally empty: implement the
`AssetProvider` interface in `providers/asset_base.py`, register it in the
relevant `providers/*/__init__.py`, and configure credentials through the
environment. Until then every required asset is recorded as `pending` with a
reason, and rendering refuses to proceed.

---

## 11. Tests

Run the full suite with the standard library (no pytest needed):

```powershell
..\.venv\Scripts\python.exe tests\run_all.py
```

Covered, with real execution rather than mocks wherever possible:

- narration splitting, offsets, coverage, sanitisation, purity, repetition
- Shorts round-trip **against the legacy parser's own regex read from
  `old/pixel_tts.py` via AST**
- JSON extraction/repair/validation, reasoning stripping
- every schema validates against Draft 2020-12 and has resolvable refs
- every prompt template renders with every declared placeholder
- every profile loads, validates and produces a stable fingerprint
- FFmpeg argument builders, concat escaping, limiter/music maths
- **a real FFmpeg render**: test patterns → clips → concat → mux → ffprobe
- project validation failures (missing files, bad JSON, metadata in narration,
  wrong Shorts count, broken headings)
- pipeline state: persistence, resume, forced stages, downstream invalidation,
  approval gate
- asset planning: prompt de-duplication, `not_required` overlays, honest
  `pending` with no fabricated files
- client budgeting, context overflow, retry/repair loop, unreachable server
- **live server**: probe, context detection, tiny schema-constrained completion,
  reasoning isolation

The two expensive tests (real render, live model call) skip cleanly when FFmpeg
or the server is unavailable. Measured on this machine: **132 tests, 0 failures,
0 errors, 0 skipped, ~35 s**.

`tools/verify_tts_compat.py` is a separate acceptance check because it imports
Kokoro/Torch. Measured result on a generated project: 6/6 Shorts parsed by the
real legacy parser, `clean_text` preserved 100% of the long narration.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ServerNotAvailable` or probe fails | Start `llama-server` with the command above; check `base_url` |
| Empty model content, `finish_reason: "length"` | Missing `--reasoning-budget 0` |
| `ContextOverflowError` | Prompt + output exceed `-c`; raise the server context or lower `default_max_tokens` |
| `InvalidStructuredOutput` | The model failed every repair attempt; the error includes the validation errors and raw output |
| `ffmpeg not found` | Set `video.ffmpeg_path` / `video.ffprobe_path` in `pipeline_settings.json` |
| Rendering refuses to start | Assets are `pending`; configure a provider or keep `video.render` off |
| A Short is missing after TTS | Check `verify-tts`; headings must use the em-dash format |

Logs are written per run under `logs/`.

---

## 13. Honest limitations

- **Model speed** is the binding constraint: ~2.3 tok/s. The engine is designed
  for background runs and resumability, not interactivity.
- **No asset generation** is implemented. The scenes describe what to generate;
  the provider slots are empty.
- **No rendering of real videos** yet: the FFmpeg path is proven with test
  sources only, because no real assets exist.
- **Research is off by default.** Without a provider the script deliberately
  avoids specific statistics and marks claims as uncertain.
- **No YouTube upload.**
- The loop is linear by design; there is no DAG scheduler.
#   p i x e l _ v i d e o _ e n g i n e  
 