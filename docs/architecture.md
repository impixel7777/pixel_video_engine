# PIXEL AI — architecture notes

This document explains *why* the engine is shaped the way it is. The README
covers usage; this covers invariants and trade-offs.

---

## 1. Data flow

```
                       profiles/<id>.json
                              |
topic --> TOPIC -----> ContentBrief.json
              |               |
              v               v
          RESEARCH ----> Research.json
                              |
                              v
            PLAN --------> Outline.json
                              |
                              v
          SCRIPT --------> Long_Sections.json  +  Long.txt
                              |
                              v
          SHORTS --------> Short.txt
                              |
              +---------------+---------------+
              |                               |
              v                               v
   VISUALS -> Long_Production.json   VISUALS -> Shorts_Production.json
              |                               |
              +---------------+---------------+
                              v
           AUDIO --------> AudioPlan.json
                              |
                              v
         QUALITY -------> QualityReport.json
                              |
                              v
            SAVE --------> project.json + production_state.json
                              |
                              v
         VALIDATE -------> (re-checks every artefact)
                              |
              +---------------+---------------+
              v                               v
   ASSETS (optional)                 VIDEO (optional)
   assets/status.json                renders/*.mp4
```

The pipeline is linear on purpose. A DAG scheduler would complicate resume and
invalidation without buying anything: the stages genuinely depend on their
predecessors in order.

---

## 2. Ownership: who writes what

| Data | Owner | Reason |
|---|---|---|
| Narration text | Model, sanitised by Python | Creative work |
| `Long.txt` / `Short.txt` layout | **Python** | Must stay byte-compatible with the legacy Kokoro parser |
| Block IDs, char offsets, durations | **Python** | Deterministic and auditable |
| Research sources / status | **Provider + Python** | The model must not invent citations |
| Visual/audio direction | Model, validated by Python | Creative work with strict schema |
| Quality verdict | Python checks + optional model review | Determinism first |
| Asset files | Providers only | Never fabricated |
| Publishing | Nobody | Out of scope |

The model never writes a file directly. Every AI response is JSON, validated
against a schema, then Python renders the artefact.

---

## 3. The narration contract

`Long.txt` and `Short.txt` are TTS input. They must contain spoken words only.
Two layers enforce this:

1. **Prompt-level**: the model is told to return narration in a JSON field, not
   to produce a formatted script.
2. **Code-level**: `sanitize_narration()` removes markdown, links, labels,
   stage directions and preambles; `is_pure_narration()` then rejects anything
   that still looks like production metadata. Validation fails the project if a
   narration file contains a heading, a duration, a visual prompt, a music cue,
   a URL, a speaker label or a reasoning tag.

Shorts headings are rendered by `format_shorts()` using
`SHORT {index:02d} — {title}`. The test suite reads the legacy regex out of
`old/pixel_tts.py` with `ast` and asserts our emitted headings match it exactly.
That is stronger than testing a copy of the pattern.

---

## 4. Scene anchoring and coverage

The visual director does not let the model choose where scenes begin. Python
splits narration into blocks of roughly N sentences / M words, records exact
`start_char`/`end_char` offsets, and asks the model only for *direction* per
block. The resulting plan is checked for coverage:

```
covered_chars / non_whitespace_narration_chars >= required_ratio
```

If coverage is short, the director inserts `coverage_insert` scenes for the
missing spans. The schema requires every scene to carry its narration, offsets,
duration, image prompt and asset block, so a downstream renderer never has to
guess.

Durations are computed from word count and the profile's words-per-minute, not
requested from the model. This keeps timing stable when the prose changes.

---

## 5. Structured output and repair

Every AI call uses `response_format: json_schema` when the server supports it.
The client:

1. strips `<think>` blocks and ignores `reasoning_content`;
2. extracts JSON even if the model wraps it in prose or fences;
3. repairs trailing commas and truncated brackets;
4. validates against the schema;
5. on failure, re-prompts with the exact validation errors (bounded retries);
6. raises `InvalidStructuredOutput` with the raw answer if all attempts fail.

Invalid output is never silently accepted or "fixed" by guessing semantics.

---

## 6. Why quality runs after visuals and audio

The original concept diagram put quality before production. That makes the
visual and audio checks impossible: there is nothing to validate yet. The
implemented order is:

```
... -> VISUALS -> AUDIO -> QUALITY -> SAVE -> VALIDATE
```

Quality can therefore check scene coverage, duration arithmetic, asset status,
music intensity variety, SFX anchors and the mix policy, in addition to the
script and research.

`VALIDATE` then re-runs schema and narration checks on the saved artefacts, so
a stale or hand-edited file cannot pass unnoticed.

---

## 7. Resumability and invalidation

`production_state.json` records, per stage: status, attempts, timestamps,
duration, error, artefact list and notes. A stage counts as complete only if its
status is `complete` **and** its declared artefacts exist on disk. `--force`
overrides completion.

Regenerating stage X marks every stage in `DOWNSTREAM[X]` as `pending` with an
invalidation note. This is deliberately conservative: changing the brief can
change every later artefact, so all of them are invalidated.

The script stage persists each written section in `Long_Sections.json`; a resume
reuses completed sections instead of paying for them again. This matters when a
single model call can take minutes.

---

## 8. Honest failure

Three rules shape error handling:

1. **No fabricated research.** With no provider, `research_status` is
   `"unavailable"` and the prompt instructs the model to avoid specific figures.
   Provider output is written by Python after the model call, so the model
   cannot upgrade the status or add sources.
2. **No fabricated assets.** An unconfigured provider yields `pending` entries
   with a configuration hint. `validate_render_inputs()` reports every missing
   asset and the renderer refuses to start.
3. **No hidden stage failures.** A failed stage is recorded as `failed` with the
   error and the pipeline stops unless `continue_on_stage_failure` is enabled.
   Doctor warnings are advisory; stage failures are not.

---

## 9. Configuration and prompts

No model parameter, path or credential is hardcoded in engine code. Values are
read through `config/settings.py` from `model_settings.json` and
`pipeline_settings.json`.

Prompts are external text files with `{{PLACEHOLDER}}` markers. Rendering fails
if a placeholder is missing, if a value is `None`, or if any placeholder
survives substitution. This catches wiring mistakes at the first call instead of
sending a half-formed prompt to the model.

The reasoning budget is server-side (`--reasoning-budget 0`). The config records
this and `doctor` reports the effective context; the client also auto-detects
`n_ctx` from `/props`.

---

## 10. Rendering strategy

Rendering is one clip per scene because it is debuggable: a bad scene can be
re-rendered without touching the rest.

```
scene asset + duration + motion
        -> scene_clip_args() -> clip_NNN.mp4 (H.264, yuv420p, exact fps)
        -> concat demuxer (stream copy)
        -> mux_args(): narration anchor + looped music + delayed SFX
                       + amix + alimiter(true peak)
        -> final.mp4
```

`smoke_test()` proves this path with FFmpeg test sources and is part of the test
suite. Production rendering additionally requires every asset to be `ready` and
narration audio to exist.

---

## 11. Deliberate non-goals

- Automatic `llama-server` startup (the app probes instead).
- Publishing to YouTube.
- A general workflow engine.
- Editing or replacing the legacy Kokoro script.
- Storing chain-of-thought in any output.
