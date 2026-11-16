"""PIXEL AI - command line interface.

Examples
--------
    python main.py --list-profiles
    python main.py --profile pixel_tech --topic "The Rise of Artificial Intelligence"
    python main.py produce --profile pixel_history --topic "The Fall of the Roman Empire" --auto-approve
    python main.py resume --project pixel_tech/the-rise-of-artificial-intelligence --approve
    python main.py run-stage --project pixel_tech/the-rise-of-artificial-intelligence --stage quality
    python main.py status  --project pixel_tech/the-rise-of-artificial-intelligence
    python main.py review  --project pixel_tech/the-rise-of-artificial-intelligence
    python main.py validate --project pixel_tech/the-rise-of-artificial-intelligence
    python main.py doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from config.settings import (
    LEGACY_DIR,
    PROJECT_ROOT,
    REPO_ROOT,
    get_config,
)
from engine.channel_profile import ProfileError, ProfileRegistry
from engine.file_manager import FileManager, ProjectPaths, read_json, slugify
from engine.logging_setup import configure_logging, get_logger

COMMANDS = (
    "produce",
    "resume",
    "list-profiles",
    "list-projects",
    "status",
    "run-stage",
    "review",
    "validate",
    "doctor",
    "config",
    "link-tts",
    "verify-tts",
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_APPROVAL = 3

log = get_logger("cli")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="PIXEL AI - multi-channel AI YouTube production engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Backwards-compatible form:\n"
            '  python main.py --profile pixel_tech --topic "A topic"\n'
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="only show warnings and errors")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level",
    )

    sub = parser.add_subparsers(dest="command", required=False)

    # -- produce ---------------------------------------------------------- #
    produce = sub.add_parser("produce", help="run a full production")
    _add_production_args(produce)

    # -- resume ----------------------------------------------------------- #
    resume = sub.add_parser("resume", help="resume an interrupted production")
    resume.add_argument("--project", required=True, help="profile/slug or project path")
    resume.add_argument("--approve", action="store_true", help="approve the script and continue")
    resume.add_argument("--auto-approve", action="store_true", help="skip all approval checkpoints")
    resume.add_argument("--stage", action="append", default=[], help="limit to specific stage(s)")
    resume.add_argument("--assets", action="store_true", help="also run the asset stage")
    resume.add_argument("--render", action="store_true", help="also render video")

    # -- listing ---------------------------------------------------------- #
    sub.add_parser("list-profiles", help="show every channel profile")
    list_projects = sub.add_parser("list-projects", help="show every production project")
    list_projects.add_argument("--profile", default=None, help="filter by profile id")

    # -- inspection ------------------------------------------------------- #
    status = sub.add_parser("status", help="show pipeline state for a project")
    status.add_argument("--project", required=True)
    status.add_argument("--json", action="store_true", help="machine-readable output")

    run_stage = sub.add_parser("run-stage", help="run one or more stages for an existing project")
    run_stage.add_argument("--project", required=True)
    run_stage.add_argument("--stage", action="append", required=True, help="stage name (repeatable)")
    run_stage.add_argument("--auto-approve", action="store_true")
    run_stage.add_argument("--force", action="store_true", help="re-run even if already complete")

    review = sub.add_parser("review", help="human-readable summary of generated output")
    review.add_argument("--project", required=True)

    validate = sub.add_parser("validate", help="validate all artefacts against their schemas")
    validate.add_argument("--project", required=True)

    # -- environment ------------------------------------------------------ #
    sub.add_parser("doctor", help="check server, ffmpeg, profiles, prompts and schemas")

    config_cmd = sub.add_parser("config", help="print the effective configuration")
    config_cmd.add_argument("--json", action="store_true")

    link = sub.add_parser(
        "link-tts",
        help="link this project's scripts into the legacy Kokoro folder",
    )
    link.add_argument("--profile", default=None, help="profile to link (default: all)")

    verify = sub.add_parser(
        "verify-tts",
        help="verify Long.txt/Short.txt with the untouched legacy Kokoro parser",
    )
    verify.add_argument("--project", required=True)
    verify.add_argument(
        "--generate-audio",
        action="store_true",
        help="actually run Kokoro (slow on CPU) instead of parser-only verification",
    )

    return parser


def _add_production_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="channel profile id")
    parser.add_argument("--topic", required=True, help="video topic")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="skip the manual script approval checkpoint",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing state and regenerate everything (asks for confirmation)",
    )
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="run only these stages (repeatable). Implies resume.",
    )
    parser.add_argument("--assets", action="store_true", help="also run the asset stage")
    parser.add_argument("--render", action="store_true", help="also render video with FFmpeg")
    parser.add_argument(
        "--asset-dry-run",
        action="store_true",
        default=True,
        help="plan assets but do not call generation providers (default)",
    )
    parser.add_argument(
        "--asset-generate",
        dest="asset_dry_run",
        action="store_false",
        help="actually call configured asset providers",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _configure(args: argparse.Namespace) -> None:
    config = get_config()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(config.pipeline.logs_dir, level=level, quiet=args.quiet)


def _registry() -> ProfileRegistry:
    return ProfileRegistry(get_config().profiles_dir)


def _file_manager() -> FileManager:
    return FileManager(scripts_dir=get_config().pipeline.scripts_dir)


def _load_profile_or_exit(profile_id: str):
    try:
        return _registry().load(profile_id)
    except ProfileError as exc:
        print(f"Profile error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _resolve_project_or_exit(reference: str) -> ProjectPaths:
    try:
        return _file_manager().resolve(reference)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Project error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _profile_for_project(paths: ProjectPaths):
    manifest = read_json(paths.manifest, default={}) or {}
    profile_id = manifest.get("profile_id") or paths.profile_id
    return _load_profile_or_exit(profile_id)


def _print_table(rows: list[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {
        key: max(len(header), *(len(str(row.get(key, ""))) for row in rows))
        for key, header in columns
    }
    header_line = "  ".join(header.ljust(widths[key]) for key, header in columns)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_list_profiles(_args: argparse.Namespace) -> int:
    registry = _registry()
    rows = registry.describe_all()
    if not rows:
        print(f"No profiles found in {get_config().profiles_dir}")
        return EXIT_OK

    table = []
    for row in rows:
        if not row.get("valid"):
            table.append(
                {
                    "id": row["id"],
                    "name": "INVALID",
                    "category": row.get("error", "")[:40],
                    "long": "",
                    "shorts": "",
                    "visual": "",
                }
            )
            continue
        table.append(
            {
                "id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "long": f"{row['long_minutes']}m",
                "shorts": str(row["shorts"]),
                "visual": row["visual_style"],
            }
        )
    _print_table(
        table,
        [
            ("id", "ID"),
            ("name", "CHANNEL"),
            ("category", "CATEGORY"),
            ("long", "LONG"),
            ("shorts", "SHORTS"),
            ("visual", "VISUAL STYLE"),
        ],
    )
    print(f"\n{len(rows)} profile(s) in {get_config().profiles_dir}")
    return EXIT_OK


def cmd_list_projects(args: argparse.Namespace) -> int:
    manager = _file_manager()
    projects = manager.list_projects(args.profile)
    rows = []
    for project in projects:
        state = read_json(project.state, default={}) or {}
        manifest = read_json(project.manifest, default={}) or {}
        metrics = manifest.get("metrics", {}) or {}
        rows.append(
            {
                "project": f"{project.profile_id}/{project.slug}",
                "topic": project.topic[:40],
                "status": state.get("status", "unknown"),
                "words": metrics.get("long_words", ""),
                "shorts": metrics.get("shorts", ""),
            }
        )
    _print_table(
        rows,
        [
            ("project", "PROJECT"),
            ("topic", "TOPIC"),
            ("status", "STATE"),
            ("words", "WORDS"),
            ("shorts", "SHORTS"),
        ],
    )
    print(f"\n{len(projects)} project(s) in {get_config().pipeline.scripts_dir}")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    paths = _resolve_project_or_exit(args.project)
    state = read_json(paths.state, default={}) or {}
    if not state:
        print(f"No production_state.json in {paths.root}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return EXIT_OK

    print(f"Project : {paths.profile_id}/{paths.slug}")
    print(f"Topic   : {state.get('topic', paths.topic)}")
    print(f"Folder  : {paths.root}")
    print(f"State   : {state.get('status', 'unknown')}")
    approval = state.get("approval", {}) or {}
    print(f"Script approved: {approval.get('script_approved')} ({approval.get('mode') or '-'})")
    print()
    rows = []
    for stage in state.get("stage_order", []):
        record = (state.get("stages") or {}).get(stage, {})
        rows.append(
            {
                "stage": stage,
                "status": record.get("status", "pending"),
                "attempts": record.get("attempts", 0),
                "seconds": record.get("duration_seconds", ""),
                "error": (record.get("error") or "")[:60],
            }
        )
    _print_table(
        rows,
        [
            ("stage", "STAGE"),
            ("status", "STATUS"),
            ("attempts", "TRIES"),
            ("seconds", "SECONDS"),
            ("error", "ERROR"),
        ],
    )
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    config = get_config()
    payload = {
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(REPO_ROOT),
        "model": {
            "base_url": config.model.base_url,
            "model": config.model.model,
            "context_size": config.model.context_size,
            "default_max_tokens": config.model.default_max_tokens,
            "timeout_seconds": config.model.timeout_seconds,
            "max_retries": config.model.max_retries,
            "reasoning_budget": config.model.server_reasoning_budget,
        },
        "pipeline": {
            "scripts_dir": str(config.pipeline.scripts_dir),
            "output_dir": str(config.pipeline.output_dir),
            "logs_dir": str(config.pipeline.logs_dir),
            "words_per_minute": config.pipeline.words_per_minute,
            "long_min_words": config.pipeline.long_min_words,
            "long_max_words": config.pipeline.long_max_words,
            "require_approval": config.pipeline.require_approval_after_script,
            "max_stage_retries": config.pipeline.max_stage_retries,
            "research_provider": config.pipeline.research_provider,
            "image_provider": config.pipeline.image_provider,
            "video_provider": config.pipeline.video_provider,
            "sfx_provider": config.pipeline.sfx_provider,
            "music_provider": config.pipeline.music_provider,
            "render_enabled": config.pipeline.render_enabled,
        },
        "ffmpeg": str(config.pipeline.ffmpeg_path),
        "ffprobe": str(config.pipeline.ffprobe_path),
        "profiles": str(config.profiles_dir),
        "prompts": str(config.prompts_dir),
        "schemas": str(config.schemas_dir),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return EXIT_OK

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return EXIT_OK


def cmd_doctor(_args: argparse.Namespace) -> int:
    config = get_config()
    problems: list[str] = []
    warnings: list[str] = []

    print("PIXEL AI doctor")
    print("=" * 64)

    # -- Python ------------------------------------------------------------ #
    print(f"\n[python] {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 11):
        problems.append("Python 3.11+ is required")

    # -- DeepSeek ---------------------------------------------------------- #
    print("\n[deepseek]")
    from engine.deepseek_client import DeepSeekClient

    client = DeepSeekClient(config.model)
    status = client.probe()
    if status.online:
        print(f"  online   : {status.base_url} ({status.latency_ms} ms)")
        print(f"  model    : {config.model.model}")
        print(f"  context  : {status.context_size} tokens (detected)")
        if status.context_size < config.model.min_context_tokens:
            warnings.append(
                f"Context window is {status.context_size}; recommended >= "
                f"{config.model.min_context_tokens}. Restart llama-server with -c 8192."
            )
        print(f"  timeout  : {config.model.timeout_seconds:.0f}s")
    else:
        print(f"  OFFLINE  : {status.detail}")
        print(client.startup_hint())
        problems.append("DeepSeek server is not reachable")
    client.close()

    # -- FFmpeg ------------------------------------------------------------ #
    print("\n[ffmpeg]")
    for label, path in (("ffmpeg", config.pipeline.ffmpeg_path), ("ffprobe", config.pipeline.ffprobe_path)):
        if path and path.is_file():
            print(f"  {label:8s}: {path}")
        else:
            print(f"  {label:8s}: MISSING ({path})")
            problems.append(f"{label} not found")
    print(f"  render   : {'enabled' if config.pipeline.render_enabled else 'disabled (video.render=false)'}")

    # -- Profiles ---------------------------------------------------------- #
    print("\n[profiles]")
    registry = ProfileRegistry(config.profiles_dir)
    for row in registry.describe_all():
        if row.get("valid"):
            print(
                f"  OK   {row['id']:16s} {row['name']:16s} "
                f"{row['long_minutes']}m / {row['shorts']} shorts"
            )
        else:
            print(f"  FAIL {row['id']:16s} {row.get('error', '')[:60]}")
            problems.append(f"Profile {row['id']} is invalid")

    # -- Prompts ----------------------------------------------------------- #
    print("\n[prompts]")
    from engine.prompt_library import PromptLibrary

    library = PromptLibrary(config.prompts_dir)
    for row in library.describe():
        print(f"  OK   {row['name']:20s} {len(row['placeholders'])} placeholder(s)")
    if len(library.describe()) < 8:
        warnings.append("Fewer than 8 prompt templates found")

    # -- Schemas ----------------------------------------------------------- #
    print("\n[schemas]")
    from engine.json_utils import validate_json

    schema_files = sorted(config.schemas_dir.glob("*.json"))
    for path in schema_files:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            DraftError = validate_json({}, schema)  # noqa: F841 - validates the schema itself
            print(f"  OK   {path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {path.name}: {exc}")
            problems.append(f"Schema {path.name} is invalid")
    if not schema_files:
        problems.append("No schemas found")

    # -- Kokoro ------------------------------------------------------------ #
    print("\n[kokoro]")
    try:
        import kokoro  # noqa: F401

        print(f"  OK   kokoro {getattr(kokoro, '__version__', 'installed')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL kokoro not importable: {exc}")
        problems.append("Kokoro is not importable in this environment")

    legacy_script = LEGACY_DIR / "pixel_tts.py"
    if legacy_script.is_file():
        print(f"  OK   legacy TTS script: {legacy_script}")
    else:
        print(f"  WARN legacy TTS script not found at {legacy_script}")
        warnings.append("Legacy Kokoro script not found")

    # -- Providers --------------------------------------------------------- #
    print("\n[providers]")
    from providers.research import REGISTRY as RESEARCH_REGISTRY
    from providers.audio import get_audio_provider
    from providers.image import get_image_provider
    from providers.video import get_video_provider

    research_name = config.pipeline.research_provider
    research_cls = RESEARCH_REGISTRY.get(research_name)
    research_provider = research_cls() if research_cls else None
    if research_provider is None:
        print(f"  research : {research_name} (UNKNOWN provider)")
        problems.append(f"Unknown research provider '{research_name}'")
    else:
        configured = research_provider.is_configured()
        print(
            f"  research : {research_provider.name} "
            f"({'configured' if configured else 'NOT configured'})"
        )
        if not configured:
            warnings.append(
                "Research provider not configured - Research.json will be marked unavailable"
            )

    for kind, provider in (
        ("image", get_image_provider(config.pipeline.image_provider)),
        ("video", get_video_provider(config.pipeline.video_provider)),
        ("sfx", get_audio_provider("sfx", config.pipeline.sfx_provider)),
        ("music", get_audio_provider("music", config.pipeline.music_provider)),
    ):
        configured = provider.is_configured()
        print(f"  {kind:9s}: {provider.name} ({'configured' if configured else 'NOT configured'})")
        if not configured:
            warnings.append(f"{kind} provider not configured - assets stay pending")

    # -- Writable directories ---------------------------------------------- #
    print("\n[paths]")
    for label, path in (
        ("scripts", config.pipeline.scripts_dir),
        ("output", config.pipeline.output_dir),
        ("logs", config.pipeline.logs_dir),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            print(f"  OK   {label:8s} {path}")
        except OSError as exc:
            print(f"  FAIL {label:8s} {path}: {exc}")
            problems.append(f"{label} directory is not writable")

    # -- Summary ----------------------------------------------------------- #
    print("\n" + "=" * 64)
    if problems:
        print(f"RESULT: {len(problems)} problem(s), {len(warnings)} warning(s)")
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        for warning in warnings:
            print(f"  WARN   : {warning}")
        return EXIT_ERROR
    if warnings:
        print(f"RESULT: ready, with {len(warnings)} warning(s)")
        for warning in warnings:
            print(f"  WARN: {warning}")
        return EXIT_OK
    print("RESULT: everything configured and reachable")
    return EXIT_OK


def cmd_link_tts(args: argparse.Namespace) -> int:
    """Expose new projects to the legacy Kokoro script via directory junctions."""
    config = get_config()
    legacy_scripts = LEGACY_DIR / "scripts"
    if not LEGACY_DIR.is_dir():
        print(f"Legacy folder not found: {LEGACY_DIR}", file=sys.stderr)
        return EXIT_ERROR
    legacy_scripts.mkdir(parents=True, exist_ok=True)

    registry = ProfileRegistry(config.profiles_dir)
    profile_ids = [args.profile] if args.profile else registry.available()
    created = 0

    for profile_id in profile_ids:
        source = config.pipeline.scripts_dir / profile_id
        if not source.is_dir():
            print(f"  skip {profile_id}: no generated scripts yet")
            continue
        target = legacy_scripts / profile_id
        if target.exists():
            print(f"  ok   {target} already exists")
            continue
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            created += 1
            print(f"  link {target} -> {source}")
        else:
            print(f"  FAIL {profile_id}: {result.stdout.strip() or result.stderr.strip()}")

    print(
        f"\n{created} link(s) created.\n"
        "The legacy script now sees these projects as folders. Run:\n"
        f'  python "{LEGACY_DIR / "pixel_tts.py"}" pixel_tech/<topic-slug>'
    )
    return EXIT_OK


def cmd_verify_tts(args: argparse.Namespace) -> int:
    paths = _resolve_project_or_exit(args.project)
    script = Path(__file__).resolve().parent / "tools" / "verify_tts_compat.py"
    command = [sys.executable, str(script), "--project", str(paths.root)]
    if args.generate_audio:
        command.append("--generate-audio")
    return subprocess.call(command)


def cmd_review(args: argparse.Namespace) -> int:
    paths = _resolve_project_or_exit(args.project)
    profile = _profile_for_project(paths)

    print("=" * 68)
    print(f"  {profile.name}  |  {paths.topic}")
    print(f"  {paths.root}")
    print("=" * 68)

    brief = read_json(paths.brief, default={}) or {}
    if brief:
        print("\nBRIEF")
        print(f"  Central message : {brief.get('central_message', '')[:300]}")
        print(f"  Viewer promise  : {brief.get('viewer_promise', '')[:300]}")
        angles = brief.get("angles", []) or []
        print(f"  Angles          : {len(angles)}")
        print(f"  Questions       : {len(brief.get('key_questions', []) or [])}")

    research = read_json(paths.research, default={}) or {}
    if research:
        print("\nRESEARCH")
        print(f"  Status   : {research.get('research_status', '?').upper()}")
        provider = research.get("provider", {}) or {}
        print(f"  Provider : {provider.get('name', 'none')} (configured={provider.get('configured')})")
        print(f"  Sources  : {len(research.get('sources', []) or [])}")
        print(f"  Facts    : {len(research.get('verified_facts', []) or [])}")
        print(f"  Uncertain: {len(research.get('uncertain_claims', []) or [])}")
        if research.get("research_status") == "unavailable":
            print("  NOTE     : no external research - the script is hedged by design")

    outline = read_json(paths.outline, default={}) or {}
    if outline:
        print("\nOUTLINE")
        print(f"  Title    : {outline.get('working_title', '')}")
        print(f"  Target   : {outline.get('target_word_count', '?')} words")
        for section in outline.get("sections", []) or []:
            print(
                f"    {section.get('id', '?'):>3s}  {section.get('title', ''):44s} "
                f"{section.get('estimated_words', '?')}w"
            )

    if paths.long_script.is_file():
        from engine.narration import count_words

        text = paths.long_script.read_text(encoding="utf-8")
        words = count_words(text)
        seconds = words / max(1, config_wpm()) * 60
        print("\nLONG SCRIPT")
        print(f"  Words    : {words}")
        print(f"  Duration : ~{seconds / 60:.1f} min at {config_wpm()} wpm")
        print(f"  Opening  : {text.strip().splitlines()[0][:160]}")

    if paths.short_script.is_file():
        from engine.narration import count_words, parse_shorts_text

        shorts = parse_shorts_text(paths.short_script.read_text(encoding="utf-8"))
        print("\nSHORTS")
        for short in shorts:
            print(
                f"    {short.index:02d}  {short.title:40s} "
                f"{count_words(short.narration):3d} words"
            )

    long_plan = read_json(paths.long_plan, default={}) or {}
    if long_plan:
        from engine.ffmpeg_builder import describe_plan

        summary = describe_plan(long_plan)
        print("\nLONG STORYBOARD")
        print(f"  Scenes   : {summary['scenes']}")
        print(f"  Duration : ~{summary['estimated_seconds'] / 60:.1f} min (estimated)")
        print(f"  Assets   : {', '.join(summary['asset_types'])}")
        print(f"  Pending  : {summary['pending_assets']}")
        coverage = (long_plan.get("coverage") or {}).get("coverage_ratio")
        if coverage is not None:
            print(f"  Coverage : {coverage:.1%}")

    shorts_plan = read_json(paths.shorts_plan, default={}) or {}
    if shorts_plan:
        print("\nSHORTS STORYBOARDS")
        for short in shorts_plan.get("shorts", []) or []:
            print(f"    {short.get('short_id')}  {short.get('title', ''):40s} "
                  f"{len(short.get('scenes', []))} scene(s)")

    audio = read_json(paths.audio_plan, default={}) or {}
    if audio:
        music = audio.get("music", {}) or {}
        print("\nAUDIO")
        print(f"  Music    : {music.get('style', '')} "
              f"({len(music.get('segments', []) or [])} segment(s))")
        print(f"  Effects  : {len(audio.get('sfx', []) or [])} cue(s)")
        mix = audio.get("mix", {}) or {}
        print(f"  Narration priority: {mix.get('narration_priority', '')[:120]}")

    quality = read_json(paths.quality, default={}) or {}
    if quality:
        print("\nQUALITY")
        print(f"  Status   : {quality.get('overall_status', '?').upper()}")
        for name, category in (quality.get("categories") or {}).items():
            print(f"    {name:12s} {category.get('status', '?')}")
        issues = quality.get("issues", []) or []
        if issues:
            print("  Issues:")
            for issue in issues[:12]:
                print(
                    f"    [{issue.get('severity', '?'):7s}] "
                    f"{issue.get('category', ''):12s} {issue.get('message', '')[:90]}"
                )
            if len(issues) > 12:
                print(f"    ... and {len(issues) - 12} more (see QualityReport.json)")
        ai = quality.get("ai_review") or {}
        if ai.get("status") == "completed":
            print(f"  Editorial: {ai.get('verdict', '?')} - "
                  f"{str(ai.get('overall_assessment', ''))[:200]}")

    print("\n" + "=" * 68)
    return EXIT_OK


def config_wpm() -> int:
    return get_config().pipeline.words_per_minute


def cmd_validate(args: argparse.Namespace) -> int:
    from engine.output_validator import validate_project

    paths = _resolve_project_or_exit(args.project)
    profile = _profile_for_project(paths)
    result = validate_project(paths, get_config().schemas_dir, profile)

    print(f"Validating {paths.root}")
    print(f"Checked {len(result.checked)} artefact(s)")
    for warning in result.warnings:
        print(f"  WARN  {warning}")
    if result.failures:
        print(f"\nFAILED with {len(result.failures)} problem(s):")
        for failure in result.failures:
            print(f"  FAIL  {failure}")
        return EXIT_ERROR
    print("\nAll artefacts valid.")
    return EXIT_OK


def _run_production(args: argparse.Namespace, *, resume: bool) -> int:
    from engine.pipeline import ApprovalRequired, Pipeline, PipelineError, ServerNotAvailable

    config = get_config()
    profile = _load_profile_or_exit(args.profile)
    manager = FileManager(scripts_dir=config.pipeline.scripts_dir)
    existed = manager.project_exists(profile.id, args.topic)

    if existed and not resume and not getattr(args, "yes", False):
        print(
            f"\nA project already exists at:\n  {manager.project_dir(profile.id, args.topic)}\n"
            "A fresh run will regenerate the content files in that folder."
        )
        try:
            answer = input("Regenerate from scratch? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("Cancelled. Use --resume semantics by dropping --fresh, or --yes to force.")
            return EXIT_OK

    pipeline = Pipeline(
        config=config,
        profile=profile,
        topic=args.topic,
        resume=resume,
        auto_approve=args.auto_approve,
        force_stages=args.stage if getattr(args, "fresh", False) else [],
        run_assets=getattr(args, "assets", False),
        render_video=getattr(args, "render", False),
        asset_dry_run=getattr(args, "asset_dry_run", True),
    )

    try:
        result = pipeline.run(args.stage or None)
    except ServerNotAvailable as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_ERROR
    except PipelineError as exc:
        print(f"\nPipeline error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted. Resume with:")
        print(f"  python main.py resume --project {profile.id}/{slugify(args.topic)}")
        return EXIT_ERROR
    finally:
        pipeline.close()

    print()
    print("=" * 68)
    if result.status == "complete":
        print(f"PRODUCTION COMPLETE  {result.message}")
        print(f"  Project : {result.paths.root}")
        print(f"  Long    : {result.paths.long_script.name}")
        print(f"  Short   : {result.paths.short_script.name}")
        if result.quality_status:
            print(f"  Quality : {result.quality_status.upper()}")
        print("\nNext steps:")
        print(f'  python main.py review   --project {profile.id}/{result.paths.slug}')
        print(f'  python main.py validate --project {profile.id}/{result.paths.slug}')
        print(f'  python main.py verify-tts --project {profile.id}/{result.paths.slug}')
        return EXIT_OK

    if result.status == "awaiting_approval":
        print("SCRIPT AWAITING APPROVAL")
        print(f"  Project : {result.paths.root}")
        print(f"  {result.message}")
        return EXIT_APPROVAL

    print(f"PRODUCTION FAILED at stage '{result.failed}': {result.message}")
    print(f"  Project : {result.paths.root}")
    print(f"  Resume  : python main.py resume --project {profile.id}/{result.paths.slug}")
    return EXIT_ERROR


def cmd_produce(args: argparse.Namespace) -> int:
    config = get_config()
    manager = FileManager(scripts_dir=config.pipeline.scripts_dir)
    existed = manager.project_exists(args.profile, args.topic)
    resume = not args.fresh
    if args.fresh and existed:
        resume = False
    return _run_production(args, resume=resume)


def cmd_resume(args: argparse.Namespace) -> int:
    from engine.pipeline import Pipeline, PipelineError

    config = get_config()
    paths = _resolve_project_or_exit(args.project)
    manifest = read_json(paths.manifest, default={}) or {}
    profile_id = manifest.get("profile_id") or paths.profile_id
    profile = _load_profile_or_exit(profile_id)
    topic = manifest.get("topic") or paths.topic

    pipeline = Pipeline(
        config=config,
        profile=profile,
        topic=topic,
        resume=True,
        auto_approve=args.auto_approve or args.approve,
        run_assets=args.assets,
        render_video=args.render,
    )
    if args.approve:
        pipeline.approve_script(mode="cli")

    try:
        result = pipeline.run(args.stage or None)
    except PipelineError as exc:
        print(f"\nPipeline error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        pipeline.close()

    if result.status == "complete":
        print(f"\nPRODUCTION COMPLETE ({result.message})")
        print(f"  Project : {result.paths.root}")
        return EXIT_OK
    if result.status == "awaiting_approval":
        print("\nStill awaiting approval.")
        print(f"  {result.message}")
        return EXIT_APPROVAL
    print(f"\nFAILED at '{result.failed}': {result.message}")
    return EXIT_ERROR


def cmd_run_stage(args: argparse.Namespace) -> int:
    from engine.pipeline import Pipeline, PipelineError

    config = get_config()
    paths = _resolve_project_or_exit(args.project)
    manifest = read_json(paths.manifest, default={}) or {}
    profile = _load_profile_or_exit(manifest.get("profile_id") or paths.profile_id)
    topic = manifest.get("topic") or paths.topic

    force = args.stage if args.force else []
    pipeline = Pipeline(
        config=config,
        profile=profile,
        topic=topic,
        resume=True,
        auto_approve=args.auto_approve,
        force_stages=force,
    )
    try:
        result = pipeline.run(args.stage)
    except PipelineError as exc:
        print(f"\nPipeline error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        pipeline.close()

    if result.status == "complete":
        print(f"\nStages complete: {', '.join(result.ran) or '(nothing to do)'}")
        return EXIT_OK
    print(f"\nFAILED at '{result.failed}': {result.message}")
    return EXIT_ERROR


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow the flat form: `python main.py --profile X --topic Y`
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["produce", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    _configure(args)

    handlers = {
        "produce": cmd_produce,
        "resume": cmd_resume,
        "list-profiles": cmd_list_profiles,
        "list-projects": cmd_list_projects,
        "status": cmd_status,
        "run-stage": cmd_run_stage,
        "review": cmd_review,
        "validate": cmd_validate,
        "doctor": cmd_doctor,
        "config": cmd_config,
        "link-tts": cmd_link_tts,
        "verify-tts": cmd_verify_tts,
    }

    try:
        return handlers[args.command](args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - the CLI must never dump a raw traceback
        log.debug("Unhandled error", exc_info=True)
        print(f"\nUnexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Re-run with --log-level DEBUG for details.", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
