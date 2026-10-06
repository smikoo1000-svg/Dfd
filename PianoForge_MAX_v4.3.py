#!/usr/bin/env python3
"""PianoForge MAX v4.3 MAX-FIDELITY 원파일 실행기 — Google Colab / Codespaces / 로컬 Linux.

Colab에서:  이 파일을 업로드한 뒤 셀에서  %run pianoforge_colab.py
            (또는 파일 전체를 셀 하나에 붙여 넣고 실행)
그 외:      python pianoforge_colab.py   → http://localhost:8765

하는 일
1. 시스템 패키지(FluidSynth, GM SoundFont)와 빠진 Python 패키지만 설치한다.
   Colab에 이미 있는 torch/torchaudio/numpy는 건드리지 않는다.
2. 아래 SOURCES에 들어 있는 PianoForge 프로젝트(src/, cli.py, config.yaml, tests/ …)를
   작업 디렉터리에 그대로 써서 import 한다. 파이프라인 로직은 그 프로젝트와 같다.
3. 표준 라이브러리 HTTP 서버로 웹 화면을 띄운다(업로드 → 6단계 변환 → MIDI/WAV/리포트).
   변환이 끝난 곡은 "보정하기"로 2단계 보정을 돌린다(src/refine.py):
   1단계 충실도 복원(piano_fixed.mid) · 선택 2단계 편곡 개선(piano_arranged.mid) · 근거가 담긴 보정 리포트.
4. 접속 경로를 만든다.
   - 공개 터널(기본, Colab): cloudflared 빠른 터널로 https://….trycloudflare.com 주소를 만든다.
     Colab 로그인 없이 휴대폰 어느 브라우저에서도 열린다. 주소에 붙은 접근 키(k=…)가 없으면 401.
   - Colab 셀 출력 안 화면과 Colab 프록시 링크도 함께 출력한다(같은 브라우저에서만 동작).

환경 변수: PIANOFORGE_HOME(작업 폴더), PIANOFORGE_PORT(기본 8765),
PIANOFORGE_SKIP_INSTALL=1(설치 건너뛰기), PIANOFORGE_SOUNDFONT(다른 .sf2),
PIANOFORGE_TUNNEL=cloudflare|none(Colab 기본 cloudflare, 그 외 none), PIANOFORGE_TOKEN(접근 키 고정),
PIANOFORGE_PIANO=salamander|ydp|system(렌더 피아노 음색, 기본 salamander),
PIANOFORGE_SOUNDFONT_WAIT_S=60(렌더 시작 시 음색 다운로드 대기 최대 초; 0이면 즉시 폴백),
PIANOFORGE_DRIVE=1(Colab: 결과·업로드·피아노 음색·접근 키를 Google Drive에 저장, 중간 파일은 런타임 디스크),
PIANOFORGE_CACHE(중간 파일 폴더 직접 지정),
PIANOFORGE_HOST(서버 바인드 주소, 기본 127.0.0.1 — 컨테이너 밖에서 직접 접속하려면 0.0.0.0).
"""
from __future__ import annotations

import builtins
import base64
import hashlib
import logging
import logging.handlers
import collections
import ctypes.util
import html
import platform
import secrets
import stat
import urllib.request
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init for determinism

IN_COLAB = importlib.util.find_spec("google.colab") is not None
IN_KERNEL = "ipykernel" in sys.modules
def _mount_drive() -> Optional[Path]:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:  # no Drive access: keep working on the runtime disk
        print(f"! Google Drive 마운트 실패({exc}) — 런타임 디스크에 저장합니다.")
        return None
    return Path("/content/drive/MyDrive/pianoforge")


def resolve_base(env: Mapping[str, str], in_colab: bool, cwd: Path,
                 mount_drive: Callable[[], Optional[Path]]) -> tuple[Path, Optional[Path]]:
    """(work folder, cache folder override). PIANOFORGE_HOME wins. PIANOFORGE_DRIVE=1 on Colab keeps results
    (finished songs, uploads, piano sounds, access key) on Google Drive so they survive the runtime, while the
    big intermediate files stay on the fast local disk (PIANOFORGE_CACHE overrides that folder)."""
    cache = Path(env["PIANOFORGE_CACHE"]) if env.get("PIANOFORGE_CACHE") else None
    if env.get("PIANOFORGE_HOME"):
        return Path(env["PIANOFORGE_HOME"]), cache
    if in_colab and env.get("PIANOFORGE_DRIVE") == "1":
        drive = mount_drive()
        if drive is not None:
            return drive, cache or Path("/content/pianoforge_cache")
    return (Path("/content/pianoforge") if in_colab else cwd / "pianoforge_app"), cache


BASE, _CACHE_OVERRIDE = resolve_base(os.environ, IN_COLAB, Path.cwd(), _mount_drive)
PROJECT = BASE / "project"
JOBS_DIR = BASE / "jobs"
UPLOADS = BASE / "uploads"
CACHE = _CACHE_OVERRIDE or BASE / "cache"
PERF_PATH = BASE / "perf.json"
LOG_PATH = BASE / "server.log"
PORT = int(os.environ.get("PIANOFORGE_PORT", "8765"))
HOST = os.environ.get("PIANOFORGE_HOST", "127.0.0.1")  # tunnel / Colab proxy / Codespaces all reach loopback
SOUNDFONT = os.environ.get("PIANOFORGE_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2")
MAX_JSON_BYTES = 1 << 20
MAX_UPLOAD_BYTES = 95 * 1024 * 1024  # Cloudflare quick tunnels reject request bodies over 100 MB
TUNNEL = os.environ.get("PIANOFORGE_TUNNEL", "cloudflare" if IN_COLAB else "none")
VERSION = "4.3"
__version__ = VERSION

# ---------------------------------------------------------------------------
# IMPLEMENTED MUSIC FEATURE REGISTRY (v4.2)
# ---------------------------------------------------------------------------
# Only capabilities actually wired into the pipeline are listed here.
IMPLEMENTED_MUSIC_FEATURES: dict[str, dict[str, object]] = {
    "88건반": {"status": "enforced", "range": (21, 108)},
    "음높이(Pitch)": {"status": "observed", "module": "transcription"},
    "음의 길이(Duration)": {"status": "observed", "module": "transcription"},
    "셈여림(Dynamics)": {"status": "estimated", "module": "postprocess"},
    "템포(BPM)": {"status": "estimated", "module": "tempo_map"},
    "MIDI": {"status": "output", "module": "transcription/render"},
    "벨로시티": {"status": "output", "range": (1, 127)},
    "CC64(서스테인 페달 메시지)": {"status": "output", "range": (0, 127)},
    "서스테인 페달": {"status": "estimated", "module": "pedal"},
    "반음": {"status": "derived", "module": "theory"},
    "온음": {"status": "derived", "module": "theory"},
    "음정(Interval)": {"status": "derived", "module": "theory"},
    "음계(Scale)": {"status": "derived", "module": "theory"},
    "장음계": {"status": "derived", "module": "theory"},
    "단음계": {"status": "derived", "module": "theory"},
    "선법(모드)": {"status": "estimated", "module": "theory"},
    "화음(코드)": {"status": "derived", "module": "theory"},
    "기능화성": {"status": "estimated", "module": "theory"},
    "전조(조바꿈)": {"status": "estimated", "module": "theory"},
    "종지법": {"status": "estimated", "module": "theory"},
    "모티프(동기)": {"status": "estimated", "module": "style"},
    "프레이즈(악구)": {"status": "estimated", "module": "style"},
    "프레이징": {"status": "estimated", "module": "style"},
    "아티큘레이션": {"status": "estimated", "module": "postprocess"},
    "보이싱(성부 분리)": {"status": "estimated", "module": "postprocess"},
    "동시발음수(보이스 폴리포니)": {"status": "observed", "module": "postprocess"},
    "폴리리듬": {"status": "estimated", "module": "style/theory"},
    "트릴": {"status": "estimated", "module": "style"},
    "트레몰로": {"status": "estimated", "module": "style"},
    "글리산도": {"status": "estimated", "module": "style"},
    "오스티나토": {"status": "estimated", "module": "style"},
    "음색(Timbre)": {"status": "analyzed", "module": "style/render"},
    "공명(Resonance)": {"status": "rendered", "module": "render"},
    "배음(Overtone)": {"status": "analyzed", "module": "style"},
    "위상 정합(Phase)": {"status": "analyzed", "module": "audio/downmix"},
    "MusicXML 큰보표": {"status": "output", "module": "score"},
    "스테레오 마이킹(A/B·X/Y·ORTF)": {"status": "not_recoverable_exactly", "reason": "MP3에는 실제 마이크 배열 메타데이터가 없음"},
}

def music_capabilities() -> dict[str, dict[str, object]]:
    return {k: dict(v) for k, v in IMPLEMENTED_MUSIC_FEATURES.items()}

MAX_AUDIO_SECONDS = 30 * 60
MIN_FREE_BYTES = int(1.5 * 2**30)   # refuse to start a conversion with less free disk than this
STARTED_AT = time.time()
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"


def _write_secret(path: Path, text: str) -> None:
    """Write the access key readable by the owner only (best effort: Drive / FAT mounts ignore modes)."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


def _load_token() -> str:
    """Access key required on every request (the tunnel URL is public)."""
    if os.environ.get("PIANOFORGE_TOKEN"):
        return os.environ["PIANOFORGE_TOKEN"]
    path = BASE / "access_token"
    try:
        tok = path.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{16,}", tok):
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(18)
    BASE.mkdir(parents=True, exist_ok=True)
    _write_secret(path, tok)
    return tok
AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")
MIDI_EXT = (".mid", ".midi")
OUTPUT_FILES = {
    "piano.mid": "audio/midi", "piano.wav": "audio/wav", "report.json": "application/json",
    "report.md": "text/markdown; charset=utf-8", "run_summary.json": "application/json",
    "piano_fixed.mid": "audio/midi", "piano_fixed.wav": "audio/wav",
    "piano_arranged.mid": "audio/midi", "piano_arranged.wav": "audio/wav",
    "refine_report.md": "text/markdown; charset=utf-8", "refine_report.json": "application/json",
    "piano_preview.mp3": "audio/mpeg", "piano_fixed_preview.mp3": "audio/mpeg",
    "piano_arranged_preview.mp3": "audio/mpeg", "original_preview.mp3": "audio/mpeg",
    "piano.musicxml": "application/vnd.recordare.musicxml+xml", "prompt.txt": "text/plain; charset=utf-8",
    "keywords_report.md": "text/markdown; charset=utf-8", "keywords_report.json": "application/json",
    "piano_theory.mid": "audio/midi", "piano_theory.wav": "audio/wav", "piano_theory_preview.mp3": "audio/mpeg",
    "theory_refine_report.json": "application/json",
}
REFINE_OUTPUTS = ("piano_fixed.mid", "piano_fixed.wav", "piano_arranged.mid", "piano_arranged.wav",
                  "refine_report.md", "refine_report.json", "piano_fixed_preview.mp3", "piano_arranged_preview.mp3")


_EVENT_LOGGER: Optional[logging.Logger] = None
_EVENT_LOGGER_LOCK = threading.Lock()

def event_log(event: str, **fields: Any) -> None:
    """Write JSON events through one long-lived rotating handler."""
    global _EVENT_LOGGER
    with _EVENT_LOGGER_LOCK:
        if _EVENT_LOGGER is None:
            logger = logging.getLogger("pianoforge.events")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=8 * 1024 * 1024, backupCount=3, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            _EVENT_LOGGER = logger
        _EVENT_LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))

def make_mp3(src: Path, out: Path, bitrate: str = "192k") -> Optional[Path]:
    """MP3 copy for in-browser playback (~7x smaller than WAV, so streaming over the tunnel does not
    stall). Uses the ffmpeg binary that Colab ships; without it the page buffers the WAV instead."""
    exe = shutil.which("ffmpeg")
    if exe is None or not src.is_file():
        return None
    tmp = out.with_name(out.name + ".part.mp3")
    proc = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-codec:a", "libmp3lame", "-b:a", bitrate, str(tmp)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        event_log("preview_failed", src=str(src), stderr=proc.stderr[-500:])
        return None
    tmp.replace(out)
    return out


def make_preview(wav: Path) -> Optional[Path]:
    return make_mp3(wav, wav.with_name(wav.stem + "_preview.mp3"))


def probe_audio(path: Path) -> Optional[dict[str, Any]]:
    """{duration_s, sample_rate, channels} of an audio file, or None when no probe tool is available.
    Raises ValueError when a tool exists and the file cannot be decoded (corrupt / not audio)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.duration and info.duration > 0:
            return {"duration_s": float(info.duration), "sample_rate": int(info.samplerate), "channels": int(info.channels)}
    except Exception:  # not readable by libsndfile (e.g. old libsndfile without MP3): try ffprobe
        pass
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    proc = subprocess.run([exe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels:format=duration",
                           "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    try:
        data = json.loads(proc.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        dur = float((data.get("format") or {}).get("duration") or 0.0)
        if proc.returncode != 0 or dur <= 0 or not stream:
            raise ValueError
        return {"duration_s": dur, "sample_rate": int(stream.get("sample_rate") or 0), "channels": int(stream.get("channels") or 0)}
    except (ValueError, TypeError, KeyError):
        raise ValueError("오디오로 읽을 수 없는 파일입니다(손상되었거나 오디오가 아님)") from None


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"

# =========================================================================== helpers (testable, no server state)

STAGE_ORDER = ("normalize", "separate", "transcribe", "postprocess", "render", "evaluate")
PERF_KEEP = 8


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def load_perf(path: Path) -> dict[str, dict[str, list[float]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_perf(path: Path, perf: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(perf), encoding="utf-8")
    tmp.replace(path)


def record_perf(perf: dict[str, Any], key: str, audio_s: float, stages: list[dict[str, Any]]) -> None:
    """Remember seconds of work per second of audio for every stage that really ran (a cache hit is not a
    timing of the work). The last PERF_KEEP runs per device/quality are kept."""
    if audio_s <= 0:
        return
    per = perf.setdefault(key, {})
    for s in stages:
        if s.get("cache_hit") or s.get("name") not in STAGE_ORDER:
            continue
        lst = per.setdefault(s["name"], [])
        lst.append(float(s["duration_s"]) / audio_s)
        del lst[:-PERF_KEEP]


def eta_seconds(perf: dict[str, Any], key: str, audio_s: Optional[float], stage: Optional[str],
                stage_elapsed_s: float, progress: Optional[float] = None) -> Optional[float]:
    """Estimated seconds until the conversion ends: the rest of the current stage plus the typical time of
    every later stage. Needs history for all of them (None on the very first runs). When the stage reports
    progress, its own pace replaces the historical estimate."""
    per = perf.get(key) or {}
    if stage not in STAGE_ORDER or not audio_s or audio_s <= 0:
        return None
    names = STAGE_ORDER[STAGE_ORDER.index(stage):]
    if any(not per.get(n) for n in names):
        return None
    est = {n: _median(per[n]) * audio_s for n in names}
    cur_total = stage_elapsed_s / progress if progress is not None and 0.03 < progress < 1.0 else est[stage]
    return max(0.0, cur_total - stage_elapsed_s) + sum(est[n] for n in names[1:])


def _cuda_batch_size(default: int = 16) -> int:
    """Transcription batch (10 s segments) scaled to the free GPU memory: more throughput on A100/L4-class
    cards, no out-of-memory on small ones. Falls back to the old fixed 16 when the query fails."""
    try:
        import torch

        free_gb = torch.cuda.mem_get_info()[0] / 2**30
    except Exception:
        return default
    return 32 if free_gb >= 20 else 16 if free_gb >= 8 else 8 if free_gb >= 4 else 4


def runtime_fingerprint() -> str:
    """Short id of everything that changes what a cached transcription backend would produce."""
    parts = [VERSION, SOURCES_BUNDLE_SHA256]
    for mod in ("torch", "piano_transcription_inference", "transkun"):
        try:
            parts.append(f"{mod}={importlib.metadata.version(mod)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{mod}=-")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class AuthLimiter:
    """Blocks a client for a while after too many wrong access keys (the key is long and random, so this
    only stops noise and scanners)."""

    def __init__(self, max_failures: int = 20, window_s: float = 60.0) -> None:
        self.max_failures, self.window_s = max_failures, window_s
        self._fails: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, ip: str, now: float) -> "collections.deque[float]":
        q = self._fails.setdefault(ip, collections.deque())
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def allow(self, ip: str, now: Optional[float] = None) -> bool:
        with self._lock:
            return len(self._prune(ip, time.time() if now is None else now)) < self.max_failures

    def fail(self, ip: str, now: Optional[float] = None) -> None:
        with self._lock:
            now = time.time() if now is None else now
            self._prune(ip, now).append(now)


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def token_matches(given: str, expected: str) -> bool:
    return bool(expected) and secrets.compare_digest(given.encode("utf-8", "replace"), expected.encode("utf-8"))


class ProgressBackend:
    """Wraps the transcription model: counts processed segments for the progress bar and stops at the next
    batch when the job was cancelled. Everything else is passed through to the real backend."""

    def __init__(self, inner: Any, job: "Job", total: Optional[int], cancelled_exc: type) -> None:
        self._inner, self._job, self._total, self._exc, self._done = inner, job, total, cancelled_exc, 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def transcribe_batch(self, segments: Any) -> Any:
        if self._job.cancel_requested:
            raise self._exc("사용자가 취소했습니다")
        out = self._inner.transcribe_batch(segments)
        self._done += len(out)
        if self._total:
            self._job.progress = min(0.99, self._done / self._total)
            self._job.progress_label = f"전사 {min(self._done, self._total)}/{self._total}"
        else:
            self._job.progress_label = f"전사 {self._done}개 처리"
        return out


# (import name, pip spec). Installed only when the module is missing.
PIP_PACKAGES: list[tuple[str, str]] = [
    ("demucs", "demucs==4.0.1"),
    ("piano_transcription_inference", "piano_transcription_inference"),
    ("pretty_midi", "pretty_midi==0.2.10"),
    ("mido", "mido==1.3.3"),
    ("fluidsynth", "pyfluidsynth==1.3.4"),
    ("soundfile", "soundfile==0.12.1"),
    ("mir_eval", "mir_eval==0.8.2"),
    ("typer", "typer==0.12.5"),
    ("yaml", "PyYAML==6.0.2"),
    ("structlog", "structlog==24.4.0"),
    ("pytest", "pytest==8.3.3"),
]
# Installed one by one; a failure only disables that feature.
OPTIONAL_PIP_PACKAGES: list[tuple[str, str, str]] = [
    ("transkun", "transkun", "second transcription model (more notes found, fewer false ones)"),
    ("segno", "segno", "QR code for opening the page on a phone"),
]

# Piano SoundFonts (FreePats, https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html).
# Both are Creative Commons Attribution 3.0: credit the authors when you publish renders.
SOUNDFONTS: dict[str, tuple[str, str, str]] = {
    "salamander": ("https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz",
                   "SalamanderGrandPiano.sf2", "Salamander Grand Piano V3 — Yamaha C5, 16 velocity layers "
                   "(Alexander Holm, CC BY 3.0)"),
    "ydp": ("https://freepats.zenvoid.org/Piano/YDP-GrandPiano/YDP-GrandPiano-SF2-20160804.tar.bz2",
            "YDP-GrandPiano.sf2", "YDP Grand Piano — Yamaha Disklavier Pro, 5 velocity layers (FreePats, CC BY 3.0)"),
}
PIANO_SOUND = os.environ.get("PIANOFORGE_PIANO", "salamander")  # salamander | ydp | system
SF_DIR = BASE / "soundfonts"
_SF_THREAD: Optional[threading.Thread] = None


# =========================================================================== installation


def _sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def install_system_packages() -> None:
    have_lib = ctypes.util.find_library("fluidsynth") is not None
    if have_lib and Path(SOUNDFONT).is_file():
        return
    if shutil.which("apt-get") is None:
        print("! apt-get이 없습니다. FluidSynth와 SoundFont를 직접 설치하고 PIANOFORGE_SOUNDFONT를 지정하세요.")
        return
    prefix = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else (["sudo"] if shutil.which("sudo") else [])
    _sh(prefix + ["apt-get", "-qq", "update"])
    _sh(prefix + ["apt-get", "-qq", "install", "-y", "fluidsynth", "fluid-soundfont-gm"])


def install_python_packages() -> None:
    missing = [spec for mod, spec in PIP_PACKAGES if importlib.util.find_spec(mod) is None]
    if importlib.util.find_spec("torch") is None:
        missing += ["torch==2.5.1", "torchaudio==2.5.1"]
    elif importlib.util.find_spec("torchaudio") is None:
        missing.append("torchaudio==" + importlib.metadata.version("torch").split("+")[0])
    try:
        if int(importlib.metadata.version("pydantic").split(".")[0]) < 2:
            missing.append("pydantic==2.9.2")
    except importlib.metadata.PackageNotFoundError:
        missing.append("pydantic==2.9.2")
    if missing:
        _sh([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()
    for mod, spec, what in OPTIONAL_PIP_PACKAGES:
        if importlib.util.find_spec(mod) is None:
            proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"! 선택 패키지 {spec} 설치 실패 — {what} 기능 없이 계속합니다.\n{proc.stderr[-400:]}")
    importlib.invalidate_caches()


def _download_soundfont(name: str) -> Optional[Path]:
    """Download + extract one FreePats piano SF2 into SF_DIR (atomic: .part files never count)."""
    import tarfile

    url, filename, _ = SOUNDFONTS[name]
    dest = SF_DIR / filename
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    SF_DIR.mkdir(parents=True, exist_ok=True)
    archive = SF_DIR / (filename + ".archive.part")
    try:
        print(f"피아노 음색 내려받는 중: {SOUNDFONTS[name][2]}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh, 1 << 20)
        with tarfile.open(archive, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".sf2")]
            if not members:
                raise OSError("archive contains no .sf2 file")
            member = max(members, key=lambda m: m.size)
            src = tf.extractfile(member)
            tmp = dest.with_suffix(".sf2.part")
            with open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            with open(tmp, "rb") as chk:
                head = chk.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"sfbk" or tmp.stat().st_size < 1_000_000:
                tmp.unlink(missing_ok=True)
                raise OSError("not a valid SoundFont (RIFF/sfbk header missing)")
            tmp.replace(dest)
        print(f"피아노 음색 준비 완료: {dest.name} ({dest.stat().st_size // 2**20} MB)", flush=True)
        return dest
    except (OSError, tarfile.TarError, EOFError) as exc:
        print(f"! 피아노 음색 {name} 다운로드 실패({exc}) — 다른 음색을 씁니다.")
        return None
    finally:
        archive.unlink(missing_ok=True)


def start_soundfont_download() -> None:
    """Fetch the preferred piano in the background while the user uploads and the song is analysed."""
    global _SF_THREAD
    if os.environ.get("PIANOFORGE_SOUNDFONT") or PIANO_SOUND == "system":
        return
    order = [PIANO_SOUND] + [n for n in ("salamander", "ydp") if n != PIANO_SOUND]

    def run() -> None:
        for name in order:
            if name in SOUNDFONTS and _download_soundfont(name) is not None:
                return

    _SF_THREAD = threading.Thread(target=run, name="pianoforge-soundfont", daemon=True)
    _SF_THREAD.start()


def best_soundfont(preference: str = "auto", wait_s: float = 0.0) -> tuple[str, str]:
    """(path, label) of the best piano available; waits up to wait_s for a running download."""
    if os.environ.get("PIANOFORGE_SOUNDFONT"):
        return os.environ["PIANOFORGE_SOUNDFONT"], "PIANOFORGE_SOUNDFONT"
    if wait_s > 0 and _SF_THREAD is not None and _SF_THREAD.is_alive():
        _SF_THREAD.join(wait_s)
    pref = PIANO_SOUND if preference == "auto" else preference
    for name in ([pref] if pref in SOUNDFONTS else []) + ["salamander", "ydp"]:
        path = SF_DIR / SOUNDFONTS[name][1]
        if pref != "system" and path.is_file():
            return str(path), SOUNDFONTS[name][2]
    return SOUNDFONT, "FluidR3 GM (system)"



def _sources_build_manifest() -> dict[str, object]:
    """Build a deterministic manifest from the embedded SOURCES artifact.

    This makes the embedded project self-describing and gives cache/debug tools
    one source-of-truth hash instead of a hand-maintained version string.
    """
    import hashlib
    entries = {rel: hashlib.sha256(text.encode("utf-8")).hexdigest() for rel, text in sorted(SOURCES.items())}
    digest = hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"version": VERSION, "file_count": len(entries), "files": entries, "sha256": digest}


SOURCES_BUILD_MANIFEST: dict[str, object] = {}
SOURCES_BUILD_MANIFEST_PATH = BASE / "sources_manifest.json"

def write_sources_manifest() -> None:
    """Persist the exact SOURCES build artifact used by this executable."""
    try:
        tmp = SOURCES_BUILD_MANIFEST_PATH.with_suffix(".json.part")
        tmp.write_text(json.dumps(SOURCES_BUILD_MANIFEST, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(SOURCES_BUILD_MANIFEST_PATH)
    except OSError:
        pass

def write_project() -> None:
    write_sources_manifest()
    for rel, text in SOURCES.items():
        path = PROJECT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]  # re-import after a re-run of this script


# =========================================================================== jobs


# =========================================================================== keyword analysis (v4.3)
# Symbolic detectors for the music-theory / technique keywords. Input is the finished piano.mid
# (notes + sustain pedal) and optionally the music analysis from report.json; output is an evidence
# report. Every item is "detected" (with a count / evidence), "not_found" (searched, absent) or
# "not_recoverable" (cannot be known from audio -> MIDI; the reason is given). Nothing is faked.
import math as _math

_PC = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_SCALES: dict[str, tuple[int, ...]] = {
    "장음계(이오니아)": (0, 2, 4, 5, 7, 9, 11), "단음계(자연)": (0, 2, 3, 5, 7, 8, 10),
    "화성단음계": (0, 2, 3, 5, 7, 8, 11), "도리안": (0, 2, 3, 5, 7, 9, 10), "프리지안": (0, 1, 3, 5, 7, 8, 10),
    "리디안": (0, 2, 4, 6, 7, 9, 11), "믹솔리디안": (0, 2, 4, 5, 7, 9, 10), "로크리안": (0, 1, 3, 5, 6, 8, 10),
    "장 펜타토닉": (0, 2, 4, 7, 9), "단 펜타토닉": (0, 3, 5, 7, 10), "블루스 스케일": (0, 3, 5, 6, 7, 10),
    "온음음계": (0, 2, 4, 6, 8, 10)}
_CHORDS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
    "mM7": (0, 3, 7, 11)}
_DYN = (("pp", 0, 36), ("p", 36, 50), ("mp", 50, 64), ("mf", 64, 80), ("f", 80, 96), ("ff", 96, 128))


def _med(xs):
    s = sorted(xs)
    return 0.0 if not s else s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = _math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def _hist(notes):
    h = [0.0] * 12
    for n in notes:
        h[n.pitch % 12] += round(max(0.05, n.offset_s - n.onset_s), 3)
    return h


def estimate_key(notes):
    """(tonic_pc, 'major'|'minor', correlation) by Krumhansl-Schmuckler on duration-weighted pitch classes."""
    h = _hist(notes)
    if not any(h):
        return 0, "major", 0.0
    if max(h) - min(h) < 0.02 * sum(h) / 12:  # all twelve pitch classes equally used: no key at all
        return 0, "major", 0.0
    tot = sum(h)
    major_iv, minor_iv = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)
    cands = []
    for tonic in range(12):
        rot = [h[(tonic + i) % 12] for i in range(12)]
        for mode, prof, iv in (("major", _KS_MAJOR, major_iv), ("minor", _KS_MINOR, minor_iv)):
            cands.append((_corr(rot, prof), tonic, mode, sum(rot[i] for i in iv) / tot))
    fit = [c for c in cands if c[3] >= 0.97]  # when the notes fit one diatonic set, only those keys compete
    r, tonic, mode, _ = max(fit or cands)
    return tonic, mode, r


def _clusters(notes, win=0.045):
    """Notes whose onsets lie within `win` seconds: chords / simultaneities (list of lists, sorted by pitch)."""
    out, cur = [], []
    for n in sorted(notes, key=lambda x: x.onset_s):
        if cur and n.onset_s - cur[0].onset_s > win:
            out.append(sorted(cur, key=lambda x: x.pitch))
            cur = []
        cur.append(n)
    if cur:
        out.append(sorted(cur, key=lambda x: x.pitch))
    return out


def estimate_beat(notes):
    """Quarter-note length in seconds from the inter-onset histogram (0.5 s when it cannot be told)."""
    ons = sorted({round(c[0].onset_s, 3) for c in _clusters(notes)})
    iois = [b - a for a, b in zip(ons, ons[1:]) if 0.08 <= b - a <= 2.0]
    if len(iois) < 8:
        return 0.5
    best_b, best_s = 0.5, -1.0
    for bpm in range(40, 221):
        b = 60.0 / bpm
        s = 0.0
        for d in iois:
            for mult in (0.25, 0.5, 1.0, 2.0):
                x = d / (b * mult)
                s += max(0.0, 1 - abs(x - round(x)) / 0.12) if round(x) >= 1 else 0.0
        s += 0.0 + (0.5 if 70 <= bpm <= 140 else 0.0)
        if s > best_s:
            best_b, best_s = b, s
    return best_b


def tempo_term_ko(bpm):
    for edge, name in ((50, "라르고(Largo)"), (66, "아다지오(Adagio)"), (92, "안단테(Andante)"), (112, "모데라토(Moderato)"),
                       (144, "알레그로(Allegro)"), (1e9, "프레스토(Presto)")):
        if bpm < edge:
            return name


class _R(dict):
    """One keyword result: {found, count, detail}."""


def _res(count, detail="", **kw):
    return {"found": bool(count), "count": int(count) if not isinstance(count, float) else round(count, 3), "detail": detail, **kw}


# ----------------------------------------------------------------------------------------- detectors
def d_pitch(notes):
    ps = [n.pitch for n in notes]
    inside = sum(1 for p in ps if 21 <= p <= 108)
    return {"88건반": _res(inside == len(ps) and bool(ps), f"{min(ps)}~{max(ps)} (MIDI 21~108 안: {inside}/{len(ps)})") if ps else _res(0),
            "음높이(Pitch)": _res(len(ps), f"음 {len(ps)}개, 최저 {min(ps) if ps else '-'} 최고 {max(ps) if ps else '-'}"),
            "음의 길이(Duration)": _res(len(ps), f"중앙값 {_med([n.offset_s - n.onset_s for n in notes]):.3f}s")}


def d_intervals(notes):
    line = [c[-1].pitch for c in _clusters(notes) if c[-1].pitch >= 55]
    steps = [abs(b - a) for a, b in zip(line, line[1:])]
    semi, whole = sum(1 for s in steps if s == 1), sum(1 for s in steps if s == 2)
    leaps = sum(1 for s in steps if s >= 10)
    hist: dict[int, int] = {}
    for s in steps:
        hist[min(s, 24)] = hist.get(min(s, 24), 0) + 1
    return {"반음": _res(semi, "멜로디 반음 진행"), "온음": _res(whole, "멜로디 온음 진행"),
            "대도약": _res(leaps, "멜로디 도약 ≥ 7도(10반음)"), "음정(Interval)": _res(len(steps), "멜로디 음정 분포(반음수:횟수)", histogram=dict(sorted(hist.items())))}


def d_scale(notes, key):
    tonic, mode, r = key
    h = _hist(notes)
    tot = sum(h) or 1.0
    fits = []
    for name, iv in _SCALES.items():
        for t in range(12):
            cov = sum(h[(t + i) % 12] for i in iv) / tot
            if cov >= 0.96:
                fits.append((len(iv), -cov, t, name))
    fits.sort()
    used = sum(1 for x in h if x / tot > 0.01)
    out = {"음계(Scale)": _res(1 if notes else 0, f"추정 조성 {_PC[tonic]} {mode} (상관 {r:.2f})")}
    for kw, names in (("장음계", ("장음계(이오니아)",)), ("단음계", ("단음계(자연)", "화성단음계")),
                      ("펜타토닉", ("장 펜타토닉", "단 펜타토닉")), ("블루스 스케일", ("블루스 스케일",))):
        hit = [f for f in fits if f[3] in names and f[2] == tonic or (kw in ("펜타토닉", "블루스 스케일") and f[3] in names and used <= 6)]
        hit = [f for f in hit if f[3] in names]
        if kw in ("장음계", "단음계"):
            hit = [f for f in hit if (mode == "major") == (kw == "장음계")]
        out[kw] = _res(len(hit), f"{_PC[hit[0][2]]} {hit[0][3]} 적합 {-hit[0][1]:.0%}" if hit else "")
    modes = [f for f in fits if f[3] in ("도리안", "프리지안", "리디안", "믹솔리디안", "로크리안") and f[2] == tonic and used <= 7]
    mode_names = sorted({f[3] for f in modes})
    out["선법(모드)"] = _res(len(mode_names), ", ".join(mode_names) + " (조성 으뜸음 기준, 상대조 모호성 있음)" if mode_names else "")
    bn = {3: "b3", 6: "b5", 10: "b7"}
    blue = [bn[i] for i in bn if h[(tonic + i) % 12] / tot > 0.02 and h[(tonic + i + (1 if i == 3 else 0)) % 12] > 0 and mode == "major"] if mode == "major" else []
    out["블루노트"] = _res(len(blue), "장조 안의 " + "/".join(blue) if blue else "")
    chrom = sum(h[(tonic + i) % 12] for i in range(12) if i not in (0, 2, 4, 5, 7, 9, 11 if mode == "major" else 10, 3 if mode == "minor" else 4)) / tot
    out["__chromatic_share"] = chrom
    return out


def chord_track(notes, beat_s, tonic):
    """Chord label per half-bar (2 beats) window: list of dicts {t, root, kind, tens, bass, degree}."""
    if not notes:
        return []
    win = beat_s * 2.0
    end = max(n.offset_s for n in notes)
    out = []
    t = min(n.onset_s for n in notes)
    while t < end:
        w = [n for n in notes if n.onset_s < t + win and n.offset_s > t + 0.05]
        t0 = t
        t += win
        if len(w) < 2:
            continue
        pcw = [0.0] * 12
        for n in w:
            pcw[n.pitch % 12] += min(n.offset_s, t0 + win) - max(n.onset_s, t0) + 0.02
        tot = sum(pcw)
        bass = min(w, key=lambda x: x.pitch).pitch % 12
        best = None
        for root in range(12):
            for kind, iv in _CHORDS.items():
                inn = sum(pcw[(root + i) % 12] for i in iv) / tot
                out_w = 1.0 - inn
                sc = inn - 0.9 * out_w - 0.02 * len(iv) + (0.04 if root == bass else 0.0) - (0.03 if kind in ("aug", "sus2", "sus4", "dim7", "mM7") else 0)
                if best is None or sc > best[0]:
                    best = (sc, root, kind, inn)
        if best and best[3] >= 0.78:
            _, root, kind, _ = best
            base_iv = _CHORDS[kind]
            tens = [name for name, i in (("9", 2), ("11", 5), ("13", 9)) if i not in base_iv and pcw[(root + i) % 12] / tot > 0.08
                    and kind in ("7", "maj7", "min7", "m7b5", "maj", "min") and (len(base_iv) == 4 or name == "9") and not (name == "11" and 4 in base_iv)]
            out.append({"t": round(t0, 3), "root": root, "kind": kind, "tens": tens, "bass": bass, "degree": (root - tonic) % 12})
    comp = []
    for c in out:
        if not comp or (comp[-1]["root"], comp[-1]["kind"]) != (c["root"], c["kind"]):
            comp.append(c)
        elif c["tens"] and not comp[-1]["tens"]:
            comp[-1]["tens"] = c["tens"]
    return comp


def d_chords(chords, key, notes, phrase_ends):
    tonic, mode, _ = key
    kinds = [c["kind"] for c in chords]
    res = {"화음(코드)": _res(len(chords), "코드 구간 " + " ".join(f"{_PC[c['root']]}{c['kind']}" for c in chords[:16]) + (" …" if len(chords) > 16 else ""))}
    res["텐션 코드(9th·11th·13th)"] = _res(sum(1 for c in chords if c["tens"]), "; ".join(f"{_PC[c['root']]}{c['kind']}({','.join(c['tens'])})" for c in chords if c["tens"])[:200])
    res["증화음(어그먼트)"] = _res(kinds.count("aug"), "")
    res["감화음(디미니쉬)"] = _res(kinds.count("dim") + kinds.count("dim7"), "dim/dim7")
    fn: dict[str, int] = {"T": 0, "S": 0, "D": 0}
    for c in chords:
        d = c["degree"]
        fn["T" if d in (0, 9, 4, 3, 8) else "S" if d in (5, 2, 1) else "D" if d in (7, 11, 10) else "T"] += 1
    res["기능화성"] = _res(len(chords), f"토닉 {fn['T']} / 서브도미넌트 {fn['S']} / 도미넌트 {fn['D']}")
    for kw, degs in (("으뜸화음(토닉)", (0,)), ("버금딸림화음(서브도미넌트)", (5, 2)), ("딸림화음(도미넌트)", (7,))):
        res[kw] = _res(sum(1 for c in chords if c["degree"] in degs and c["kind"] in ("maj", "min", "7", "min7", "maj7")), "")
    pairs = list(zip(chords, chords[1:]))
    auth = sum(1 for a, b in pairs if a["degree"] == 7 and b["degree"] == 0 and a["kind"] in ("maj", "7") and b["kind"] in ("maj", "min", "maj7", "min7"))
    auth_root = sum(1 for a, b in pairs if a["degree"] == 7 and b["degree"] == 0 and a["bass"] == a["root"] and b["bass"] == b["root"])
    plagal = sum(1 for a, b in pairs if a["degree"] == 5 and b["degree"] == 0)
    deceptive = sum(1 for a, b in pairs if a["degree"] == 7 and b["degree"] in (9, 8))
    half = sum(1 for i in phrase_ends if 0 <= i < len(chords) and chords[i]["degree"] == 7)
    res["종지법"] = _res(auth + plagal + deceptive + half, f"정격 {auth} · 변격 {plagal} · 위종지 {deceptive} · 반종지 {half}")
    res["정격종지"] = _res(auth, f"V→I {auth}회 (둘 다 근음위치: {auth_root}회)")
    res["반종지"] = _res(half, "악구가 V에서 끝남")
    res["위종지"] = _res(deceptive, "V→vi/♭VI")
    iivi = sum(1 for a, b, c in zip(chords, chords[1:], chords[2:])
               if (b["root"] - a["root"]) % 12 == 5 and (c["root"] - b["root"]) % 12 == 5 and a["kind"] in ("min", "min7", "m7b5") and b["kind"] in ("7", "maj") and c["kind"] in ("maj", "maj7", "min", "min7"))
    res["투파이브원(II-V-I)"] = _res(iivi, "")
    secd = sum(1 for a, b in pairs if a["kind"] in ("7", "maj") and (a["root"] - b["root"]) % 12 == 7 and a["degree"] != 7 and b["degree"] != 0 and b["kind"] in ("maj", "min", "min7", "maj7") and a["degree"] not in (0, 5, 2, 9) or
               (a["kind"] == "7" and (a["root"] - b["root"]) % 12 == 7 and a["degree"] != 7 and b["degree"] != 0))
    res["세컨더리 도미넌트"] = _res(secd, "V7/x: 으뜸이 아닌 화음으로 해결되는 도미넌트 7th")
    tri = sum(1 for a, b in pairs if a["kind"] == "7" and (a["root"] - b["root"]) % 12 == 1 and a["degree"] != 1)
    res["대리화음(트라이톤 서브스티튜션)"] = _res(tri, "7th 화음이 반음 아래로 해결(♭II7→I)")
    borrowed = 0
    if mode == "major":
        borrowed = sum(1 for c in chords if (c["degree"] in (8, 10, 3) and c["kind"] in ("maj", "7", "maj7")) or (c["degree"] == 5 and c["kind"] in ("min", "min7")))
    res["모달 인터체인지"] = _res(borrowed, "동명단조 차용(♭VI·♭VII·♭III·iv)" if mode == "major" else "장조 조성에서만 판정")
    unresolved = 0
    resolved = 0
    for a, b in pairs:
        if a["kind"] in ("7", "dim", "dim7", "m7b5"):
            ok = (a["root"] - b["root"]) % 12 in (7, 1) or (b["root"] - a["root"]) % 12 in (1, 3)
            resolved += 1 if ok and b["kind"] in ("maj", "min", "maj7", "min7") else 0
            unresolved += 0 if ok else 1
    res["화성 해결"] = _res(resolved, f"불안정 화음 {resolved + unresolved}개 중 해결 {resolved}")
    return res


def d_dissonance(notes):
    cons = dis = 0
    for c in _clusters(notes, 0.04):
        ps = sorted({n.pitch for n in c})
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                ic = (ps[j] - ps[i]) % 12
                if ic in (1, 2, 6, 10, 11):
                    dis += 1
                else:
                    cons += 1
    tot = cons + dis
    cluster = 0
    for c in _clusters(notes, 0.05):
        ps = sorted({n.pitch for n in c})
        run = 1
        for a, b in zip(ps, ps[1:]):
            run = run + 1 if b - a <= 2 else 1
            if run >= 3:
                cluster += 1
                break
    return {"협화음": _res(cons, f"동시 음정 {tot}쌍 중 {cons / tot:.0%}" if tot else ""), "불협화음": _res(dis, f"{dis / tot:.0%}" if tot else ""),
            "클러스터 화음": _res(cluster, "온음·반음으로 붙은 3음 이상 동시 발음")}


def d_tonality(notes, key):
    h = _hist(notes)
    tot = sum(h) or 1.0
    used = sum(1 for x in h if x / tot > 0.02)
    seq = [c[-1].pitch % 12 for c in _clusters(notes)]
    rows = 0
    i = 0
    while i + 12 <= len(seq):
        if len(set(seq[i:i + 12])) == 12:
            rows += 1
            i += 12
        else:
            i += 1
    atonal = key[2] < 0.5 and used >= 10
    lo = [n for n in notes if n.pitch < 60]
    hi = [n for n in notes if n.pitch >= 60]
    bit, detail = 0, ""
    if len(lo) > 20 and len(hi) > 20:
        kl, kh = estimate_key(lo), estimate_key(hi)
        rel = (kl[0] - kh[0]) % 12
        if kl[2] > 0.75 and kh[2] > 0.75 and rel not in (0, 3, 9):
            hl, hh = _hist(lo), _hist(hi)
            ov = sum(min(a / (sum(hl) or 1), b / (sum(hh) or 1)) for a, b in zip(hl, hh))
            if ov < 0.75:
                bit, detail = 1, f"왼손권 {_PC[kl[0]]} {kl[1]} · 오른손권 {_PC[kh[0]]} {kh[1]}"
    return {"12음기법": _res(rows if rows >= 2 and atonal else 0, f"12음 열 {rows}개" if rows else "", ), "무조성(Atonality)": _res(1 if atonal else 0, f"조성 상관 {key[2]:.2f}, 사용 음이름 {used}/12"),
            "복조성(Bitonality/Polytonality)": _res(bit, detail)}


def d_modulation(notes, key):
    if len(notes) < 40:
        return {"전조(조바꿈)": _res(0)}
    end = max(n.offset_s for n in notes)
    seg = max(6.0, end / 8)
    keys, t = [], 0.0
    while t < end:
        w = [n for n in notes if t <= n.onset_s < t + seg]
        if len(w) >= 12:
            k = estimate_key(w)
            if k[2] >= 0.6:
                keys.append((round(t, 1), k[0], k[1]))
        t += seg
    changes = []
    for a, b in zip(keys, keys[1:]):
        if (a[1], a[2]) != (b[1], b[2]):
            changes.append(f"{b[0]}s {_PC[a[1]]}{a[2][:3]}→{_PC[b[1]]}{b[2][:3]}")
    # a change must persist for two segments to count (single-segment wobble is a borrowed chord, not a modulation)
    real = [c for i, c in enumerate(changes)]
    stable = 0
    for i in range(len(keys) - 2):
        if (keys[i][1], keys[i][2]) != (keys[i + 1][1], keys[i + 1][2]) and (keys[i + 1][1], keys[i + 1][2]) == (keys[i + 2][1], keys[i + 2][2]):
            stable += 1
    return {"전조(조바꿈)": _res(stable, "; ".join(real[:6]))}


def _bar_hist(notes, bars):
    out = []
    for a, b in zip(bars, bars[1:]):
        v = [0.0] * 12
        for n in notes:
            if a <= n.onset_s < b:
                v[n.pitch % 12] += 1
        out.append(v)
    return out


def _cos(a, b):
    na, nb = _math.sqrt(sum(x * x for x in a)), _math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def d_form(notes, bars, key):
    res = {}
    if len(bars) < 9:
        for k in ("2부 형식", "3부 형식", "소나타 형식", "론도 형식", "변주곡", "제시부", "전개부(발전부)", "재현부"):
            res[k] = _res(0, "곡이 짧아 형식 판정 불가")
        return res
    vecs = _bar_hist(notes, bars)
    per = 4
    secs = []
    for i in range(0, len(vecs) - per + 1, per):
        v = [sum(vecs[i + j][p] for j in range(per)) for p in range(12)]
        secs.append(v)
    labels: list[str] = []
    protos: list[list[float]] = []
    for v in secs:
        for li, p in enumerate(protos):
            if _cos(v, p) >= 0.9:
                labels.append(chr(65 + li))
                break
        else:
            protos.append(v)
            labels.append(chr(65 + len(protos) - 1))
    comp = [labels[0]]
    for l in labels[1:]:
        if l != comp[-1]:
            comp.append(l)
    form = "".join(comp)
    n_unique = len(set(form))
    rondo = n_unique >= 3 and form.count(form[0]) >= 3 and all(form[i] == form[0] for i in range(0, len(form), 2))
    res["2부 형식"] = _res(1 if form in ("AB", "AABB", "ABAB") else 0, f"섹션 {form}")
    res["3부 형식"] = _res(1 if form in ("ABA", "ABCA") or (n_unique == 2 and form[0] == form[-1] and len(form) == 3) else 0, f"섹션 {form}")
    res["론도 형식"] = _res(1 if rondo else 0, f"섹션 {form} (A 회귀 {form.count(form[0])}회)")
    var = 0
    if len(secs) >= 6:
        ch = [estimate_key([n for n in notes if bars[i * per] <= n.onset_s < bars[min((i + 1) * per, len(bars) - 1)]]) for i in range(len(secs))]
        bass = [min([n.pitch for n in notes if bars[i * per] <= n.onset_s < bars[min((i + 1) * per, len(bars) - 1)]] or [0]) for i in range(len(secs))]
        var = 1 if n_unique >= 3 and len(set(bass)) <= 3 and sum(1 for i in range(1, len(secs)) if _cos(secs[i], secs[0]) < 0.97 and _cos(secs[i], secs[0]) > 0.6) >= 3 else 0
    res["변주곡"] = _res(var, "같은 저음/화성 골격 위 선율 변화(추정)" if var else "")
    mid = estimate_key([n for n in notes if bars[len(bars) // 3] <= n.onset_s < bars[2 * len(bars) // 3]])
    first = estimate_key([n for n in notes if n.onset_s < bars[len(bars) // 3]])
    last = estimate_key([n for n in notes if n.onset_s >= bars[2 * len(bars) // 3]])
    sonata = first[:2] == last[:2] and mid[:2] != first[:2] and len(bars) >= 32 and mid[2] > 0.5
    res["소나타 형식"] = _res(1 if sonata else 0, f"제시부 {_PC[first[0]]}{first[1][:3]} → 중간 {_PC[mid[0]]}{mid[1][:3]} → 재현 {_PC[last[0]]}{last[1][:3]} (추정)" if sonata else "")
    res["제시부"] = _res(1 if sonata else 0, "곡 앞 1/3 (추정)" if sonata else "")
    res["전개부(발전부)"] = _res(1 if sonata else 0, "곡 가운데 1/3: 조성이 달라짐 (추정)" if sonata else "")
    res["재현부"] = _res(1 if sonata else 0, "곡 뒤 1/3: 으뜸조 복귀 (추정)" if sonata else "")
    res["__form"] = form
    return res


def d_texture(notes):
    if not notes:
        return {}
    end = max(n.offset_s for n in notes)
    samples = [i * 0.1 for i in range(int(end / 0.1))]
    cnt = []
    for t in samples:
        cnt.append(sum(1 for n in notes if n.onset_s <= t < n.offset_s))
    cnt_nz = [c for c in cnt if c > 0]
    mono = sum(1 for c in cnt_nz if c == 1) / len(cnt_nz) if cnt_nz else 0
    cl = _clusters(notes)
    chordal = sum(1 for c in cl if len(c) >= 3) / len(cl)
    top = [c[-1].pitch for c in cl]
    bot = [c[0].pitch for c in cl]
    motion = 0
    ind = 0
    for i in range(1, len(cl)):
        du, db = top[i] - top[i - 1], bot[i] - bot[i - 1]
        if du and db:
            motion += 1
            ind += 1 if (du > 0) != (db > 0) else 0
    contrary = ind / motion if motion else 0
    sync = chordal
    poly = contrary >= 0.4 and sync < 0.35 and sum(1 for c in cnt if c >= 3) / max(1, len(cnt)) > 0.4
    return {"모노포니(단성음악)": _res(1 if mono >= 0.85 else 0, f"한 음만 울리는 시간 {mono:.0%}"),
            "호모포니": _res(1 if chordal >= 0.35 and not poly else 0, f"3음 이상 동시 타건 비율 {chordal:.0%}"),
            "폴리포니(대위법)": _res(1 if poly else 0, f"반진행 비율 {contrary:.0%}, 동시 타건 {sync:.0%}"),
            "텍스처(짜임새)": _res(1, f"평균 동시 발음 {sum(cnt_nz) / len(cnt_nz):.1f}, 최대 {max(cnt)}" if cnt_nz else ""),
            "동시발음수(보이스 폴리포니)": _res(max(cnt) if cnt else 0, f"최대 {max(cnt) if cnt else 0}음 동시")}


def _bass_line(notes, hi=55):
    return [c[0] for c in _clusters(notes) if c[0].pitch < hi]


def d_patterns(notes, beat_s):
    res = {}
    cl = _clusters(notes)
    # trill: >=6 alternating notes within 2 semitones, IOI < 0.13
    seq = [c[-1] for c in cl]  # top line: inner / bass notes of the same chord must not interrupt a figure
    trills = tremolo = gliss = rep = brokenoct = dbl = 0
    i = 0
    while i < len(seq) - 5:
        run = [seq[i]]
        j = i + 1
        while j < len(seq) and seq[j].onset_s - seq[j - 1].onset_s < 0.14:
            run.append(seq[j])
            j += 1
        ps = [n.pitch for n in run]
        k = 0
        while k < len(ps) - 5:  # maximal alternating / stepwise stretches inside the fast run
            e = k + 2
            while e < len(ps) and ps[e] == ps[e - 2] and ps[k] != ps[k + 1]:
                e += 1
            if e - k >= 6:
                d = abs(ps[k] - ps[k + 1])
                if d <= 2:
                    trills += 1
                elif d == 12:
                    brokenoct += 1
                else:
                    tremolo += 1
                k = e
                continue
            e = k + 1
            while e < len(ps) and 0 < ps[e] - ps[e - 1] <= 3:
                e += 1
            f = k + 1
            while f < len(ps) and 0 < ps[f - 1] - ps[f] <= 3:
                f += 1
            end_ = max(e, f)
            if end_ - k >= 7 and run[end_ - 1].onset_s - run[k].onset_s < 1.2:
                gliss += 1
                k = end_
                continue
            k += 1
        i = max(i + 1, j - 1) if len(run) < 6 else j
    for c in cl:
        pass
    # tremolo of chords: alternating two chord shapes quickly
    for a, b, c2, d in zip(cl, cl[1:], cl[2:], cl[3:]):
        if len(a) >= 2 and [n.pitch for n in a] == [n.pitch for n in c2] and [n.pitch for n in b] == [n.pitch for n in d] and \
                [n.pitch for n in a] != [n.pitch for n in b] and d.__class__ and (b[0].onset_s - a[0].onset_s) < 0.12:
            tremolo += 1
    res["트릴"] = _res(trills, "인접 두 음 빠른 교대")
    res["트레몰로"] = _res(tremolo, "두 음/화음의 빠른 교대")
    res["글리산도"] = _res(gliss, "8음 이상 스텝 진행이 1.2초 안")
    bo = 0
    k = 0
    while k < len(seq) - 5:  # the same two pitches an octave apart, alternating at up to 8th-note speed
        e = k + 2
        while e < len(seq) and seq[e].pitch == seq[e - 2].pitch and abs(seq[k].pitch - seq[k + 1].pitch) == 12 and seq[e].onset_s - seq[e - 1].onset_s < 0.4:
            e += 1
        if e - k >= 6:
            bo += 1
            k = e
        else:
            k += 1
    res["브로큰 옥타브"] = _res(max(brokenoct, bo), "옥타브 두 음 교대")
    # repeated notes
    r = 0
    run = 1
    for a, b in zip(seq, seq[1:]):
        if a.pitch == b.pitch and 0.04 < b.onset_s - a.onset_s < 0.4:
            run += 1
            if run == 4:
                r += 1
        else:
            run = 1
    res["동음 연타"] = _res(r, "같은 음 4번 이상 연속")
    # octave playing: simultaneous pairs ic 12 on the top line
    octs = sum(1 for c in cl if len(c) >= 2 and c[-1].pitch - c[0].pitch == 12 and len(c) == 2)
    octs_any = sum(1 for c in cl if any(c[-1].pitch - n.pitch == 12 for n in c[:-1]))
    res["옥타브 주법"] = _res(octs_any if octs_any >= 4 else 0, f"옥타브 동시 발음 {octs_any}회")
    # double notes: consecutive 3rds/6ths pairs
    dn = 0
    run = 0
    for c in cl:
        ok = len(c) == 2 and (c[1].pitch - c[0].pitch) in (3, 4, 8, 9)
        run = run + 1 if ok else 0
        if run == 4:
            dn += 1
    res["더블 노트(3도·6도 주법)"] = _res(dn, "3도/6도 병행 4연속 이상")
    # rolled chords: onsets staggered 15-80 ms, >=3 notes ascending
    roll = 0
    s2 = sorted(notes, key=lambda n: n.onset_s)
    k = 0
    while k < len(s2) - 2:
        g = [s2[k]]
        m = k + 1
        while m < len(s2) and 0.012 <= s2[m].onset_s - s2[m - 1].onset_s <= 0.08 and s2[m].pitch > s2[m - 1].pitch and len(g) < 8:
            g.append(s2[m])
            m += 1
        if len(g) >= 3 and g[-1].onset_s - g[0].onset_s <= 0.25:
            roll += 1
            k = m
        else:
            k += 1
    res["롤링 코드"] = _res(roll, "아래에서 위로 시차를 둔 화음")
    # arpeggio / broken chord: notes of one triad played in sequence (ioi 0.07..0.45)
    arp = 0
    ln = [c[0] for c in cl]
    run = []
    for n in ln:
        if run and (n.onset_s - run[-1].onset_s > 0.5 or n.onset_s - run[-1].onset_s < 0.07):
            run = []
        run.append(n)
        pcs = {x.pitch % 12 for x in run}
        if len(run) >= 4 and len(pcs) <= 4 and (max(x.pitch for x in run) - min(x.pitch for x in run)) >= 7 and len(pcs) >= 3:
            if len(run) == 4:
                arp += 1
    res["분산화음(아르페지오)"] = _res(arp, "")
    # Alberti: low-high-mid-high in bass register
    alb = 0
    bl = [n for n in sorted(notes, key=lambda n: n.onset_s) if n.pitch < 66]
    for a, b, c3, d in zip(bl, bl[1:], bl[2:], bl[3:]):
        if a.pitch < c3.pitch < b.pitch and b.pitch == d.pitch and d.onset_s - a.onset_s < 1.6 and b.pitch - a.pitch <= 12:
            alb += 1
    res["알베르티 베이스"] = _res(alb if alb >= 3 else 0, f"패턴 {alb}회")
    # stride: single low note, then mid chord, alternating
    stride = 0
    run = 0
    for a, b in zip(cl, cl[1:]):
        low, ch = (a, b) if len(a) == 1 else (b, a)
        ok = len(low) == 1 and low[0].pitch < 50 and len(ch) >= 3 and 48 <= ch[0].pitch <= 70 and (len(a) == 1) != (len(b) == 1)
        run = run + 1 if ok else 0
        if run == 6:
            stride += 1
    res["스트라이드 주법"] = _res(stride, "저음 단음 ↔ 중음역 화음 교대")
    # walking bass: >= 6 bass notes, mostly stepwise (<=3), near-beat spacing
    walk = 0
    bs = _bass_line(notes, 55)
    run = 1
    for a, b in zip(bs, bs[1:]):
        step = abs(b.pitch - a.pitch) <= 4 and a.pitch != b.pitch
        gap = b.onset_s - a.onset_s
        run = run + 1 if step and 0.6 * beat_s <= gap <= 1.4 * beat_s else 1
        if run == 6:
            walk += 1
    res["워킹 베이스"] = _res(walk, "")
    boogie = 0
    pat = (0, 4, 7, 9, 10, 9, 7, 4)
    pb = [n.pitch for n in bs]
    for i in range(len(pb) - 8):
        if all(pb[i + k] - pb[i] == pat[k] for k in range(8)):
            boogie += 1
    res["부기우기"] = _res(boogie, "1-3-5-6-♭7-6-5-3 베이스 패턴")
    # comping: short offbeat mid-register chords
    comp = 0
    for c in cl:
        if len(c) >= 3 and 48 <= c[0].pitch <= 72:
            ph = (c[0].onset_s / beat_s) % 1.0
            d = c[0].offset_s - c[0].onset_s
            if 0.35 < ph < 0.65 and d < 0.6 * beat_s:
                comp += 1
    res["컴핑(Comping)"] = _res(comp if comp >= 6 else 0, f"오프비트 짧은 화음 {comp}회")
    # ostinato: repeated pitch-interval pattern of length 3..6 at least 4 times back to back (any voice)
    ost = 0
    iv = [b.pitch - a.pitch for a, b in zip(ln, ln[1:])]
    for L in (2, 3, 4, 6, 8):
        for i in range(len(iv) - 4 * L):
            blk = iv[i:i + L]
            if len(set(blk)) > 1 and all(iv[i + m * L:i + (m + 1) * L] == blk for m in range(4)):
                ost += 1
                break
    res["오스티나토"] = _res(ost, "")
    return res


def d_rhythm(notes, beat_s):
    res = {}
    bpm = 60.0 / beat_s
    res["템포(BPM)"] = _res(1, f"약 {bpm:.0f} BPM · {tempo_term_ko(bpm)}")
    for kw, (lo, hi) in (("라르고", (0, 50)), ("아다지오", (50, 66)), ("안단테", (66, 92)), ("모데라토", (92, 112)), ("알레그로", (112, 144)), ("프레스토", (144, 1e9))):
        res[kw] = _res(1 if lo <= bpm < hi else 0, f"{bpm:.0f} BPM" if lo <= bpm < hi else "")
    cl = _clusters(notes)
    ons = [c[0].onset_s for c in cl]
    durs = {"온음표": 0, "2분음표": 0, "4분음표": 0, "8분음표": 0, "16분음표": 0}
    dotted = 0
    for a, b in zip(ons, ons[1:]):
        q = (b - a) / beat_s
        for name, v in (("온음표", 4.0), ("2분음표", 2.0), ("4분음표", 1.0), ("8분음표", 0.5), ("16분음표", 0.25)):
            if abs(q - v) <= 0.12 * v:
                durs[name] += 1
            if abs(q - 1.5 * v) <= 0.12 * v:
                dotted += 1
    for name, v in durs.items():
        res[name] = _res(v, "IOI 기준 박 길이 일치")
    res["음표"] = _res(len(notes), "")
    res["점음표"] = _res(dotted, "1.5배 길이")
    rest = sum(1 for a, b in zip(cl, cl[1:]) if b[0].onset_s - max(n.offset_s for n in a) > 0.45 * beat_s)
    res["쉼표"] = _res(rest, "소리가 완전히 끊기는 구간")
    trip = 0
    synco = 0
    for band in ([c for c in cl if c[-1].pitch >= 60], [c for c in cl if c[0].pitch < 60]):
        by_beat: dict[int, list[float]] = {}
        for c in band:
            by_beat.setdefault(int(c[0].onset_s // beat_s), []).append((c[0].onset_s % beat_s) / beat_s)
        for ph in by_beat.values():
            if len(ph) == 3 and abs(ph[1] - ph[0] - 1 / 3) < 0.07 and abs(ph[2] - ph[1] - 1 / 3) < 0.07:
                trip += 1
            elif len(ph) == 6 and all(abs((ph[i + 1] - ph[i]) - 1 / 6) < 0.05 for i in range(5)):
                trip += 1
    for c in cl:
        ph = (c[0].onset_s / beat_s) % 1.0
        if 0.4 < ph < 0.6 and max(n.offset_s for n in c) - c[0].onset_s > 0.9 * beat_s and not any(abs(((x[0].onset_s / beat_s) % 1.0) - 0) < 0.1 and c[0].onset_s < x[0].onset_s < c[0].onset_s + beat_s for x in cl[:0]):
            synco += 1
    res["셋잇단음표"] = _res(trip, "한 박에 균등 3분할")
    res["당김음(싱코페이션)"] = _res(synco if synco >= 4 else 0, f"오프비트에서 시작해 다음 박을 넘기는 음 {synco}개")
    hemi = 0
    lo = [c for c in cl if c[0].pitch < 60]
    hi = [c for c in cl if c[-1].pitch >= 60]
    bar = 2 * beat_s * 1.5
    t = 0.0
    end = ons[-1] if ons else 0
    while t < end:
        lw = [c[0].onset_s - t for c in lo if t <= c[0].onset_s < t + bar]
        hw = [c[-1].onset_s - t for c in hi if t <= c[-1].onset_s < t + bar]
        def even(xs, n):
            return len(xs) == n and all(abs((xs[i + 1] - xs[i]) - bar / n) < 0.1 * bar / n for i in range(n - 1))
        if (even(lw, 2) and even(hw, 3)) or (even(lw, 3) and even(hw, 2)):
            hemi += 1
        t += bar
    res["폴리리듬"] = _res(hemi, "양손이 2:3으로 어긋나게 균등 분할")
    res["헤미올라"] = _res(hemi, "3박 마디 안 2:3 재그룹(폴리리듬 근거와 동일, 추정)")
    return res


def d_dynamics(notes, pedals):
    res = {}
    if not notes:
        return res
    end = max(n.offset_s for n in notes)
    wins = []
    t = 0.0
    while t < end:
        v = [n.velocity for n in notes if t <= n.onset_s < t + 2.0]
        wins.append(_med(v) if v else None)
        t += 2.0
    cnt = {k: 0 for k, _, _ in _DYN}
    for w in wins:
        if w is not None:
            for k, lo, hi in _DYN:
                if lo <= w < hi:
                    cnt[k] += 1
    names = {"pp": "피아니시모(pp)", "p": "피아노(p)", "mp": "메조피아노(mp)", "mf": "메조포르테(mf)", "f": "포르테(f)", "ff": "포르티시모(ff)"}
    for k, v in cnt.items():
        res[names[k]] = _res(v, f"2초 구간 {v}개")
    res["셈여림(Dynamics)"] = _res(len([w for w in wins if w is not None]), f"중앙 벨로시티 {_med([n.velocity for n in notes]):.0f}")
    cre = dec = 0
    vs = [w for w in wins if w is not None]
    for i in range(len(vs) - 2):
        if vs[i + 2] - vs[i] >= 12 and vs[i + 1] >= vs[i]:
            cre += 1
        if vs[i] - vs[i + 2] >= 12 and vs[i + 1] <= vs[i]:
            dec += 1
    res["크레셴도"] = _res(cre, "4초 안에 벨로시티 +12 이상")
    res["데크레셴도"] = _res(dec, "4초 안에 벨로시티 -12 이상")
    sfz = fp = 0
    seq = sorted(notes, key=lambda n: n.onset_s)
    for i, n in enumerate(seq):
        loc = [m.velocity for m in seq[max(0, i - 12):i]]
        if loc and n.velocity >= _med(loc) + 30:
            sfz += 1
            nxt = [m.velocity for m in seq[i + 1:i + 6] if m.onset_s - n.onset_s < 1.0]
            if nxt and _med(nxt) <= n.velocity - 25:
                fp += 1
    res["스포르찬도(sfz)"] = _res(sfz, "직전 평균보다 벨로시티 +30 이상")
    res["포르테피아노(fp)"] = _res(fp, "강한 타건 직후 약화")
    # articulation
    cl = _clusters(notes)
    arts = {"legato": 0, "legatissimo": 0, "staccato": 0, "staccatissimo": 0, "tenuto": 0, "portato": 0, "nonlegato": 0, "accent": 0, "marcato": 0, "martellato": 0}
    for a, b in zip(cl, cl[1:]):
        ioi = b[0].onset_s - a[0].onset_s
        if ioi < 0.05 or ioi > 1.5:
            continue
        n = a[-1]
        r = (n.offset_s - n.onset_s) / ioi
        if r >= 1.15:
            arts["legatissimo"] += 1
        if r >= 1.0:
            arts["legato"] += 1
        elif r >= 0.95:
            arts["tenuto"] += 1
        elif r >= 0.75:
            arts["nonlegato"] += 1
        elif r >= 0.45:
            arts["portato"] += 1
        else:
            arts["staccato"] += 1
            if r < 0.25 and n.offset_s - n.onset_s < 0.12:
                arts["staccatissimo"] += 1
    for i, c in enumerate(cl):
        loc = [x[-1].velocity for x in cl[max(0, i - 8):i]]
        if loc and c[-1].velocity >= _med(loc) + 15:
            arts["accent"] += 1
            if c[-1].velocity >= _med(loc) + 20 and (c[-1].offset_s - c[-1].onset_s) < 0.4:
                arts["marcato"] += 1
            if len(c) >= 2 and c[-1].velocity >= 100:
                arts["martellato"] += 1
    for kw, key in (("레가토", "legato"), ("레가티시모", "legatissimo"), ("스타카토", "staccato"), ("스타카티시모", "staccatissimo"), ("테누토", "tenuto"),
                    ("포르타토", "portato"), ("논 레가토", "nonlegato"), ("악센트", "accent"), ("마르카토", "marcato"), ("마르텔라토", "martellato")):
        res[kw] = _res(arts[key], "음 길이 ÷ 다음 타건까지 간격 기준" if key not in ("accent", "marcato", "martellato") else "벨로시티 기준")
    res["아티큘레이션"] = _res(sum(arts[k] for k in ("legato", "staccato", "tenuto", "portato", "nonlegato")), "")
    # fermata: very long note followed by silence/change
    ferm = sum(1 for a, b in zip(cl, cl[1:]) if (max(n.offset_s for n in a) - a[0].onset_s) > 3.0 * max(0.2, _med([x[0].offset_s - x[0].onset_s for x in cl[:200]])) and (max(n.offset_s for n in a) - a[0].onset_s) > 1.5)
    res["페르마타"] = _res(ferm, "중앙값의 3배 이상 길게 유지(템포 완화 포함)")
    return res


def d_pedal(pedals, notes):
    res = {}
    n = len(pedals)
    res["서스테인 페달"] = _res(n, f"페달 구간 {n}개, 총 {sum(p.offset_s - p.onset_s for p in pedals):.1f}s")
    res["CC64(서스테인 페달 메시지)"] = _res(n, "0/127 이진 값으로 기록")
    flutter = 0
    ps = sorted(pedals, key=lambda p: p.onset_s)
    for i in range(len(ps) - 3):
        if ps[i + 3].onset_s - ps[i].onset_s < 2.0 and all(p.offset_s - p.onset_s < 0.45 for p in ps[i:i + 4]):
            flutter += 1
    res["플러터 페달"] = _res(flutter, "2초 안 짧은 페달 4회 이상")
    synco = 0
    ons = sorted(n_.onset_s for n_ in notes)
    import bisect
    for p in ps:
        j = bisect.bisect_left(ons, p.onset_s)
        if j < len(ons) and 0.0 <= ons[j] - p.onset_s <= 0.0 and False:
            pass
    for a, b in zip(ps, ps[1:]):
        gap = b.onset_s - a.offset_s
        if 0 <= gap < 0.12:
            j = bisect.bisect_left(ons, a.offset_s - 0.03)
            if j < len(ons) and ons[j] <= b.onset_s + 0.03:
                synco += 1
    res["싱코페이티드 페달링"] = _res(synco, "건반을 친 직후에 페달을 바꿔 밟음(페달 교체 후 음이 이어짐)")
    for kw, why in (("하프 페달", "CC64를 0/127 이진값으로만 기록 — 중간 값(부분 페달)은 오디오에서 복원하지 않음"),
                    ("쿼터 페달", "CC64 중간 값 미기록"), ("소프트 페달(우나 코르다)", "CC67 미기록(악보에 una corda 제안만 표기)"),
                    ("소스테누토 페달", "CC66 미기록(악보에 sostenuto 제안만 표기)")):
        res[kw] = {"found": False, "count": 0, "detail": why, "status": "not_recoverable"}
    return res


def seq_by_onset(notes):
    return sorted(notes, key=lambda n: n.onset_s)


def d_structure(notes, beat_s):
    res = {}
    cl = _clusters(notes)
    gaps = [0.0] + [b[0].onset_s - max(n.offset_s for n in a) for a, b in zip(cl, cl[1:])]
    cuts = [i for i, g in enumerate(gaps) if g > 1.5 * beat_s]
    phrases = max(1, len(cuts) + 1)
    res["프레이즈(악구)"] = _res(phrases, "쉼 1.5박 이상으로 구분")
    res["프레이징"] = _res(phrases, "")
    mel = [c[-1].pitch for c in cl if c[-1].pitch >= 55]
    ivs = [b - a for a, b in zip(mel, mel[1:])]
    grams: dict[tuple, int] = {}
    for i in range(len(ivs) - 4):
        g = tuple(ivs[i:i + 4])
        if len(set(g)) > 1:
            grams[g] = grams.get(g, 0) + 1
    top = max(grams.items(), key=lambda kv: kv[1], default=(None, 0))
    res["모티프(동기)"] = _res(1 if top[1] >= 3 else 0, f"4음 음정 패턴 {list(top[0])} 이(가) {top[1]}회 반복" if top[1] >= 3 else "")
    # voicing: how many notes sit in the top line, the bass and the inner voices
    top_n = sum(1 for c in cl)
    inner = sum(max(0, len(c) - 2) for c in cl)
    bass_n = sum(1 for c in cl if len(c) >= 2)
    res["보이싱(성부 분리)"] = _res(len(cl), f"상성부 {top_n} · 저성부 {bass_n} · 내성부 {inner}음")
    short = 0
    for a, b in zip(seq_by_onset(notes), seq_by_onset(notes)[1:]):
        if (a.offset_s - a.onset_s) < 0.07 and 0 < b.onset_s - a.onset_s < 0.13 and (b.offset_s - b.onset_s) >= 3 * (a.offset_s - a.onset_s) and abs(b.pitch - a.pitch) <= 2 and a.pitch != b.pitch:
            short += 1
    res["꾸밈음(전타음)"] = _res(short, "아주 짧은 음(<70ms)이 인접한 긴 음 바로 앞에서 울림")
    res["__phrase_cuts"] = cuts
    return res


def d_solo_styles(notes, beat_s, form_str, chords, res_all):
    bpm = 60.0 / beat_s
    cl = _clusters(notes)
    fast = sum(1 for a, b in zip(cl, cl[1:]) if 0.0 < b[0].onset_s - a[0].onset_s < 0.16) / max(1, len(cl))
    runs = res_all.get("virtuoso", 0)
    arp = res_all.get("분산화음(아르페지오)", {}).get("count", 0)
    lines = {
        "독주(솔로)": _res(1, "피아노 한 대 악기로 변환된 결과"),
        "비르투오소": _res(1 if fast > 0.45 and bpm >= 100 else 0, f"빠른 음 비율 {fast:.0%}, {bpm:.0f} BPM"),
        "에튀드(연습곡)": _res(1 if fast > 0.55 else 0, "한 가지 빠른 음형이 곡 전체를 지속(휴리스틱)"),
        "녹턴(야상곡)": _res(1 if bpm < 85 and arp >= 4 else 0, "느린 템포 + 분산화음 반주 + 노래하는 선율(휴리스틱)"),
        "발라드": _res(1 if bpm < 90 and fast < 0.2 else 0, "느림 + 단순한 짜임새(휴리스틱)"),
        "전주곡": _res(1 if arp >= 8 and fast > 0.3 else 0, "한 가지 분산화음 음형 지속(휴리스틱)"),
        "환상곡": _res(0, "자유로운 형식은 악보 정보만으로 판정하지 않음"),
        "즉흥곡": _res(0, "즉흥성은 악보 정보만으로 판정하지 않음"),
        "협주곡": {"found": False, "count": 0, "detail": "오케스트라 파트 없음(피아노 독주 변환)", "status": "not_applicable"},
        "소나타": _res(1 if res_all.get("sonata") else 0, "소나타 형식 판정과 동일(추정)"),
        "칸타빌레": _res(1 if res_all.get("cantabile", 0) >= 0.6 else 0, f"선율 레가토 비율 {res_all.get('cantabile', 0):.0%} (노래하듯, 추정)"),
    }
    return lines


def d_cantabile(notes):
    cl = _clusters(notes)
    top = [c[-1] for c in cl if c[-1].pitch >= 60]
    if len(top) < 8:
        return 0.0
    leg = 0
    tot = 0
    for a, b in zip(top, top[1:]):
        ioi = b.onset_s - a.onset_s
        if 0.2 <= ioi <= 2.0:
            tot += 1
            leg += 1 if a.offset_s >= b.onset_s - 0.03 else 0
    return leg / tot if tot else 0.0


def _virtuosic(notes):
    seq = sorted(notes, key=lambda n: n.onset_s)
    runs, cur = 0, 1
    for a, b in zip(seq, seq[1:]):
        if 0 < b.onset_s - a.onset_s < 0.09:
            cur += 1
            if cur == 8:
                runs += 1
        else:
            cur = 1
    return runs


def d_hands(musicxml_path):
    """Thumb-under and hand-crossing hints from the fingered grand staff (piano.musicxml), if present."""
    out = {"운지 번호": _res(0), "엄지 넘기기": _res(0), "손 교차": _res(0)}
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(musicxml_path)).getroot()
    except Exception:
        return out
    fing = under = cross = 0
    for part in root.iter("part"):
        hands: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
        step_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        for meas in part.iter("measure"):
            hi, lo = [], []
            for note in meas.iter("note"):
                p = note.find("pitch")
                if p is None:
                    continue
                midi = (int(p.findtext("octave", "4")) + 1) * 12 + step_pc[p.findtext("step", "C")] + int(float(p.findtext("alter", "0")))
                staff = int(note.findtext("staff", "1"))
                f = note.find(".//fingering")
                if f is not None and (f.text or "").strip().isdigit():
                    fing += 1
                    hands.setdefault(staff, []).append((midi, int(f.text.strip())))
                (hi if staff == 1 else lo).append(midi)
            if hi and lo and max(lo) > min(hi):
                cross += 1
        for staff, seq in hands.items():
            for (p1, f1), (p2, f2) in zip(seq, seq[1:]):
                asc = p2 > p1 and p2 - p1 <= 4
                desc = p2 < p1 and p1 - p2 <= 4
                if (staff == 1 and asc and f2 == 1 and f1 in (3, 4)) or (staff == 2 and desc and f2 == 1 and f1 in (3, 4)):
                    under += 1
    out["운지 번호"] = _res(fing, "piano.musicxml 에 기입된 운지 수")
    out["엄지 넘기기"] = _res(under, "음계형 진행에서 3·4 → 1")
    out["손 교차"] = _res(cross, "왼손 보표 음이 같은 마디 오른손 최저음보다 높은 마디 수(추정)")
    return out


# ----------------------------------------------------------------------------------------- registry
# status of keywords that are not detectors: where the pipeline already outputs them, what the renderer
# does, and what is physically impossible to recover from an MP3/WAV -> MIDI.
_SCORE = "MusicXML 큰보표(piano.musicxml)에 출력"
_STATIC: dict[str, tuple[str, str]] = {}
for _k in ("큰보표", "높은음자리표", "낮은음자리표", "가온다", "덧줄", "옥타브 기호(8va/8vb)", "조표", "임시표", "올림표(샵)", "내림표(플랫)", "제자리표(내추럴)", "더블샵", "더블플랫",
           "마디", "세로줄", "박자표", "붙임줄", "이음줄", "리타르단도", "아첼레란도", "아 템포", "템포 루바토", "도돌이표", "서스테인 페달 기호"):
    _STATIC[_k] = ("output", _SCORE + " (더블샵/더블플랫은 조표 철자상 필요할 때만)")
for _k in ("볼타", "다 카포(D.C.)", "달 세뇨(D.S.)", "코다(Coda)", "피네(Fine)"):
    _STATIC[_k] = ("not_output", "정확히 같은 구간 반복은 도돌이표로 출력하지만, 반복 지시어(D.C./D.S./Coda/Fine/볼타)는 자동 판정하지 않음 — 연주자의 작곡 의도 정보라 오디오에서 복원 불가")
for _k, _v in {
        "피아노 액션 구조": "악기 내부 구조 — 오디오/MIDI에 없음", "해머": "악기 내부 구조 — 오디오/MIDI에 없음", "이스케이프먼트": "악기 내부 구조 — 오디오/MIDI에 없음",
        "더블 이스케이프먼트": "악기 내부 구조(빠른 연타 가능 여부는 동음 연타 검출로 간접 확인)", "향판(사운드보드)": "악기 내부 구조 — 렌더 음색(SoundFont 샘플)에 이미 포함",
        "브릿지": "악기 내부 구조", "주철 프레임": "악기 내부 구조", "댐퍼": "악기 내부 구조(페달 CC64로 댐퍼 동작만 기록)", "튜닝 핀": "악기 내부 구조", "조율": "녹음 악기의 실제 조율은 MIDI에 없음(렌더는 A4=440Hz)",
        "순정률": "MIDI는 평균율 음높이만 표현", "피타고라스 음률": "MIDI는 평균율 음높이만 표현",
        "인토네이션(해머 보이싱)": "해머 펠트 상태 — 오디오 음색 분석 영역(복원 불가)", "프리페어드 피아노": "특수 주법 음색은 음높이 MIDI로 구분 불가",
        "피아노 현 직접 주법(인사이드 피아노)": "특수 주법 음색은 음높이 MIDI로 구분 불가", "피아노 피치카토": "특수 주법 음색은 음높이 MIDI로 구분 불가",
        "피아노 하모닉스": "특수 주법 음색은 음높이 MIDI로 구분 불가", "애프터터치": "ByteDance/Transkun 모델이 출력하지 않음(피아노 건반은 대부분 애프터터치 없음)",
        "피치벤드": "피아노 음높이는 고정 — 모델 출력 없음", "릴리즈 트리거": "렌더러(FluidSynth) 샘플 기능 — MIDI 정보 아님",
        "물리 모델링": "렌더는 샘플 기반(Salamander 16 벨로시티 레이어) — 물리 모델 합성 아님", "스테레오 마이킹(A/B·X/Y·ORTF)": "MP3에는 실제 마이크 배열 메타데이터가 없음",
        "피아노 리드 각도 조절": "녹음 시 뚜껑 각도 — 오디오에서 복원 불가", "팔 무게 주법": "연주 동작(신체) 정보 — MIDI/오디오로 복원 불가", "릴랙세이션(이완)": "연주 동작(신체) 정보 — 복원 불가",
        "멀티 벨로시티 레이어 샘플링": "렌더 단계에서 사용(Salamander 16단계 벨로시티 레이어)", "위상 정합(Phase)": "정규화 단계에서 스테레오→모노 위상 손실 경고로 점검"}.items():
    _STATIC[_k] = ("not_recoverable" if _k not in ("멀티 벨로시티 레이어 샘플링", "위상 정합(Phase)") else "rendered", _v)
for _k in ("음색(Timbre)", "배음(Overtone)", "공명(Resonance)"):
    _STATIC[_k] = ("analyzed", "음색 분석은 style 단계, 공명은 렌더(홀 리버브·SoundFont 공명)에서 처리 — MIDI 음표 자체에 값이 없음")
for _k in ("MIDI", "벨로시티"):
    _STATIC[_k] = ("output", "piano.mid 로 출력(벨로시티 1~127)")
_STATIC["반음"] = ("derived", "멜로디 반음 진행 검출 참조")
_STATIC["평균율"] = ("rendered", "MIDI 음높이는 12평균율, 렌더는 A4=440Hz 평균율(순정률·피타고라스 음률은 MIDI로 표현 불가)")


def analyze_keywords(notes, pedals, *, bars=None, musicxml=None, beat_s=None):
    """notes: objects with onset_s/offset_s/pitch/velocity; pedals: onset_s/offset_s. Returns the full report."""
    notes = [n for n in notes if n.offset_s > n.onset_s]
    out: dict[str, dict] = {}
    if not notes:
        return {"keywords": {}, "summary": {"error": "음표가 없습니다"}}
    beat = beat_s or estimate_beat(notes)
    key = estimate_key(notes)
    bars = list(bars) if bars and len(bars) >= 2 else [i * beat * 4 for i in range(int(max(n.offset_s for n in notes) / (beat * 4)) + 2)]
    out.update(d_pitch(notes))
    out.update(d_intervals(notes))
    sc = d_scale(notes, key)
    chrom = sc.pop("__chromatic_share")
    out.update(sc)
    chords = chord_track(notes, beat, key[0])
    st = d_structure(notes, beat)
    cuts = st.pop("__phrase_cuts")
    cuts_chord = [max(0, min(len(chords) - 1, int(c * len(chords) / max(1, len(_clusters(notes)))) - 1)) for c in cuts]
    out.update(st)
    out.update(d_chords(chords, key, notes, cuts_chord))
    out.update(d_dissonance(notes))
    out.update(d_tonality(notes, key))
    out.update(d_modulation(notes, key))
    fm = d_form(notes, bars, key)
    form_str = fm.pop("__form", "")
    out.update(fm)
    out.update(d_texture(notes))
    out.update(d_patterns(notes, beat))
    out.update(d_rhythm(notes, beat))
    out.update(d_dynamics(notes, pedals))
    out.update(d_pedal(pedals, notes))
    if musicxml and __import__("os").path.isfile(str(musicxml)):
        out.update(d_hands(musicxml))
    else:
        for k in ("운지 번호", "엄지 넘기기", "손 교차"):
            out[k] = {"found": False, "count": 0, "detail": "piano.musicxml 이 없어 판정하지 못함", "status": "unavailable"}
    iv = (0, 2, 4, 5, 7, 9, 11) if key[1] == "major" else (0, 2, 3, 5, 7, 8, 10, 11)
    acc = sum(1 for n in notes if (n.pitch - key[0]) % 12 not in iv)
    out["임시표"] = _res(acc, f"조표(추정 조성) 밖 음 {acc}개 = {acc / len(notes):.0%} (악보에는 임시표로 출력)")
    for kw in ("올림표(샵)", "내림표(플랫)", "제자리표(내추럴)"):
        out[kw] = {"found": True, "count": 0, "detail": _SCORE + " (조표·임시표 철자에 따라 자동 선택)", "status": "output"}
    cant = d_cantabile(notes)
    ctx = {"virtuoso": _virtuosic(notes), "cantabile": cant, "sonata": out.get("소나타 형식", {}).get("found"),
           "분산화음(아르페지오)": out.get("분산화음(아르페지오)", {})}
    out.update(d_solo_styles(notes, beat, form_str, chords, ctx))
    for k, (status, why) in _STATIC.items():
        if k not in out:
            out[k] = {"found": status in ("output", "rendered", "derived", "analyzed"), "count": 0, "detail": why, "status": status}
    for k, v in out.items():
        if "status" not in v:
            v["status"] = "detected" if v.get("found") else "not_found"
    bpm = 60.0 / beat
    summary = {"조성(추정)": f"{_PC[key[0]]} {key[1]}", "조성 상관": round(key[2], 3), "템포(BPM)": round(bpm, 1), "템포 용어": tempo_term_ko(bpm), "형식": form_str,
               "음표 수": len(notes), "곡 길이(s)": round(max(n.offset_s for n in notes), 1)}
    return {"summary": summary, "keywords": out}


def keywords_markdown(report, wanted):
    """Markdown with one row per requested keyword (in the user's order) so nothing is silently missing."""
    kw = report["keywords"]
    icon = {"detected": "✅ 검출", "not_found": "➖ 이 곡에는 없음", "output": "📄 출력", "rendered": "🎹 렌더", "derived": "🧮 파생",
            "analyzed": "🔬 분석", "not_recoverable": "⛔ 복원 불가", "not_output": "⚠️ 자동 판정 안 함", "not_applicable": "— 해당 없음",
            "unavailable": "… 판정 불가"}
    lines = ["# 키워드 검출 리포트", "", "> 오디오→MIDI 변환은 확률 모델이라 원곡과 100% 일치를 보장할 수 없습니다. 아래는 변환 결과(MIDI)에서 **실제로 근거가 잡힌 것**과 **원리상 복원할 수 없는 것**을 구분한 표입니다. 정확도는 참조 MIDI를 함께 올리면 evaluation 단계의 F-measure로 측정됩니다.", ""]
    for k, v in report["summary"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "| 키워드 | 상태 | 횟수 | 근거 |", "|---|---|---|---|"]
    for k in wanted:
        v = kw.get(k)
        if v is None:
            lines.append(f"| {k} | ⚠️ 미등록 | | |")
        else:
            lines.append(f"| {k} | {icon.get(v['status'], v['status'])} | {v.get('count', '')} | {str(v.get('detail', '')).replace('|', '/')[:160]} |")
    return "\n".join(lines) + "\n"


KEYWORD_LIST: tuple[str, ...] = (
    '큰보표',
    '높은음자리표',
    '낮은음자리표',
    '가온다',
    '덧줄',
    '옥타브 기호(8va/8vb)',
    '조표',
    '임시표',
    '올림표(샵)',
    '내림표(플랫)',
    '제자리표(내추럴)',
    '더블샵',
    '더블플랫',
    '88건반',
    '반음',
    '온음',
    '마디',
    '세로줄',
    '박자표',
    '음표',
    '온음표',
    '2분음표',
    '4분음표',
    '8분음표',
    '16분음표',
    '점음표',
    '쉼표',
    '셋잇단음표',
    '붙임줄',
    '이음줄',
    '당김음(싱코페이션)',
    '템포(BPM)',
    '라르고',
    '아다지오',
    '안단테',
    '모데라토',
    '알레그로',
    '프레스토',
    '리타르단도',
    '아첼레란도',
    '아 템포',
    '템포 루바토',
    '피아니시모(pp)',
    '피아노(p)',
    '메조피아노(mp)',
    '메조포르테(mf)',
    '포르테(f)',
    '포르티시모(ff)',
    '크레셴도',
    '데크레셴도',
    '스포르찬도(sfz)',
    '포르테피아노(fp)',
    '레가토',
    '레가티시모',
    '스타카토',
    '스타카티시모',
    '포르타토',
    '테누토',
    '악센트',
    '마르카토',
    '마르텔라토',
    '논 레가토',
    '페르마타',
    '서스테인 페달',
    '소프트 페달(우나 코르다)',
    '소스테누토 페달',
    '하프 페달',
    '쿼터 페달',
    '플러터 페달',
    '싱코페이티드 페달링',
    '운지 번호',
    '엄지 넘기기',
    '손 교차',
    '팔 무게 주법',
    '릴랙세이션(이완)',
    '칸타빌레',
    '화음(코드)',
    '롤링 코드',
    '분산화음(아르페지오)',
    '알베르티 베이스',
    '스트라이드 주법',
    '보이싱(성부 분리)',
    '동음 연타',
    '옥타브 주법',
    '브로큰 옥타브',
    '더블 노트(3도·6도 주법)',
    '대도약',
    '트릴',
    '트레몰로',
    '꾸밈음(전타음)',
    '글리산도',
    '폴리리듬',
    '헤미올라',
    '오스티나토',
    '프레이징',
    '아티큘레이션',
    '도돌이표',
    '볼타',
    '다 카포(D.C.)',
    '달 세뇨(D.S.)',
    '코다(Coda)',
    '피네(Fine)',
    '음높이(Pitch)',
    '음의 길이(Duration)',
    '셈여림(Dynamics)',
    '음색(Timbre)',
    '배음(Overtone)',
    '공명(Resonance)',
    '음정(Interval)',
    '음계(Scale)',
    '장음계',
    '단음계',
    '펜타토닉',
    '블루스 스케일',
    '선법(모드)',
    '기능화성',
    '으뜸화음(토닉)',
    '버금딸림화음(서브도미넌트)',
    '딸림화음(도미넌트)',
    '종지법',
    '정격종지',
    '반종지',
    '위종지',
    '전조(조바꿈)',
    '협화음',
    '불협화음',
    '화성 해결',
    '텐션 코드(9th·11th·13th)',
    '투파이브원(II-V-I)',
    '대리화음(트라이톤 서브스티튜션)',
    '세컨더리 도미넌트',
    '모달 인터체인지',
    '증화음(어그먼트)',
    '감화음(디미니쉬)',
    '클러스터 화음',
    '12음기법',
    '무조성(Atonality)',
    '복조성(Bitonality/Polytonality)',
    '모티프(동기)',
    '프레이즈(악구)',
    '2부 형식',
    '3부 형식',
    '소나타 형식',
    '론도 형식',
    '변주곡',
    '제시부',
    '전개부(발전부)',
    '재현부',
    '텍스처(짜임새)',
    '모노포니(단성음악)',
    '호모포니',
    '폴리포니(대위법)',
    '소나타',
    '녹턴(야상곡)',
    '에튀드(연습곡)',
    '발라드',
    '환상곡',
    '즉흥곡',
    '전주곡',
    '협주곡',
    '비르투오소',
    '독주(솔로)',
    '피아노 액션 구조',
    '해머',
    '이스케이프먼트',
    '더블 이스케이프먼트',
    '향판(사운드보드)',
    '브릿지',
    '주철 프레임',
    '댐퍼',
    '튜닝 핀',
    '조율',
    '평균율',
    '순정률',
    '피타고라스 음률',
    '인토네이션(해머 보이싱)',
    '프리페어드 피아노',
    '피아노 현 직접 주법(인사이드 피아노)',
    '피아노 피치카토',
    '피아노 하모닉스',
    '컴핑(Comping)',
    '워킹 베이스',
    '부기우기',
    '블루노트',
    'MIDI',
    '벨로시티',
    'CC64(서스테인 페달 메시지)',
    '애프터터치',
    '피치벤드',
    '멀티 벨로시티 레이어 샘플링',
    '릴리즈 트리거',
    '물리 모델링',
    '동시발음수(보이스 폴리포니)',
    '스테레오 마이킹(A/B·X/Y·ORTF)',
    '피아노 리드 각도 조절',
    '위상 정합(Phase)',
)



# =========================================================================== theory-guided refinement (v4.3)
# Uses the keyword analysis above as evidence to clean the transcription: the key, chords and repeated
# patterns tell which detections are probably ghosts and which expected notes were missed. It is
# deliberately conservative: a note is removed only when several independent signals agree, and notes are
# added only when a pattern repeats in at least two other places. The result is written as piano_theory.mid
# NEXT TO piano.mid: the original is never overwritten, and with a reference MIDI both are scored.
from dataclasses import dataclass as _dc


@_dc(frozen=True)
class TNote:
    onset_s: float
    offset_s: float
    pitch: int
    velocity: int


def note_f1(est, ref, tol=0.05):
    """(precision, recall, F1) of note onsets: same pitch and onset within tol seconds (greedy 1:1)."""
    by_pitch: dict[int, list[float]] = {}
    for n in ref:
        by_pitch.setdefault(n.pitch, []).append(n.onset_s)
    for v in by_pitch.values():
        v.sort()
    used: dict[int, set[int]] = {}
    tp = 0
    for n in sorted(est, key=lambda x: x.onset_s):
        cand = by_pitch.get(n.pitch, [])
        u = used.setdefault(n.pitch, set())
        best, bd = None, tol + 1e-9
        for i, t in enumerate(cand):
            if i not in u and abs(t - n.onset_s) < bd:
                best, bd = i, abs(t - n.onset_s)
        if best is not None:
            u.add(best)
            tp += 1
    p = tp / len(est) if est else 0.0
    r = tp / len(ref) if ref else 0.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def _diatonic(tonic, mode):
    iv = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 9, 10, 11)  # minor: natural + raised 6th/7th
    return {(tonic + i) % 12 for i in iv}


def theory_refine(notes, *, beat_s=None, bars=None):
    """Returns (new_notes, info). info = {removed: [...], added: [...], reasons: {...}}."""
    notes = sorted(notes, key=lambda n: (n.onset_s, n.pitch))
    if len(notes) < 40:
        return list(notes), {"removed": [], "added": [], "reasons": {}, "skipped": "음표가 너무 적어 보정하지 않음"}
    beat = beat_s or estimate_beat(notes)
    end = max(n.offset_s for n in notes)
    med_vel = _med([n.velocity for n in notes])
    # local keys (window ~ 8 beats*2); a window without a clear key gets no key-based removal at all
    win = max(6.0, beat * 16)
    seg_keys = []
    t = 0.0
    while t < end:
        w = [n for n in notes if t <= n.onset_s < t + win]
        k = estimate_key(w) if len(w) >= 16 else None
        seg_keys.append((t, t + win, k if k and k[2] >= 0.7 else None))
        t += win
    def key_at(x):
        for a, b, k in seg_keys:
            if a <= x < b:
                return k
        return None
    chords = chord_track(notes, beat, 0)
    ctimes = [c["t"] for c in chords]
    import bisect
    def chord_pcs_at(x):
        if not chords:
            return None
        i = bisect.bisect_right(ctimes, x) - 1
        if i < 0 or x - ctimes[i] > beat * 4:
            return None
        c = chords[i]
        return {(c["root"] + j) % 12 for j in _CHORDS[c["kind"]]}
    pitch_count: dict[int, int] = {}
    for n in notes:
        pitch_count[n.pitch] = pitch_count.get(n.pitch, 0) + 1
    removed: list[int] = []
    reasons: dict[str, int] = {"조성 밖 약한 고립음": 0, "옥타브 유령음": 0}
    for i, n in enumerate(notes):
        dur = n.offset_s - n.onset_s
        weak = dur < 0.09 or n.velocity < med_vel - 22
        k = key_at(n.onset_s)
        if weak and k and (n.pitch - k[0]) % 12 not in _diatonic(k[0], k[1]):
            cp = chord_pcs_at(n.onset_s)
            prev_n = [m for m in notes[max(0, i - 6):i] if abs(m.onset_s - n.onset_s) < 0.5 and m is not n]
            nxt_n = [m for m in notes[i + 1:i + 7] if abs(m.onset_s - n.onset_s) < 0.5]
            stepwise = any(abs(m.pitch - n.pitch) <= 1 for m in prev_n) and any(abs(m.pitch - n.pitch) <= 2 for m in nxt_n)
            if (cp is None or n.pitch % 12 not in cp) and pitch_count[n.pitch] <= 2 and not stepwise:
                removed.append(i)
                reasons["조성 밖 약한 고립음"] += 1
                continue
        for m in notes[max(0, i - 8):i + 9]:
            if m is not n and abs(m.onset_s - n.onset_s) < 0.03 and (m.pitch - n.pitch) in (12, -12, 19, -19, 24, -24) \
                    and n.velocity < 0.55 * m.velocity and dur <= (m.offset_s - m.onset_s) * 1.1:
                removed.append(i)
                reasons["옥타브 유령음"] += 1
                break
    rm = set(removed)
    kept = [n for i, n in enumerate(notes) if i not in rm]
    # ---- fill one missing note in a bar whose neighbours (>=2 bars) repeat the same pattern with that note
    added: list[TNote] = []
    if bars and len(bars) > 6:
        tol = 0.045
        def contains(sig, item):
            return any(p == item[1] and abs(ph - item[0]) <= tol for ph, p, _ in sig)
        for lo_band in (True, False):  # left-hand and right-hand registers repeat their patterns independently
            sigs = [[(n.onset_s - a, n.pitch, n) for n in kept if a <= n.onset_s < b and (n.pitch < 60) == lo_band]
                    for a, b in zip(bars, bars[1:])]
            for i, sig in enumerate(sigs):
                if len(sig) < 4:
                    continue
                votes: dict[tuple[int, int], list] = {}
                for j in range(max(0, i - 8), min(len(sigs), i + 9)):
                    if j == i or len(sigs[j]) < 6 or len(sigs[j]) - len(sig) != 1:
                        continue
                    if all(contains(sigs[j], (ph, p)) for ph, p, _ in sig):
                        miss = [(ph, p, nn) for ph, p, nn in sigs[j] if not contains(sig, (ph, p))]
                        if len(miss) == 1:
                            ph, p, nn = miss[0]
                            votes.setdefault((round(ph / tol), p), []).append((ph, nn))
                for (_, p), vs in votes.items():
                    if len(vs) >= 3:
                        ph = _med([v[0] for v in vs])
                        src = vs[0][1]
                        added.append(TNote(round(bars[i] + ph, 4), round(bars[i] + ph + (src.offset_s - src.onset_s), 4), p,
                                           int(_med([v[1].velocity for v in vs]))))
    out = sorted(kept + added, key=lambda n: (n.onset_s, n.pitch))
    return out, {"removed": [(round(notes[i].onset_s, 3), notes[i].pitch) for i in removed],
                 "added": [(a.onset_s, a.pitch) for a in added], "reasons": reasons, "notes_in": len(notes), "notes_out": len(out)}


def write_theory_midi(src: Path, dest: Path, notes: list) -> None:
    """piano.mid with its notes replaced (tempo map, pedal CC and everything else is kept)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(src))
    inst = next(i for i in pm.instruments if not i.is_drum)
    inst.notes = [pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch), start=float(n.onset_s), end=float(n.offset_s)) for n in notes]
    tmp = dest.with_name(dest.name + ".part")
    pm.write(str(tmp))
    tmp.replace(dest)


_SAVE_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    name: str
    audio: str
    reference: Optional[str]
    options: dict[str, Any]
    created: float = field(default_factory=time.time)
    state: str = "queued"  # queued | running | done | failed | cancelled
    stage: Optional[str] = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    finished: Optional[float] = None
    refine_state: Optional[str] = None  # None | queued | running | done | failed
    refine_arrange: bool = False
    refine_error: Optional[str] = None
    refine_summary: dict[str, Any] = field(default_factory=dict)
    duration_s: Optional[float] = None       # length of the song (from the upload check)
    size_bytes: Optional[int] = None
    started: Optional[float] = None
    stage_started: Optional[float] = None
    progress: Optional[float] = None         # 0..1 inside the current stage (transcription reports it)
    progress_label: Optional[str] = None
    cancel_requested: bool = False
    intermediates_cleared: bool = False      # big intermediate files were deleted: refinement is unavailable

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def out(self) -> Path:
        return self.dir / "out"

    def save(self) -> None:
        with _SAVE_LOCK:  # the worker and HTTP threads both save; they must not share one tmp file
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / "job.json.tmp"
            tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.dir / "job.json")

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("audio")
        d["reference"] = Path(self.reference).name if self.reference else None
        d["files"] = [n for n in OUTPUT_FILES if (self.out / n).is_file()]
        return d


class App:
    def __init__(self) -> None:
        from src import pipeline as pl
        from src import refine as rf
        from src import render as rd
        from src import transcribe as tr

        self.pl, self.tr, self.rf, self.rd = pl, tr, rf, rd
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.jobs_lock = self.lock
        self.queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.backends: dict[str, Any] = {}
        self.base_cfg = pl.update_config(pl.load_config(PROJECT / "config.yaml"), {
            "cache.dir": str(CACHE), "render.soundfont_path": SOUNDFONT})
        self.device = tr.resolve_device(self.base_cfg.runtime.device)
        self.perf = load_perf(PERF_PATH)
        self._load_jobs()
        threading.Thread(target=self._worker, name="pianoforge-worker", daemon=True).start()

    def _job_get(self, job_id: str) -> Optional[Job]:
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def _jobs_snapshot(self) -> list[Job]:
        with self.jobs_lock:
            return list(self.jobs.values())

    # ---------------------------------------------------------------- config / backend cache
    def config_for(self, options: dict[str, Any]) -> Any:
        upd: dict[str, Any] = {}
        # v4.2 MAX-FIDELITY default: explicit mix-minus-drums source. A dedicated piano stem is opt-in.
        # A dedicated piano stem is optional, not the recommended default, because many songs
        # contain little/no piano and htdemucs_6s can then suppress useful melody/bass/harmony.
        src = options.get("piano_source", "nodrums")
        if src in ("nodrums", "residual"):
            # Both names mean the same thing in the UI/API: original mix minus drums.
            upd.update({"separation.piano_source": "residual", "separation.model": "htdemucs_ft"})
        elif src == "other":
            upd["separation.piano_source"] = "other"
        elif src == "piano":
            upd.update({"separation.model": "htdemucs_6s", "separation.piano_source": "piano"})
        elif src == "mix":
            upd.update({"separation.enabled": False, "separation.piano_source": "mix"})
        elif src != "nodrums":  # default: mix minus drums (config.yaml)
            raise self.pl.ConfigError(f"unknown piano_source {src!r}")
        if options.get("quality", "max") == "fast":  # previous, lighter analysis
            upd.update({"separation.model": "htdemucs" if src != "piano" else "htdemucs_6s", "separation.shifts": 1,
                        "transcription.extra_sources": [], "transcription.chunk_offset_tta": False,
                        "transcription.gap_fill_mode": "silence", "transcription.onset_threshold": 0.3})
        elif options.get("quality", "max") != "max":
            raise self.pl.ConfigError(f"unknown quality {options.get('quality')!r}")
        if options.get("quality", "max") == "fast":
            upd["transcription.second_model"] = "none"
        for key, allowed, cfg_key in (("piano_type", ("grand", "felt", "upright"), "render.piano_type"),
                                      ("texture", ("clean", "lofi"), "render.texture"),
                                      ("mech", ("none", "light", "normal"), "render.mechanical_noise")):
            val = options.get(key, allowed[0])
            if val not in allowed:
                raise self.pl.ConfigError(f"unknown {key} {val!r}")
            upd[cfg_key] = val
        room = options.get("room", "hall")
        rooms = {"room": (1.2, 0.16), "hall": (1.8, 0.22), "large": (2.6, 0.28)}
        if room == "dry":
            upd["render.hall_reverb"] = False
        elif room in rooms:
            upd.update({"render.hall_rt60_s": rooms[room][0], "render.hall_wet": rooms[room][1]})
        else:
            raise self.pl.ConfigError(f"unknown room {room!r}")
        notes_mode = options.get("notes", "balanced")
        if notes_mode == "strict":
            upd.update({"transcription.note_filter": "strict", "transcription.onset_threshold": 0.24,
                        "transcription.keep_votes": 2, "transcription.single_confidence": 0.64,
                        "transcription.single_support_db": -17.0, "transcription.evidence_floor_db": -34.0,
                        "transcription.max_chord_notes": 8, "transcription.max_polyphony": 14})
        elif notes_mode == "max":
            upd.update({"transcription.note_filter": "balanced", "transcription.keep_votes": 2,
                        "transcription.single_confidence": 0.42, "transcription.single_support_db": -23.0,
                        "transcription.evidence_floor_db": -40.0, "transcription.max_chord_notes": 10,
                        "transcription.max_polyphony": 16})
        elif notes_mode != "balanced":
            raise self.pl.ConfigError(f"unknown notes mode {notes_mode!r}")
        else:
            upd.update({"transcription.note_filter": "balanced", "transcription.keep_votes": 2,
                        "transcription.single_confidence": 0.50, "transcription.single_support_db": -20.0,
                        "transcription.evidence_floor_db": -40.0, "transcription.max_chord_notes": 9,
                        "transcription.max_polyphony": 16})
        # v4.2 MAX-FIDELITY: independent multi-source evidence is enabled, but other stems are NOT converted into piano notes by default.
        # The default ensemble is ByteDance normal/TTA + optional Transkun on the same source.
        # The MAX evidence block must not undo the lighter "fast" settings chosen above (it used to:
        # fast still transcribed vocals+bass+other and filled gaps, so it was barely faster than max).
        fast = options.get("quality", "max") == "fast"
        if fast:
            upd.update({"transcription.octave_relocate": True, "transcription.noise_filter": True,
                        "transcription.music_cleanup": True,  # note-filter strictness stays with the chosen notes mode
                        "postprocess.snap.strength": 0.34, "postprocess.snap.max_shift_s": 0.024})
        else:
          upd.update({"transcription.extra_sources": ["vocals", "bass", "other"],
                    "transcription.second_model_sources": ["piano_source", "vocals", "other"],
                    "transcription.gap_fill_mode": "uncovered", "transcription.gap_fill_min_gap_s": 0.30,
                    "transcription.gap_fill_min_voiced_prob": 0.68, "transcription.gap_fill_floor_db": -45.0,
                    "transcription.music_cleanup": True, "transcription.keep_votes": 3,
                    "transcription.max_chord_notes": 10, "transcription.max_polyphony": 18,
                    "transcription.octave_relocate": True, "transcription.noise_filter": True,
                    "postprocess.snap.strength": 0.34, "postprocess.snap.max_shift_s": 0.024})
        if self.device == "cuda" and options.get("device", "auto") in ("auto", "cuda"):
            upd["transcription.batch_size"] = _cuda_batch_size()
        upd["render.soundfont_path"] = best_soundfont(options.get("piano", "auto"))[0]
        upd["runtime.device"] = options.get("device", "auto")
        upd["postprocess.pedal.mode"] = "cc" if options.get("pedal", True) else "drop"
        upd["postprocess.snap.enabled"] = bool(options.get("snap", True))
        return self.pl.update_config(self.base_cfg, upd)

    def _backend(self, cfg: Any, device: str) -> Any:
        key = json.dumps([cfg.transcription.model_dump(mode="json"), str(cfg.cache.dir), device, runtime_fingerprint()], sort_keys=True, default=str)
        if key not in self.backends:
            # Keep multiple compatible backends cached. Rebuilding the model on every option change
            # is much slower than the small extra VRAM/RAM cost, and it makes Colab presets painful.
            self.backends[key] = self.pl.build_backend(cfg, device)
        return self.backends[key]

    # ---------------------------------------------------------------- job lifecycle
    def _load_jobs(self) -> None:
        for f in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                job = Job(**json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            if job.state in ("queued", "running"):
                job.state, job.error = "failed", "서버가 다시 시작되어 중단됨 — '다시 실행'을 누르면 캐시된 단계부터 이어서 진행합니다."
                job.save()
            if job.refine_state in ("queued", "running"):
                job.refine_state, job.refine_error = "failed", "서버가 다시 시작되어 보정이 중단됨 — 다시 누르세요."
                job.save()
            self.jobs[job.id] = job

    def submit(self, audio: str, reference: Optional[str], options: dict[str, Any]) -> Job:
        self.config_for(options)  # validate early → 400 on bad options
        job = Job(uuid.uuid4().hex[:12], Path(audio).name, audio, reference, options)
        job.size_bytes = Path(audio).stat().st_size
        try:
            info = probe_audio(Path(audio))
            job.duration_s = info["duration_s"] if info else None
        except (ValueError, OSError, subprocess.SubprocessError):
            job.duration_s = None  # the upload check already accepted it; the length is only used for estimates
        job.save()
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(("convert", job.id))
        return job

    def retry(self, job_id: str) -> Job:
        job = self._job_get(job_id)
        if job is None:
            raise KeyError(job_id)
        with self.lock:
            if job.state in ("queued", "running"):
                return job
            job.state, job.error, job.exit_code, job.stage, job.finished = "queued", None, None, None, None
            job.cancel_requested = False
            job.save()
            self.queue.put(("convert", job.id))
        return job

    def request_refine(self, job_id: str, arrange: bool) -> Job:
        job = self._job_get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state != "done":
            raise self.rf.RefineError("변환이 끝난 곡만 보정할 수 있습니다")
        with self.lock:
            if job.refine_state in ("queued", "running"):
                return job
            job.refine_state, job.refine_arrange, job.refine_error, job.refine_summary = "queued", arrange, None, {}
            job.save()
            self.queue.put(("refine", job.id))
        return job

    def _worker(self) -> None:
        while True:
            kind, job_id = self.queue.get()
            job = self._job_get(job_id)
            if job is None:
                continue
            try:
                if kind == "refine":
                    if job.refine_state == "queued":
                        self._refine(job)
                elif job.state == "queued":  # a conversion cancelled while waiting is simply skipped
                    self._run(job)
            except Exception:  # e.g. disk full while saving job.json: log it, keep serving the queue
                event_log("worker_error", job=job_id, kind=kind, traceback=traceback.format_exc())

    def _refine(self, job: Job) -> None:
        pl, rf, rd = self.pl, self.rf, self.rd
        job.refine_state = "running"
        job.save()
        for name in REFINE_OUTPUTS:  # never mix results of two refine runs
            (job.out / name).unlink(missing_ok=True)
        try:
            if job.intermediates_cleared:
                raise rf.RefineError("중간 파일을 정리해서 보정할 수 없습니다 — '다시 실행'으로 복구한 뒤 보정하세요")
            dirs = {s["name"]: Path(s["stage_dir"]) for s in job.stages}
            missing = [n for n in ("normalize", "separate", "postprocess") if n not in dirs]
            if missing:
                raise rf.RefineError(f"중간 결과({', '.join(missing)})가 없어 보정할 수 없습니다 — 변환을 다시 실행하세요")
            try:
                _, norm = pl.read_manifest(dirs["normalize"], stage="normalize")
                _, sep = pl.read_manifest(dirs["separate"], stage="separate")
                _, post = pl.read_manifest(dirs["postprocess"], stage="postprocess")
            except pl.CacheError as exc:
                raise rf.RefineError(f"캐시가 지워져 보정할 수 없습니다 — 캐시 무시 옵션으로 변환을 다시 실행하세요 ({exc})") from exc
            res = rf.refine(midi=job.out / "piano.mid", events=post.events_path, piano_source=sep.piano_stem_path,
                            mix=norm.audio_path, out_dir=job.out, params=rf.RefineParams(arrange=job.refine_arrange),
                            reference=Path(job.reference) if job.reference else None)
            summary: dict[str, Any] = {"resolved": res.resolved, "remaining": list(res.remaining),
                                       "undeterminable": res.undeterminable}
            cfg = self.config_for(job.options)
            for midi in (res.fixed_midi, res.arranged_midi):
                if midi is None:
                    continue
                try:
                    rd.render_midi(midi, midi.with_suffix(".wav"), pl.render_params(cfg.render, cfg.postprocess.pedal))
                    make_preview(midi.with_suffix(".wav"))
                except Exception as exc:  # MIDI stays usable even when rendering fails
                    summary["remaining"].append(f"{midi.name} 렌더 실패: {type(exc).__name__}: {exc}")
            job.refine_summary, job.refine_state = summary, "done"
        except Exception as exc:  # surface every failure in the UI
            job.refine_state = "failed"
            job.refine_error = f"{type(exc).__name__}: {exc}"
            event_log("refine_failed", job=job.id, traceback=traceback.format_exc())
        job.save()

    def _write_keywords(self, job: Job) -> None:
        """keywords_report.md/json: which of the requested music keywords the converted MIDI really shows.
        An analysis problem never fails the conversion."""
        try:
            notes, pedals = self.tr.read_midi(job.out / "piano.mid")
            bars = self.roll(job).get("bars") or None
            rep = analyze_keywords(notes, pedals, bars=bars, musicxml=job.out / "piano.musicxml")
            (job.out / "keywords_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            (job.out / "keywords_report.md").write_text(keywords_markdown(rep, KEYWORD_LIST), encoding="utf-8")
        except Exception:
            event_log("keywords_failed", job=job.id, traceback=traceback.format_exc())

    def _write_theory(self, job: Job) -> None:
        """piano_theory.mid: piano.mid minus probable ghost notes plus pattern-confirmed missing notes
        (see theory_refine). piano.mid itself is untouched. With a reference MIDI both versions are scored
        so the user sees whether it really helped for this song."""
        try:
            notes, _ = self.tr.read_midi(job.out / "piano.mid")
            bars = self.roll(job).get("bars") or None
            new, info = theory_refine([TNote(n.onset_s, n.offset_s, n.pitch, n.velocity) for n in notes], bars=bars)
            if "skipped" in info:
                info["written"] = False
            else:
                write_theory_midi(job.out / "piano.mid", job.out / "piano_theory.mid", new)
                info["written"] = True
                try:
                    cfg = self.config_for(job.options)
                    self.rd.render_midi(job.out / "piano_theory.mid", job.out / "piano_theory.wav",
                                        self.pl.render_params(cfg.render, cfg.postprocess.pedal))
                    make_preview(job.out / "piano_theory.wav")
                except Exception as exc:  # the MIDI is the product; a failed render only costs the WAV
                    info["render_error"] = f"{type(exc).__name__}: {exc}"
            if job.reference:
                ref, _ = self.tr.read_midi(Path(job.reference))
                refn = [TNote(n.onset_s, n.offset_s, n.pitch, n.velocity) for n in ref]
                info["f1_piano_mid"] = note_f1([TNote(n.onset_s, n.offset_s, n.pitch, n.velocity) for n in notes], refn)
                info["f1_piano_theory_mid"] = note_f1(new, refn)
                info["f1_note"] = "(정밀도, 재현율, F1) — 음높이 일치 + 시작 시각 ±50ms"
            (job.out / "theory_refine_report.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            event_log("theory_refine_failed", job=job.id, traceback=traceback.format_exc())

    def pl_device(self, job: Job) -> str:
        return self.tr.resolve_device(job.options.get("device", "auto"))

    def perf_key(self, job: Job) -> str:
        return f"{self.pl_device(job)}:{job.options.get('quality', 'max')}"

    def _progress_backend(self, job: Job, cfg: Any, device: str) -> Any:
        tc = cfg.transcription
        total = None
        if job.duration_s:
            sr = 16000
            chunks = len(self.tr.plan_chunks(max(1, int(job.duration_s * sr)), sr, tc.chunk_seconds, tc.overlap_seconds))
            total = chunks * (1 + len(tc.extra_sources)) * (2 if tc.chunk_offset_tta else 1)
            # Transkun is an additional full-source analysis when enabled. It does not use chunk TTA.
            if tc.second_model == "transkun":
                total += chunks
        return ProgressBackend(self._backend(cfg, device), job, total, self.pl.JobCancelled)

    def _run(self, job: Job) -> None:
        pl = self.pl
        job.state, job.error, job.exit_code, job.stage = "running", None, None, None
        job.cancel_requested = False
        job.started, job.finished, job.progress, job.progress_label = time.time(), None, None, None
        job.save()
        perf_key = self.perf_key(job)

        def wrap(name: str, fn: Any) -> Any:
            def inner(upstream: Any, out_dir: Path, cfg: Any, ctx: Any) -> Any:
                if job.cancel_requested:
                    raise pl.JobCancelled("사용자가 취소했습니다")
                job.stage, job.stage_started, job.progress, job.progress_label = name, time.time(), None, None
                job.save()
                return fn(upstream, out_dir, cfg, replace(ctx, should_cancel=lambda: job.cancel_requested))
            return inner

        fns = {n: wrap(n, f) for n, f in pl.DEFAULT_STAGE_FNS.items()}
        fns["transcribe"] = wrap("transcribe", pl.make_stage_transcribe(lambda cfg, dev: self._progress_backend(job, cfg, dev)))
        try:
            free = shutil.disk_usage(BASE).free
            if free < MIN_FREE_BYTES:
                raise OSError(f"디스크 여유 공간이 부족합니다({free / 2**30:.1f} GB 남음, 최소 {MIN_FREE_BYTES / 2**30:.1f} GB 필요). "
                              "오래된 곡을 삭제하거나 '저장공간 정리'를 하세요.")
            job.out.mkdir(parents=True, exist_ok=True)
            if not (job.out / "original_preview.mp3").is_file():  # for the A/B comparison player
                try:
                    make_mp3(Path(job.audio), job.out / "original_preview.mp3", "128k")
                except Exception as exc:  # the comparison player is a convenience, never a reason to fail
                    event_log("original_preview_failed", job=job.id, error=str(exc))
            best_soundfont(job.options.get("piano", "auto"), wait_s=float(os.environ.get("PIANOFORGE_SOUNDFONT_WAIT_S", "60")))  # let a running piano download finish
            cfg = self.config_for(job.options)
            summary = pl.Pipeline(cfg, stage_fns=fns).run(
                Path(job.audio), job.out, reference=Path(job.reference) if job.reference else None,
                use_cache=not job.options.get("no_cache", False))
            make_preview(job.out / "piano.wav")
            self._write_keywords(job)
            self._write_theory(job)
            job.state, job.stage, job.intermediates_cleared = "done", None, False
            job.summary = list(summary.evaluation.get("summary", ()))
            job.duration_s = job.duration_s or float(summary.audio_duration_s)
            record_perf(self.perf, perf_key, float(summary.audio_duration_s), [asdict(r) for r in summary.stages])
            try:
                save_perf(PERF_PATH, self.perf)
            except OSError:
                pass
        except pl.JobCancelled:
            job.state, job.stage, job.exit_code = "cancelled", None, 130
            job.error = "사용자가 취소했습니다. '다시 실행'을 누르면 끝난 단계부터 이어서 합니다."
        except Exception as exc:  # report every failure to the UI, never kill the worker
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.exit_code = int(getattr(exc, "exit_code", 1))
            event_log("job_failed", job=job.id, traceback=traceback.format_exc())
        finally:
            job.cancel_requested, job.progress, job.progress_label = False, None, None
        rs = job.out / "run_summary.json"
        if rs.is_file():
            try:
                job.stages = json.loads(rs.read_text(encoding="utf-8")).get("stages", [])
            except ValueError:
                pass
        job.finished = time.time()
        job.save()

    # ---------------------------------------------------------------- user actions
    def cancel(self, job_id: str) -> Job:
        job = self._job_get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state == "queued":
            job.state, job.error, job.exit_code = "cancelled", "대기 중에 취소했습니다.", 130
            job.save()
        elif job.state == "running":
            job.cancel_requested = True  # takes effect at the next safe point (stage start / next model batch)
            job.save()
        else:
            raise ValueError("진행 중이거나 대기 중인 변환만 취소할 수 있습니다")
        return job

    def _referenced_dirs(self, exclude: Optional[str] = None) -> set[Path]:
        refs: set[Path] = set()
        for j in self._jobs_snapshot():
            if j.id != exclude:
                refs.update(Path(s["stage_dir"]).resolve() for s in j.stages)
        return refs

    def _remove_stage_dirs(self, job: Job, stages: Optional[tuple[str, ...]] = None) -> int:
        refs = self._referenced_dirs(exclude=job.id)
        freed = 0
        for s in job.stages:
            d = Path(s["stage_dir"])
            if (stages is None or s["name"] in stages) and d.exists() and d.resolve() not in refs:
                freed += dir_size(d)
                shutil.rmtree(d, ignore_errors=True)
        return freed

    def _busy(self, job: Job) -> bool:
        return job.state in ("queued", "running") or job.refine_state in ("queued", "running")

    def delete_job(self, job_id: str) -> int:
        """Remove a finished/failed/cancelled job with its results, upload and unshared cache entries.
        Returns the bytes freed."""
        job = self._job_get(job_id)
        if job is None:
            raise KeyError(job_id)
        if self._busy(job):
            raise ValueError("진행 중인 곡은 먼저 취소하세요")
        freed = self._remove_stage_dirs(job) + dir_size(job.dir)
        others = {Path(p).resolve().parent for j in self._jobs_snapshot() if j.id != job_id
                  for p in (j.audio, j.reference) if p}
        for p in (job.audio, job.reference):
            if p and Path(p).resolve().parent not in others and Path(p).parent.parent.resolve() == UPLOADS.resolve():
                freed += dir_size(Path(p).parent)
                shutil.rmtree(Path(p).parent, ignore_errors=True)
        shutil.rmtree(job.dir, ignore_errors=True)
        with self.lock:
            self.jobs.pop(job_id, None)
        return freed

    def cleanup_intermediates(self, job_id: str) -> int:
        """Drop the big intermediate files (normalized audio and separated stems, ~400 MB for a 3-minute
        song). Results stay; refinement needs them and becomes unavailable until the song is converted again."""
        job = self._job_get(job_id)
        if job is None:
            raise KeyError(job_id)
        if self._busy(job):
            raise ValueError("진행 중인 곡은 정리할 수 없습니다")
        freed = self._remove_stage_dirs(job, ("normalize", "separate"))
        job.intermediates_cleared = True
        job.save()
        return freed

    def purge_orphans(self) -> int:
        """Delete cache entries no job refers to any more (left behind by re-runs with other settings and
        by aborted runs). Model weights are never touched."""
        refs = self._referenced_dirs()
        freed = 0
        for stage in self.pl.STAGES:
            root = CACHE / stage
            if not root.is_dir():
                continue
            for sub in [p for p in root.iterdir() if p.is_dir()]:
                if sub.name == ".tmp":
                    entries = [p for p in sub.iterdir() if time.time() - p.stat().st_mtime > 3600]
                else:
                    entries = [p for p in sub.iterdir() if p.is_dir() and p.resolve() not in refs]
                for p in entries:
                    freed += dir_size(p)
                    shutil.rmtree(p, ignore_errors=True)
        return freed

    def storage_stats(self) -> dict[str, Any]:
        refs = self._referenced_dirs()
        stage_bytes = orphan = 0
        for stage in self.pl.STAGES:
            root = CACHE / stage
            if root.is_dir():
                for sub in root.iterdir():
                    for p in (sub.iterdir() if sub.is_dir() else []):
                        size = dir_size(p) if p.is_dir() else 0
                        stage_bytes += size
                        if sub.name == ".tmp" or p.resolve() not in refs:
                            orphan += size
        return {"free_bytes": shutil.disk_usage(BASE).free, "jobs_bytes": dir_size(JOBS_DIR), "uploads_bytes": dir_size(UPLOADS),
                "cache_bytes": stage_bytes, "orphan_bytes": orphan, "models_bytes": dir_size(CACHE / "models"),
                "soundfonts_bytes": dir_size(SF_DIR)}

    def queue_position(self, job: Job) -> Optional[int]:
        """How many conversions are ahead of a queued one (the running one included)."""
        if job.state != "queued":
            return None
        with self.queue.mutex:
            ids = [jid for kind, jid in self.queue.queue if kind == "convert"]
        snapshot = {j.id: j for j in self._jobs_snapshot()}
        ahead = [snapshot[j] for j in (ids[: ids.index(job.id)] if job.id in ids else []) if snapshot.get(j) and snapshot[j].state == "queued"]
        return len(ahead) + (1 if any(j.state == "running" for j in snapshot.values()) else 0)

    def public(self, job: Job) -> dict[str, Any]:
        d = job.public()
        now = time.time()
        d["queue_position"] = self.queue_position(job)
        d["elapsed_s"] = (now - job.started) if job.state == "running" and job.started else (
            (job.finished - job.started) if job.finished and job.started else None)
        d["stage_elapsed_s"] = (now - job.stage_started) if job.state == "running" and job.stage_started else None
        d["eta_s"] = (eta_seconds(self.perf, self.perf_key(job), job.duration_s, job.stage, d["stage_elapsed_s"] or 0.0, job.progress)
                      if job.state == "running" and job.stage else None)
        return d

    def roll(self, job: Job) -> dict[str, Any]:
        """Piano roll data: notes plus bar lines and chord labels from the music analysis, if present."""
        notes = self.notes(job)
        bars: list[float] = []
        chords: list[list[Any]] = []
        key = None
        rep = job.out / "report.json"
        if rep.is_file():
            try:
                an = (json.loads(rep.read_text(encoding="utf-8")).get("context", {}).get("postprocess", {}) or {}).get("analysis") or {}
                key = an.get("key")
                chords = [[round(float(c["start"]), 3), c["label"]] for c in an.get("chords", []) if c.get("label")]
                if an.get("unit") == "bar":
                    bars = [round(float(c["start"]), 3) for c in an.get("chords", [])]
            except (ValueError, KeyError, TypeError):
                pass
        return {"notes": notes, "bars": bars, "chords": chords, "key": key, "end": max((n[1] for n in notes), default=0.0)}

    def notes(self, job: Job) -> list[list[float]]:
        midi = job.out / "piano.mid"
        if not midi.is_file():
            return []
        notes, _ = self.tr.read_midi(midi)
        return [[round(n.onset_s, 4), round(n.offset_s, 4), n.pitch, n.velocity] for n in notes[:30000]]

    def health(self) -> dict[str, Any]:
        v = self.pl.library_versions()
        sf_path, sf_label = best_soundfont()
        gpu = None
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gpu = {"name": torch.cuda.get_device_name(0), "total_gb": round(total / 2**30, 1), "free_gb": round(free / 2**30, 1)}
        except Exception:  # health must never fail because of a driver quirk
            gpu = None
        tunnel = getattr(builtins, "_pianoforge_tunnel", None)
        return {"version": VERSION, "device": self.device, "gpu": gpu, "soundfont": sf_label, "soundfont_ok": Path(sf_path).is_file(),
                "transkun": shutil.which("transkun") is not None,
                "fluidsynth_lib": ctypes.util.find_library("fluidsynth") is not None,
                "disk_free_gb": round(shutil.disk_usage(BASE).free / 2**30, 1),
                "running": sum(1 for j in self._jobs_snapshot() if j.state == "running"),
                "queued": sum(1 for j in self._jobs_snapshot() if j.state == "queued"),
                "uptime_s": round(time.time() - STARTED_AT),
                "public_url": getattr(builtins, "_pianoforge_url", None),
                "tunnel_alive": bool(tunnel is not None and tunnel.poll() is None),
                "versions": {k: v.get(k) for k in ("python", "torch", "torchaudio", "demucs",
                                                    "piano_transcription_inference", "pretty_midi", "mir_eval")}}


def configure_file_logging() -> None:
    import logging

    import structlog

    fh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    structlog.configure(
        processors=[structlog.contextvars.merge_contextvars, structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso", utc=True), structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(ensure_ascii=False)],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=fh), cache_logger_on_first_use=False)


def project_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, text in SOURCES.items():
            zf.writestr(f"pianoforge/{rel}", text)
    return buf.getvalue()


def safe_name(name: str) -> str:
    base = Path(unquote(name)).name.strip() or "upload"
    return re.sub(r"[^\w.\- ()\[\]가-힣]", "_", base)[:120]


# =========================================================================== HTTP


LIMITER = AuthLimiter()


def tail_lines(path: Path, n: int = 80, max_bytes: int = 1 << 18, contains: Optional[tuple[str, ...]] = None) -> list[str]:
    """Last lines of a (possibly huge) log file without reading all of it; optionally only lines mentioning one of ``contains``."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = data.splitlines()
    if size > max_bytes:
        lines = lines[1:]  # the first line is usually cut in the middle
    if contains:
        lines = [ln for ln in lines if any(c in ln for c in contains)]
    return lines[-n:]


def cookie_header(token: str) -> str:
    return f"pf_k={token}; Path=/; Max-Age=2592000; SameSite=Lax; Secure; HttpOnly"


def rotate_token() -> str:
    """New access key; every old link stops working. Not possible when PIANOFORGE_TOKEN pins the key."""
    global TOKEN
    if os.environ.get("PIANOFORGE_TOKEN"):
        raise ValueError("PIANOFORGE_TOKEN 환경 변수로 접근 키가 고정되어 있어 바꿀 수 없습니다")
    new = secrets.token_urlsafe(18)
    _write_secret(BASE / "access_token", new)
    TOKEN = new
    return new


class Handler(BaseHTTPRequestHandler):
    app: App
    server_version = f"PianoForge/{VERSION}"
    timeout = 120  # per socket read: a stalled client cannot hold a server thread forever

    def log_message(self, fmt: str, *args: Any) -> None:  # keep notebook output clean
        return

    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")  # the access key is in the address: never pass it on
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _file(self, path: Path, ctype: str, download: Optional[str]) -> None:
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng)
        if m and size:
            s, e = m.groups()
            if s:
                start, end = int(s), min(int(e) if e else size - 1, size - 1)
            elif e:
                start = max(0, size - int(e))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            ascii_name = download.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(download)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                block = fh.read(min(left, 1 << 16))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)

    def _authorized(self) -> bool:
        q = parse_qs(urlparse(self.path).query)
        if token_matches(q.get("k", [""])[0], TOKEN):
            return True
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == "pf_k" and token_matches(value, TOKEN):
                return True
        return False

    def _client_ip(self) -> str:
        # CF-Connecting-IP is only trustworthy when the request really came through our own tunnel (loopback peer)
        peer = self.client_address[0]
        if peer in ("127.0.0.1", "::1"):
            return self.headers.get("CF-Connecting-IP") or peer
        return peer

    def _content_length(self) -> int:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("Content-Length가 올바르지 않습니다") from None
        if n < 0:
            raise ValueError("Content-Length가 올바르지 않습니다")
        return n

    def _read_json(self, length: int) -> dict[str, Any]:
        if length > MAX_JSON_BYTES:
            raise ValueError("요청 본문이 너무 큽니다")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ValueError("JSON 형식이 올바르지 않습니다") from None
        if not isinstance(body, dict):
            raise ValueError("JSON 객체가 필요합니다")
        return body

    def _gate(self, html_page: bool) -> bool:
        """Access key check with a brake on repeated wrong keys."""
        ip = self._client_ip()
        if not LIMITER.allow(ip):
            self._send(429, "잘못된 접근 시도가 너무 많습니다. 1분 뒤에 다시 시도하세요.".encode("utf-8"),
                       "text/plain; charset=utf-8", {"Retry-After": "60"})
            return False
        if self._authorized():
            return True
        LIMITER.fail(ip)
        if html_page:
            self._deny()
        else:
            self._err(401, "접근 키가 없습니다")
        return False

    def _deny(self) -> None:
        body = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                "<body style='font-family:system-ui;padding:24px;line-height:1.6'><h2>접근 키가 필요합니다</h2>"
                "<p>Colab 셀 출력에 나온 <b>전체 주소</b>(끝에 <code>?k=…</code> 포함)로 다시 여세요.</p></body>")
        self._send(401, body.encode("utf-8"), "text/html; charset=utf-8")

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._gate(True):
            return
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._err(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        if not self._gate(False):
            return
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception as exc:
            code = 400 if getattr(exc, "exit_code", None) == 2 or type(exc) is ValueError else 500
            self._err(code, f"{type(exc).__name__}: {exc}")

    def _get(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        app = self.app
        if not parts:
            page = INDEX_HTML.replace("__PF_TOKEN__", html.escape(TOKEN, quote=True))
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8",
                       {"Set-Cookie": cookie_header(TOKEN)})
        elif parts == ["api", "health"]:
            self._json(app.health())
        elif parts == ["api", "jobs"]:
            jobs = sorted(app._jobs_snapshot(), key=lambda j: j.created, reverse=True)
            self._json([app.public(j) for j in jobs])
        elif parts == ["api", "storage"]:
            self._json(app.storage_stats())
        elif parts == ["api", "project.zip"]:
            self._send(200, project_zip(), "application/zip",
                       {"Content-Disposition": "attachment; filename=\"pianoforge.zip\""})
        elif parts == ["api", "log"]:
            jid = parse_qs(url.query).get("job", [""])[0]
            job = app._job_get(jid)
            only = (job.name, job.id) if job is not None else None
            self._send(200, "\n".join(tail_lines(LOG_PATH, 80, contains=only)).encode("utf-8"), "text/plain; charset=utf-8")
        elif len(parts) >= 3 and parts[:2] == ["api", "jobs"] and app._job_get(parts[2]) is not None:
            job = app._job_get(parts[2])
            if job is None:
                raise KeyError(parts[2])
            if len(parts) == 3:
                self._json(app.public(job))
            elif parts[3:] == ["notes"]:
                self._json(app.notes(job))
            elif parts[3:] == ["roll"]:
                self._json(app.roll(job))
            elif parts[3:] == ["all.zip"]:
                self._send_job_zip(job)
            elif len(parts) == 5 and parts[3] == "files" and parts[4] in OUTPUT_FILES and (job.out / parts[4]).is_file():
                stem = Path(job.name).stem
                dl = None if parse_qs(url.query).get("inline") else f"{stem}_{parts[4]}"
                self._file(job.out / parts[4], OUTPUT_FILES[parts[4]], dl)
            else:
                self._err(404, "not found")
        else:
            self._err(404, "not found")

    def _drain(self, length: int) -> None:
        left = length
        try:
            while left > 0:
                block = self.rfile.read(min(left, 1 << 16))
                if not block:
                    break
                left -= len(block)
        except OSError:
            pass

    def _send_job_zip(self, job: Job) -> None:
        """Every result file of one song in a single zip (the MP3 previews are left out: they are copies)."""
        names = [n for n in OUTPUT_FILES if (job.out / n).is_file() and not n.endswith(".mp3")]
        if not names:
            self._err(404, "내려받을 결과가 없습니다")
            return
        stem = Path(job.name).stem
        tmp_dir = BASE / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".zip", dir=tmp_dir)
        try:
            with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                for n in names:
                    zf.write(job.out / n, f"{stem}/{n}")
            self._file(Path(tmp), "application/zip", f"{stem}_pianoforge.zip")
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _post(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        q = parse_qs(url.query)
        length = self._content_length()
        if length > MAX_UPLOAD_BYTES:
            self._err(413, f"파일이 너무 큽니다 (최대 {MAX_UPLOAD_BYTES // 2**20} MB)")
            return
        if parts == ["api", "upload"]:
            kind = q.get("kind", ["audio"])[0]
            name = safe_name(q.get("name", ["upload"])[0])
            allowed = AUDIO_EXT if kind == "audio" else MIDI_EXT
            if not name.lower().endswith(allowed):
                self._err(400, f"허용되는 확장자: {', '.join(allowed)}")
                return
            if length == 0:
                self._err(400, "빈 파일입니다")
                return
            if shutil.disk_usage(BASE).free < length + MIN_FREE_BYTES:
                self._drain(length)  # read the body first: a browser that is still sending would only see a broken connection
                self._err(507, "디스크 여유 공간이 부족합니다. 오래된 곡을 삭제하거나 '저장공간 정리'를 하세요")
                return
            uid = uuid.uuid4().hex[:12]
            dest = UPLOADS / uid / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                left = length
                while left > 0:
                    block = self.rfile.read(min(left, 1 << 16))
                    if not block:
                        break
                    fh.write(block)
                    left -= len(block)
            if left:
                shutil.rmtree(dest.parent, ignore_errors=True)
                self._err(400, "업로드가 중간에 끊겼습니다")
                return
            meta: dict[str, Any] = {"id": uid, "name": name, "size": length}
            try:  # reject a corrupt / too long / unreadable file now, not after minutes in the queue
                if kind == "audio":
                    info = probe_audio(dest)
                    if info:
                        if info["duration_s"] > MAX_AUDIO_SECONDS:
                            raise ValueError(f"곡 길이가 {format_duration(info['duration_s'])}입니다. 최대 {MAX_AUDIO_SECONDS // 60}분까지 지원합니다")
                        meta.update(info)
                else:
                    ref_notes, _ = self.app.tr.read_midi(dest)
                    if not ref_notes:
                        raise ValueError("참조 MIDI에 음이 없습니다")
                    meta["notes"] = len(ref_notes)
            except (ValueError, subprocess.SubprocessError) as exc:
                shutil.rmtree(dest.parent, ignore_errors=True)
                msg = ("참조 MIDI를 읽을 수 없습니다" if isinstance(exc, self.app.tr.InvalidMidiError)
                       else "파일 검사 시간이 초과되었습니다" if isinstance(exc, subprocess.SubprocessError) else str(exc))
                self._err(400, msg)
                return
            self._json(meta)
        elif parts == ["api", "jobs"]:
            body = self._read_json(length)

            def resolve(uid: Optional[str]) -> Optional[str]:
                if not uid:
                    return None
                files = list((UPLOADS / re.sub(r"\W", "", uid)).glob("*"))
                if not files:
                    raise FileNotFoundError(f"업로드를 찾을 수 없음: {uid}")
                return str(files[0])

            audio = resolve(body.get("audio"))
            if audio is None:
                self._err(400, "오디오 파일이 필요합니다")
                return
            job = self.app.submit(audio, resolve(body.get("reference")), dict(body.get("options") or {}))
            self._json(self.app.public(job), 201)
        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "retry" and self.app._job_get(parts[2]) is not None:
            self._json(self.app.public(self.app.retry(parts[2])))
        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in ("cancel", "delete", "cleanup") and self.app._job_get(parts[2]) is not None:
            app = self.app
            try:
                if parts[3] == "cancel":
                    self._json(app.public(app.cancel(parts[2])))
                elif parts[3] == "delete":
                    self._json({"deleted": parts[2], "freed_bytes": app.delete_job(parts[2])})
                else:
                    freed = app.cleanup_intermediates(parts[2])
                    self._json({"freed_bytes": freed, "job": app.public(app._job_get(parts[2]))})
            except ValueError as exc:
                self._err(409, str(exc))
        elif parts == ["api", "storage", "purge"]:
            self._json({"freed_bytes": self.app.purge_orphans()})
        elif parts == ["api", "rotate-token"]:
            try:
                new = rotate_token()
            except ValueError as exc:
                self._err(409, str(exc))
                return
            self._send(200, json.dumps({"token": new}).encode("utf-8"), "application/json; charset=utf-8",
                       {"Set-Cookie": cookie_header(new)})
        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "refine" and self.app._job_get(parts[2]) is not None:
            body = self._read_json(length)
            try:
                job = self.app.request_refine(parts[2], bool(body.get("arrange", False)))
            except self.app.rf.RefineError as exc:
                self._err(409, str(exc))
                return
            self._json(self.app.public(job))
        else:
            self._err(404, "not found")


TOKEN = ""  # set in main()


def start_server(app: App) -> ThreadingHTTPServer:
    old = getattr(builtins, "_pianoforge_server", None)
    if old is not None:  # re-running the cell: replace the previous server
        old.shutdown()
        old.server_close()
    Handler.app = app
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    srv.request_queue_size = 64
    builtins._pianoforge_server = srv  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, name="pianoforge-http", daemon=True).start()
    return srv


def self_check() -> bool:
    """Fetch the page from inside the VM so a blank browser tab can be told apart from a dead server."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/?k={TOKEN}", timeout=5) as resp:
            ok = resp.status == 200 and resp.headers.get_content_type() == "text/html"
    except OSError as exc:
        print(f"! 서버 자체 점검 실패: {exc}")
        return False
    print("서버 자체 점검: 정상 (VM 안에서 페이지가 응답합니다)" if ok else "! 서버 자체 점검: 응답 형식 이상")
    return ok


def _cloudflared_binary() -> Optional[Path]:
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if arch is None or sys.platform != "linux":
        print(f"! cloudflared를 자동으로 받을 수 없는 플랫폼입니다 ({sys.platform}/{platform.machine()}).")
        return None
    dest = BASE / "bin" / "cloudflared"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        print("cloudflared 내려받는 중…", flush=True)
        try:
            with urllib.request.urlopen(CLOUDFLARED_URL.format(arch=arch), timeout=120) as resp, open(part, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except OSError as exc:
            part.unlink(missing_ok=True)
            print(f"! cloudflared 다운로드 실패: {exc}")
            return None
        with open(part, "rb") as fh:
            magic = fh.read(4)
        if magic != b"\x7fELF" or part.stat().st_size < 5_000_000:  # truncated download or an HTML error page
            part.unlink(missing_ok=True)
            print("! 내려받은 cloudflared 파일이 올바르지 않습니다(ELF 아님/너무 작음).")
            return None
        part.replace(dest)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def start_tunnel(timeout_s: float = 45.0) -> Optional[str]:
    """Start a Cloudflare quick tunnel to the local server and return its public base URL."""
    old = getattr(builtins, "_pianoforge_tunnel", None)
    if old is not None and old.poll() is None:  # re-running the cell: replace the previous tunnel
        old.terminate()
        try:
            old.wait(timeout=5)
        except subprocess.TimeoutExpired:
            old.kill()
    binary = _cloudflared_binary()
    if binary is None:
        return None
    prev_log = getattr(builtins, "_pianoforge_tunnel_log", None)
    if prev_log is not None:
        prev_log.close()
    log_file = open(BASE / "tunnel.log", "w", encoding="utf-8")
    builtins._pianoforge_tunnel_log = log_file  # type: ignore[attr-defined]
    proc = subprocess.Popen([str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    builtins._pianoforge_tunnel = proc  # type: ignore[attr-defined]
    found: dict[str, str] = {}
    ready = threading.Event()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                log_file.write(line)
                log_file.flush()
            except ValueError:  # closed by a restarted tunnel
                pass
            m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if m and "url" not in found:
                found["url"] = m.group(0)
                ready.set()
        ready.set()

    threading.Thread(target=pump, name="pianoforge-tunnel-log", daemon=True).start()
    ready.wait(timeout_s)
    if "url" not in found:
        print(f"! 공개 터널 주소를 받지 못했습니다. 로그: {BASE / 'tunnel.log'}")
        return None
    url = found["url"]
    for _ in range(20):  # the hostname needs a few seconds before it routes
        try:
            with urllib.request.urlopen(f"{url}/?k={TOKEN}", timeout=10) as resp:
                if resp.status == 200:
                    break
        except OSError:
            time.sleep(2)
    return url


def _show_qr(url: str) -> None:
    """QR code of the full address so a phone can open it by scanning the notebook output (needs segno)."""
    try:
        import segno
    except ImportError:
        return
    try:
        qr = segno.make(url, error="m")
        if IN_KERNEL:
            from IPython.display import HTML, display

            display(HTML(qr.svg_inline(scale=4, border=2, dark="#18202b", light="#ffffff")))
        else:
            qr.terminal(compact=True)
    except Exception as exc:  # a QR code is a convenience only
        print(f"(QR 코드를 만들지 못했습니다: {exc})")


def start_tunnel_watchdog() -> None:
    """Restart the Cloudflare tunnel when it dies. The public address changes on every restart; the new one
    is printed to the notebook and kept in /api/health (public_url)."""
    prev = getattr(builtins, "_pianoforge_watchdog_stop", None)
    if prev is not None:
        prev.set()
    stop = threading.Event()
    builtins._pianoforge_watchdog_stop = stop  # type: ignore[attr-defined]

    def run() -> None:
        restarts, delay = 0, 5.0
        while not stop.wait(10.0):
            proc = getattr(builtins, "_pianoforge_tunnel", None)
            if proc is None or proc.poll() is None:
                continue
            if restarts >= 8:
                print("! 공개 터널이 계속 끊어져 자동 재연결을 멈춥니다. 셀을 다시 실행하세요.")
                return
            restarts += 1
            print(f"! 공개 터널이 끊겨 다시 연결합니다({restarts}/8) — 주소가 바뀝니다.")
            url = start_tunnel()
            if url:
                builtins._pianoforge_url = url  # type: ignore[attr-defined]
                _show_link(f"{url}/?k={TOKEN}", "▶ 새 PianoForge 주소")
                _show_qr(f"{url}/?k={TOKEN}")
            else:
                stop.wait(delay)
                delay = min(delay * 2, 120.0)

    threading.Thread(target=run, name="pianoforge-tunnel-watchdog", daemon=True).start()


def _show_link(url: str, label: str) -> None:
    if IN_KERNEL:
        try:
            from IPython.display import HTML, display

            display(HTML(f'<p style="font-size:18px"><a href="{html.escape(url)}" target="_blank" rel="noopener">'
                         f'{html.escape(label)}</a></p>'))
        except ImportError:
            pass
    print(f"{label}: {url}")


def announce() -> None:
    print(f"\nPianoForge 서버 실행 중 (포트 {PORT}, 작업 폴더 {BASE})")
    self_check()
    public = start_tunnel() if TUNNEL == "cloudflare" else None
    if public:
        builtins._pianoforge_url = public  # type: ignore[attr-defined]
        _show_link(f"{public}/?k={TOKEN}", "▶ PianoForge 열기 (휴대폰·다른 브라우저 모두 가능)")
        _show_qr(f"{public}/?k={TOKEN}")
        start_tunnel_watchdog()
        print("  이 주소는 공개 인터넷 주소입니다. 접근 키(k=…)가 포함된 전체 주소를 남과 공유하지 마세요.")
        print("  Colab 런타임이 끝나면 주소도 사라집니다. 셀을 다시 실행하면 새 주소가 나옵니다.")
    if IN_COLAB and IN_KERNEL:
        try:
            from google.colab import output

            if not public:
                print("공개 터널 없이 Colab 안에서 엽니다. 아래 셀 출력 화면을 쓰세요.")
                output.serve_kernel_port_as_iframe(PORT, path=f"/?k={TOKEN}", height=1100)
            output.serve_kernel_port_as_window(PORT, path=f"/?k={TOKEN}",
                                               anchor_text="(대체 경로) Colab 프록시로 열기 — Colab이 열린 같은 브라우저 전용")
        except Exception as exc:  # proxy helpers differ between Colab versions
            print(f"Colab 프록시 링크 생성 실패({exc})")
    elif IN_COLAB and not public:
        print("! '!python'으로 실행하면 Colab 링크를 만들 수 없습니다. 셀에서 %run pianoforge_colab.py 로 실행하세요.")
    elif not IN_COLAB:
        _show_link(f"http://localhost:{PORT}/?k={TOKEN}", "브라우저에서 열기 (Codespaces는 포트를 자동 전달)")


def main() -> Optional[App]:
    global TOKEN
    TOKEN = _load_token()
    for d in (PROJECT, JOBS_DIR, UPLOADS, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(BASE / "tmp", ignore_errors=True)  # leftovers of interrupted zip downloads
    if str(BASE).startswith("/content/drive"):
        print(f"결과는 Google Drive에 저장됩니다: {BASE}  (중간 파일은 런타임 디스크 {CACHE})")
    if os.environ.get("PIANOFORGE_SKIP_INSTALL") != "1":
        install_system_packages()
        install_python_packages()
    write_project()
    start_soundfont_download()
    configure_file_logging()
    app = App()
    start_server(app)
    announce()
    h = app.health()
    print(f"장치: {h['device']} · SoundFont: {'있음' if h['soundfont_ok'] else '없음 (렌더 단계 실패)'} · "
          f"FluidSynth: {'있음' if h['fluidsynth_lib'] else '없음 (렌더 단계 실패)'}")
    print("첫 변환은 Demucs·전사 모델 가중치(수백 MB)를 내려받으므로 오래 걸립니다.")
    if not IN_KERNEL:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("종료")
        return None
    return app


INDEX_HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#2c56d6">
<meta name="referrer" content="no-referrer">
<title>PianoForge</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎹</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
:root{--paper:#f5f6f8;--sheet:#ffffff;--ink:#18202b;--muted:#5d6878;--rule:#d9dee6;--note:#2c56d6;--note-soft:#e5ecfd;--warn:#a2470a;--ok:#16794a;
  --fail:#b3261e;--key-w:#ffffff;--key-b:#18202b;--head:#d4380d;color-scheme:light;
  font-family:"IBM Plex Sans KR",system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12161d;--sheet:#1a2029;--ink:#e7ebf1;--muted:#9aa5b5;--rule:#2d3540;--note:#7b9bff;
  --note-soft:#232d45;--warn:#f0a35e;--ok:#5cc791;--fail:#ff8a80;--key-w:#d9dee6;--key-b:#0c0f14;--head:#ff9c6e;color-scheme:dark;}}
:root[data-theme="dark"]{--paper:#12161d;--sheet:#1a2029;--ink:#e7ebf1;--muted:#9aa5b5;--rule:#2d3540;--note:#7b9bff;
  --note-soft:#232d45;--warn:#f0a35e;--ok:#5cc791;--fail:#ff8a80;--key-w:#d9dee6;--key-b:#0c0f14;--head:#ff9c6e;color-scheme:dark;}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink)}
body{padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0);line-height:1.55;font-size:16px}
main{max-width:880px;margin:0 auto;padding:0 18px 64px}
header{padding:32px 0 14px;display:grid;grid-template-columns:auto 1fr auto;gap:18px;align-items:end}
.keys{display:flex;height:64px;align-items:flex-start}
.keys i{display:block;width:14px;height:64px;background:var(--key-w);border:1px solid var(--rule);border-radius:0 0 3px 3px;margin-right:-1px}
.keys b{display:block;width:9px;height:38px;background:var(--key-b);margin:0 -5px;position:relative;z-index:1;border-radius:0 0 2px 2px}
h1{font-size:2.4rem;line-height:1;margin:0;letter-spacing:-.02em;font-weight:700}
header p{margin:6px 0 0;color:var(--muted)}
section{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:20px;margin-top:18px}
h2{font-size:1.15rem;margin:0 0 14px}
label{display:block;font-weight:500;margin:0 0 6px}
.field{margin-bottom:16px}
.hint{color:var(--muted);font-size:.88rem;margin-top:4px}
select{font:inherit;padding:8px 10px;border:1px solid var(--rule);border-radius:6px;background:var(--sheet);color:var(--ink);width:100%}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.check{display:flex;gap:8px;align-items:center;font-weight:400;margin:6px 0}
button,.btn{font:inherit;font-weight:500;border:1px solid var(--note);background:var(--note);color:#fff;padding:10px 18px;border-radius:8px;cursor:pointer;text-decoration:none;display:inline-block}
button.ghost,.btn.ghost{background:transparent;color:var(--note)}
button.danger{border-color:var(--fail);color:var(--fail);background:transparent}
button.small{padding:5px 12px;font-size:.88rem}
button:disabled{opacity:.5;cursor:default}
@media (pointer:coarse){button,.btn{min-height:44px}button.small{min-height:36px}}
:focus-visible{outline:3px solid var(--note);outline-offset:2px}
.drop{display:block;border:2px dashed var(--rule);border-radius:10px;padding:22px 14px;text-align:center;cursor:pointer;background:var(--paper);font-weight:400}
.drop.over{border-color:var(--note);background:var(--note-soft)}
.drop input{display:none}
.picked{list-style:none;margin:10px 0 0;padding:0}
.picked li{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:6px 0;border-top:1px solid var(--rule);font-size:.92rem;word-break:break-all}
.picked li.bad{color:var(--fail)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}
.chip{background:var(--note-soft);color:var(--note);border-color:transparent;padding:6px 14px;border-radius:999px;font-size:.9rem}
.job{border-top:1px solid var(--rule);padding:16px 0}
.job:first-of-type{border-top:0;padding-top:0}
.job h3{margin:0;font-size:1.05rem;word-break:break-all}
.meta{color:var(--muted);font-size:.85rem}
.state{font-size:.88rem;font-weight:500;white-space:nowrap}
.state.done{color:var(--ok)}.state.failed{color:var(--fail)}.state.running,.state.queued{color:var(--note)}.state.cancelled{color:var(--warn)}
ol.steps{list-style:none;display:grid;grid-template-columns:repeat(6,1fr);gap:4px;padding:0;margin:12px 0 6px}
ol.steps li{font-size:.75rem;text-align:center;color:var(--muted)}
ol.steps li::before{content:"";display:block;height:6px;border-radius:3px;background:var(--rule);margin-bottom:4px}
ol.steps li.done::before{background:var(--note)}
ol.steps li.now::before{background:var(--note);animation:pulse 1.2s ease-in-out infinite}
ol.steps li.now{color:var(--ink);font-weight:500}
@keyframes pulse{50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){ol.steps li.now::before{animation:none}}
.bar{height:8px;border-radius:4px;background:var(--rule);overflow:hidden;margin:6px 0}
.bar i{display:block;height:100%;background:var(--note);transition:width .6s ease}
.err{color:var(--fail);background:color-mix(in srgb,var(--fail) 8%,transparent);padding:10px 12px;border-radius:6px;font-size:.92rem;white-space:pre-wrap;word-break:break-word}
.tip{background:var(--note-soft);padding:8px 12px;border-radius:6px;font-size:.88rem;margin-top:8px}
.summary{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.82rem;background:var(--paper);padding:10px 12px;border-radius:6px;white-space:pre-wrap;overflow-x:auto}
.files{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.files a{font-size:.9rem;padding:6px 12px}
audio{width:100%;margin:8px 0}
.rollwrap{position:relative;height:200px;margin-top:6px}
.rollwrap canvas{position:absolute;inset:0;width:100%;height:200px;display:block;border-radius:6px}
.rollwrap canvas.roll{background:var(--paper)}
.rollwrap canvas.head{cursor:pointer}
.seg{display:inline-flex;border:1px solid var(--note);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;background:transparent;color:var(--note);padding:6px 16px}
.seg button.on{background:var(--note);color:#fff}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
footer{color:var(--muted);font-size:.88rem;margin-top:24px}
footer a{color:var(--note)}
pre.log{font-size:.75rem;max-height:260px;overflow:auto;background:var(--paper);padding:10px;border-radius:6px;white-space:pre-wrap;word-break:break-all}
details summary{cursor:pointer;color:var(--muted)}
.refine{margin-top:14px;padding-top:12px;border-top:1px dashed var(--rule)}
.refine h4{margin:0 0 4px;font-size:.98rem}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
table.store{border-collapse:collapse;font-size:.88rem;margin:8px 0}
.scorebox{margin-top:12px;border:1px solid var(--rule);border-radius:8px;padding:8px}
.scorebox .osmd{background:#fff;color:#000;border-radius:6px;overflow-x:auto;min-height:60px}
@media print{body *{visibility:hidden}.print-target,.print-target *{visibility:visible}.print-target{position:absolute;left:0;top:0;width:100%}.print-target .row{display:none}}
table.store td{padding:2px 14px 2px 0}
</style>
</head>
<body>
<main>
<header>
  <div class="keys" aria-hidden="true"><i></i><b></b><i></i><b></b><i></i><i></i><b></b><i></i><b></b><i></i><b></b><i></i></div>
  <div><h1>PianoForge</h1><p>노래 파일을 올리면 피아노 MIDI와 렌더링 음원을 만듭니다.</p></div>
  <button id="theme" class="ghost small" aria-label="화면 색상 바꾸기">🌓</button>
</header>

<section aria-labelledby="new">
  <h2 id="new">새 변환</h2>
  <div class="field">
    <label class="drop" id="drop" for="audio">
      <input id="audio" type="file" multiple accept=".mp3,.wav,.flac,.ogg,.mid,.midi,audio/*">
      <b>여기에 곡을 끌어다 놓거나 눌러서 고르세요</b>
      <div class="hint">MP3·WAV·FLAC·OGG, 파일당 95 MB 이하, 30분 이하 · 여러 곡을 한꺼번에 올리면 차례로 변환합니다 · 참조 MIDI(.mid)를 함께 올리면 정확도를 측정합니다(곡이 1개일 때)</div>
    </label>
    <ul class="picked" id="picked"></ul>
  </div>
  <div class="chips" role="group" aria-label="프리셋">
    <button class="chip" data-preset="recommended">권장</button>
    <button class="chip" data-preset="best">최고 품질 (느림)</button><button class="chip" data-preset="fidelity">최대 충실도 (가장 느림)</button>
    <button class="chip" data-preset="fast">빠른 미리듣기</button>
    <button class="chip" data-preset="felt">펠트 피아노 감성</button>
    <button class="chip" data-preset="lofi">로파이</button>
  </div>
  <div class="grid">
    <div class="field">
      <label for="src">피아노로 옮길 소리</label>
      <select id="src">
        <option value="nodrums">원곡 − 드럼 (선율·베이스 포함, 권장)</option>
        <option value="other">반주 스템만 (보컬·드럼·베이스 제외)</option>
        <option value="residual">원곡 − 보컬·드럼·베이스 (잔여 신호)</option>
        <option value="piano">피아노 스템 (6스템 모델)</option>
        <option value="mix">분리 없이 원곡 전체</option>
      </select>
    </div>
    <div class="field">
      <label for="q">분석 품질</label>
      <select id="q"><option value="max">최대 (스템별 전사·이중 격자·누락 음 채우기, 느림)</option><option value="fast">빠름 (이전 방식)</option></select>
    </div>
    <div class="field">
      <label for="notes">음 잡기</label>
      <select id="notes"><option value="balanced">균형 (빠지는 음 최소, 복잡하지 않게 — 권장)</option><option value="max">최대 (여린 음까지 더, 틀린 음 조금 더)</option><option value="strict">엄격 (틀린 음 거의 없음, 빠지는 음 많음)</option></select>
    </div>
    <div class="field">
      <label for="room">공간감 (잔향)</label>
      <select id="room"><option value="hall">콘서트홀 (권장)</option><option value="room">작은 방</option><option value="large">큰 홀</option><option value="dry">잔향 없음</option></select>
    </div>
    <div class="field">
      <label for="ptype">피아노 종류</label>
      <select id="ptype"><option value="grand">그랜드 (실제 그랜드 샘플)</option><option value="felt">펠트 (부드럽고 따뜻하게, 흉내)</option><option value="upright">업라이트 (가볍고 밝게, 흉내)</option></select>
    </div>
    <div class="field">
      <label for="texture">질감</label>
      <select id="texture"><option value="clean">깨끗하게</option><option value="lofi">로파이 (테이프·LP 느낌)</option></select>
    </div>
    <div class="field">
      <label for="mech">건반·페달 소리</label>
      <select id="mech"><option value="none">없음</option><option value="light">약하게</option><option value="normal">보통</option></select>
    </div>
    <div class="field">
      <label for="piano">피아노 음색</label>
      <select id="piano"><option value="auto">자동 (Salamander 그랜드 우선)</option><option value="salamander">Salamander Grand (Yamaha C5, 16단계 세기)</option><option value="ydp">YDP Grand (Yamaha Disklavier)</option><option value="system">기본 GM 피아노</option></select>
    </div>
    <div class="field">
      <label for="dev">연산 장치</label>
      <select id="dev"><option value="auto">자동 (GPU 우선)</option><option value="cuda">GPU</option><option value="cpu">CPU</option></select>
    </div>
  </div>
  <label class="check"><input id="pedal" type="checkbox" checked> 서스테인 페달 기록</label>
  <label class="check"><input id="snap" type="checkbox" checked> 박이 안정적인 곡은 음을 곡의 박 격자에 맞추기 (불안정하면 자동으로 건너뜀)</label>
  <label class="check"><input id="nocache" type="checkbox"> 캐시 무시하고 처음부터 계산</label>
  <div class="row" style="margin-top:14px"><button id="go">변환 시작</button><span id="msg" class="hint" role="status" aria-live="polite"></span></div>
</section>

<section aria-labelledby="jobsh">
  <h2 id="jobsh">변환 목록</h2>
  <div id="jobs"><p class="hint">아직 변환한 파일이 없습니다. 위에서 오디오 파일을 골라 시작하세요.</p></div>
</section>

<footer>
  <p id="health">서버 상태 확인 중…</p>
  <details id="storage"><summary>저장공간 · 접속</summary>
    <div id="storebox" class="hint">열면 사용량을 불러옵니다.</div>
    <div class="actions">
      <button class="ghost small" id="purge">안 쓰는 중간 파일 정리</button>
      <button class="ghost small" id="rotate">접속 키 새로 만들기</button>
    </div>
    <p class="hint">접속 키를 새로 만들면 이전 주소는 모두 쓸 수 없게 됩니다(주소가 외부에 알려졌을 때 사용).</p>
  </details>
  <p><a id="zip" href="#">프로젝트 전체 코드 내려받기 (zip)</a></p>
  <p>피아노 음색: Salamander Grand Piano (Alexander Holm) · YDP Grand Piano (FreePats) — CC BY 3.0. 렌더 음원을 공개할 때 출처를 밝혀 주세요.</p>
  <details><summary>서버 로그</summary><pre class="log" id="log"></pre></details>
</footer>
</main>
<script>
/*PURE-START*/
const STAGES=[["normalize","정규화"],["separate","분리"],["transcribe","전사"],["postprocess","후처리"],["render","렌더"],["evaluate","평가"]];
const STAGE_WEIGHT={normalize:2,separate:30,transcribe:50,postprocess:8,render:8,evaluate:2};
const STATE={queued:"대기 중",running:"변환 중",done:"완료",failed:"실패",cancelled:"취소됨"};
const MAX_BYTES=95*1024*1024, AUDIO_EXT=[".mp3",".wav",".flac",".ogg"], MIDI_EXT=[".mid",".midi"];
const PRESETS={
  recommended:{quality:"max",notes:"balanced",room:"hall",piano:"auto",piano_source:"nodrums",pedal:true,snap:true,piano_type:"grand",texture:"clean",mech:"none"},
  best:{quality:"max",notes:"balanced",room:"hall",piano:"auto",piano_source:"nodrums",pedal:true,snap:true,piano_type:"grand",texture:"clean",mech:"light"},
  fast:{quality:"fast",notes:"balanced",room:"room",piano:"auto",piano_source:"nodrums",pedal:true,snap:false,piano_type:"grand",texture:"clean",mech:"none"},
  felt:{quality:"max",notes:"balanced",room:"room",piano:"auto",piano_source:"nodrums",pedal:true,snap:true,piano_type:"felt",texture:"clean",mech:"light"},
  lofi:{quality:"max",notes:"balanced",room:"room",piano:"auto",piano_source:"nodrums",pedal:true,snap:true,piano_type:"upright",texture:"lofi",mech:"light"},
  fidelity:{quality:"max",notes:"max",room:"hall",piano:"auto",piano_source:"nodrums",pedal:true,snap:true,piano_type:"grand",texture:"clean",mech:"none"}
};
const VOLATILE=["elapsed_s","stage_elapsed_s","eta_s","progress","progress_label","queue_position"];
const FILES={"piano.mid":"MIDI","piano.musicxml":"악보 (MusicXML)","prompt.txt":"AI 음악 프롬프트","piano.wav":"WAV","report.md":"리포트","keywords_report.md":"키워드 검출 리포트","piano_theory.mid":"이론 보정 MIDI","piano_theory.wav":"이론 보정 WAV","theory_refine_report.json":"이론 보정 리포트","keywords_report.json":"키워드 JSON","report.json":"리포트 JSON","run_summary.json":"처리 시간"};
const OSMD_URLS=["https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.9/build/opensheetmusicdisplay.min.js","https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.9/build/opensheetmusicdisplay.min.js"];
const RFILES={"piano_fixed.mid":"보정 MIDI","piano_fixed.wav":"보정 WAV","piano_arranged.mid":"편곡 MIDI","piano_arranged.wav":"편곡 WAV","refine_report.md":"보정 리포트","refine_report.json":"보정 리포트 JSON"};
const RSTATE={queued:"보정 대기 중",running:"보정 중…",done:"보정 완료",failed:"보정 실패"};
let TOKEN="__PF_TOKEN__";
const api=p=>"api/"+p+(p.includes("?")?"&":"?")+"k="+encodeURIComponent(TOKEN);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function fmtBytes(n){
  if(n==null||isNaN(n)) return "–";
  if(n<1024) return n+" B";
  const u=["KB","MB","GB","TB"]; let v=n/1024, i=0;
  while(v>=1024&&i<u.length-1){v/=1024;i++;}
  return (v>=100?v.toFixed(0):v.toFixed(1))+" "+u[i];
}
function fmtDur(s){
  if(s==null||isNaN(s)) return "–";
  s=Math.max(0,Math.round(s)); const h=Math.floor(s/3600), m=Math.floor(s%3600/60), r=s%60;
  return h?`${h}:${String(m).padStart(2,"0")}:${String(r).padStart(2,"0")}`:`${m}:${String(r).padStart(2,"0")}`;
}
function extOf(name){const i=String(name).lastIndexOf(".");return i<0?"":String(name).slice(i).toLowerCase();}
function checkFile(file,kind){
  const ok=kind==="audio"?AUDIO_EXT:MIDI_EXT;
  if(!ok.includes(extOf(file.name))) return `지원하지 않는 형식입니다 (${ok.join(", ")})`;
  if(!file.size) return "빈 파일입니다";
  if(file.size>MAX_BYTES) return `파일이 너무 큽니다 (${fmtBytes(file.size)} > ${fmtBytes(MAX_BYTES)})`;
  return "";
}
function splitPicked(files){
  const audio=[], midi=[], rejected=[];
  for(const f of files){
    const e=extOf(f.name);
    if(MIDI_EXT.includes(e)) midi.push(f); else if(AUDIO_EXT.includes(e)||(f.type||"").startsWith("audio/")) audio.push(f); else rejected.push(f);
  }
  return {audio,midi,rejected};
}
function errorHint(job){
  const msg=String(job.error||""), code=job.exit_code;
  if(job.state==="cancelled") return "";
  if(/디스크|No space/i.test(msg)) return "디스크가 가득 찼습니다. 아래 '저장공간 · 접속'에서 안 쓰는 중간 파일을 정리하거나 오래된 곡을 삭제한 뒤 다시 실행하세요.";
  if(/out of memory|CUDA/i.test(msg)) return "GPU 메모리가 부족합니다. 연산 장치를 CPU로 바꾸거나, 더 짧은 곡·'빠름' 품질로 다시 해 보세요.";
  const map={3:"입력 파일 문제입니다. 손상되었거나 소리가 없거나 길이 제한을 넘은 파일일 수 있습니다.",
    4:"음원 분리 단계에서 실패했습니다. GPU 메모리가 부족하면 연산 장치를 CPU로 바꿔 보세요.",
    5:"전사 모델 실행에 실패했습니다. 다시 실행하면 분리까지 끝난 결과를 이어서 씁니다.",
    6:"모델 가중치 다운로드에 실패했습니다. 네트워크를 확인하고 다시 실행하세요.",
    7:"모델 가중치 파일이 손상되었습니다. 캐시를 지우고 다시 실행하세요.",
    9:"피아노 음원 합성에 실패했습니다.",10:"FluidSynth가 설치되어 있지 않습니다.",11:"SoundFont(피아노 음색) 파일을 찾지 못했습니다. 첫 실행이면 다운로드가 끝나기를 기다린 뒤 다시 실행하세요.",
    13:"캐시 파일이 맞지 않습니다. '캐시 무시' 옵션으로 다시 실행하세요."};
  return map[code]||"다시 실행하면 끝난 단계부터 이어서 진행합니다.";
}
function stableSig(job){const o={};for(const k in job) if(!VOLATILE.includes(k)) o[k]=job[k];return JSON.stringify(o);}
function overallPct(job){
  if(job.state==="done") return 100;
  if(job.state!=="running") return 0;
  let pct=0;
  for(const [k] of STAGES){
    if(k===job.stage){pct+=STAGE_WEIGHT[k]*Math.min(1,Math.max(0,job.progress||0));break;}
    pct+=STAGE_WEIGHT[k];
  }
  return Math.min(99,Math.round(pct));
}
function liveText(job){
  if(job.state==="queued") return job.queue_position?`대기 중 — 앞에 ${job.queue_position}곡이 있습니다`:"곧 시작합니다";
  if(job.state!=="running") return "";
  const st=STAGES.find(s=>s[0]===job.stage), parts=[st?st[1]+" 중":"준비 중"];
  if(job.progress_label) parts.push(job.progress_label);
  if(job.elapsed_s!=null) parts.push("경과 "+fmtDur(job.elapsed_s));
  if(job.eta_s!=null) parts.push("남은 시간 약 "+fmtDur(job.eta_s));
  return parts.join(" · ");
}
function steps(job){
  const idx=STAGES.findIndex(s=>s[0]===job.stage);
  return `<ol class="steps">${STAGES.map(([k,l],i)=>{
    let c=""; if(job.state==="done"||(job.stages||[]).some(s=>s.name===k)) c="done";
    else if(job.state==="running"&&i<idx) c="done"; else if(job.state==="running"&&i===idx) c="now";
    return `<li class="${c}">${l}</li>`;}).join("")}</ol>`;
}
function fileUrl(job,name,inline){return api(`jobs/${job.id}/files/${name}${inline?"?inline=1":""}`);}
function abPlayer(job){
  const f=job.files||[], piano=f.includes("piano_preview.mp3")?"piano_preview.mp3":"piano.wav";
  const orig=f.includes("original_preview.mp3")?fileUrl(job,"original_preview.mp3",true):"";
  return `<div class="ab" data-job="${job.id}" data-piano="${fileUrl(job,piano,true)}" data-orig="${orig}">
    <div class="row"><button class="ghost" data-ab-load>▶ 재생 준비</button><span class="hint" data-ab-info></span>
      <span class="seg" data-ab-seg hidden><button data-ab="piano" class="on">피아노</button><button data-ab="orig">원곡</button></span></div>
    <audio controls hidden></audio>
    <div class="rollwrap"><canvas class="roll" data-job="${job.id}" aria-label="피아노 롤"></canvas><canvas class="head" data-job="${job.id}" aria-label="재생 위치(눌러서 이동)"></canvas></div>
    <div class="hint">${orig?"'원곡'과 '피아노'를 눌러 같은 위치에서 바로 비교할 수 있습니다. ":""}피아노 롤을 누르면 그 위치로 이동합니다.</div>
  </div>`;
}
function simplePlayer(job,wav){
  const prev=wav.replace(".wav","_preview.mp3"); const f=(job.files||[]).includes(prev)?prev:wav;
  return `<div class="player" data-src="${fileUrl(job,f,true)}"><button class="ghost" data-play-load>▶ 재생 준비</button> <span class="hint"></span><audio controls hidden></audio></div>`;
}
function refineBox(job){
  if(job.state!=="done") return "";
  const files=(job.files||[]).filter(f=>f in RFILES);
  const links=files.filter(f=>!f.endsWith(".wav")).map(f=>`<a class="btn ghost" href="${fileUrl(job,f)}">${RFILES[f]}</a>`).join("");
  const players=files.filter(f=>f.endsWith(".wav")).map(f=>`<div class="hint">${RFILES[f]}</div>${simplePlayer(job,f)}`).join("");
  const s=job.refine_summary||{}; const list=(t,a)=>a&&a.length?`<p class="hint" style="margin:8px 0 2px">${t}</p><div class="summary">${esc(a.map(x=>"· "+x).join("\n"))}</div>`:"";
  const busy=job.refine_state==="queued"||job.refine_state==="running";
  const cleared=!!job.intermediates_cleared;
  return `<div class="refine"><h4>변환 후 보정</h4>
    <p class="hint">1단계: 오디오와 비교해 근거 없는 음 삭제·시간 오프셋 확인·템포 맵 추가. 2단계(선택): 박 격자 정리, 선율 강조, 페달 정리. 원본 MIDI는 그대로 둡니다.</p>
    ${cleared?`<p class="hint">중간 파일을 정리해서 보정할 수 없습니다. 같은 곡을 다시 변환하면 복구됩니다.</p>`:
      busy?`<p class="state running">${RSTATE[job.refine_state]}</p>`:`<div class="row"><label class="check"><input type="checkbox" data-arr="${job.id}" ${job.refine_arrange?"checked":""}> 2단계 편곡 개선도</label><button class="ghost" data-refine="${job.id}">${job.refine_state==="done"?"다시 보정":"보정하기"}</button></div>`}
    ${job.refine_state&&!busy?`<p class="state ${job.refine_state}">${RSTATE[job.refine_state]}</p>`:""}
    ${job.refine_error?`<div class="err">${esc(job.refine_error)}</div>`:""}
    ${players}${links?`<div class="files">${links}</div>`:""}
    ${list("해결됨",s.resolved)}${list("남은 문제",s.remaining)}${list("판단 불가",s.undeterminable)}
  </div>`;
}
function card(job){
  const f=job.files||[];
  const links=f.filter(n=>n in FILES).map(n=>`<a class="btn ghost" href="${fileUrl(job,n)}">${FILES[n]}</a>`).join("");
  const stageTimes=(job.stages||[]).map(s=>`${s.name.padEnd(12)} ${s.cache_hit?"캐시":"계산"} ${s.duration_s.toFixed(1)} s`).join("\n");
  const meta=[job.duration_s?fmtDur(job.duration_s):null,job.size_bytes?fmtBytes(job.size_bytes):null].filter(Boolean).join(" · ");
  const active=job.state==="queued"||job.state==="running";
  const hint=(job.state==="failed"||job.state==="cancelled")?errorHint(job):"";
  const actions=[];
  if(active) actions.push(`<button class="ghost danger small" data-cancel="${job.id}">취소</button>`);
  if(job.state==="failed"||job.state==="cancelled") actions.push(`<button class="ghost small" data-retry="${job.id}">다시 실행</button>`);
  if(job.state==="done"){
    if(f.includes("piano.musicxml")) actions.unshift(`<button class="small" data-score="${job.id}">🎼 악보 보기</button>`);
    if(f.includes("prompt.txt")) actions.push(`<button class="ghost small" data-copyprompt="${job.id}">프롬프트 복사</button>`);
    if(f.includes("piano.mid")) actions.push(`<a class="btn ghost small" href="${api(`jobs/${job.id}/all.zip`)}">전체 결과 zip</a>`);
    if(!job.intermediates_cleared) actions.push(`<button class="ghost small" data-cleanup="${job.id}" title="분리·정규화 중간 파일(곡당 수백 MB)을 지웁니다. 결과 파일은 그대로 남지만 보정은 할 수 없게 됩니다.">중간 파일 정리</button>`);
  }
  if(!active) actions.push(`<button class="ghost danger small" data-delete="${job.id}">삭제</button>`);
  return `<article class="job" id="j-${job.id}">
    <div class="row" style="justify-content:space-between"><div><h3>${esc(job.name)}</h3><div class="meta">${esc(meta)}${job.options&&job.options.quality==="fast"?" · 빠름":""}</div></div><span class="state ${job.state}">${STATE[job.state]||esc(job.state)}</span></div>
    ${steps(job)}
    ${active?`<div class="bar" aria-hidden="true"><i data-livebar style="width:${overallPct(job)}%"></i></div><div class="hint" data-livetext role="status">${esc(liveText(job))}</div>${job.state==="running"&&job.stage==="separate"?`<div class="hint">음원 분리 중에는 취소해도 이 단계가 끝난 뒤에 멈춥니다.</div>`:""}`:""}
    ${job.error?`<div class="err">${esc(job.error)}${job.exit_code&&job.state==="failed"?` (종료 코드 ${job.exit_code})`:""}</div>`:""}
    ${hint?`<div class="tip">${esc(hint)}</div>`:""}
    ${f.includes("piano.wav")||f.includes("piano_preview.mp3")?abPlayer(job):""}
    ${links?`<div class="files">${links}</div>`:""}
    ${job.summary&&job.summary.length?`<div class="summary">${esc(job.summary.join("\n"))}</div>`:""}
    ${stageTimes?`<details><summary>단계별 처리 시간</summary><div class="summary">${esc(stageTimes)}</div></details>`:""}
    <details class="joblog" data-log="${job.id}"><summary>이 곡의 로그</summary><pre class="log"></pre></details>
    ${actions.length?`<div class="actions">${actions.join("")}</div>`:""}
    ${f.includes("piano.musicxml")?scoreBox(job):""}
    ${refineBox(job)}
  </article>`;
}
function scoreBox(job){
  return `<div class="scorebox" id="score-${job.id}" data-job="${job.id}" hidden>
    <div class="row"><button class="ghost small" data-zoom="-" aria-label="작게">−</button><button class="ghost small" data-zoom="+" aria-label="크게">+</button>
      <button class="ghost small" data-print>인쇄 · PDF</button><a class="btn ghost small" href="${fileUrl(job,"piano.musicxml")}">MusicXML 내려받기</a><span class="hint" data-score-info></span></div>
    <div class="osmd"></div>
    <p class="hint">자동 채보한 악보입니다. 박 격자에 맞춰 정리했으므로 MIDI와 미세한 타이밍이 다를 수 있습니다. una corda·sostenuto·운지는 추정값입니다. MuseScore·Finale·Dorico에서 열어 고칠 수 있습니다.</p>
  </div>`;
}
function rollLayout(data,w,h){
  const notes=data.notes||[]; if(!notes.length) return null;
  let lo=127,hi=0; for(const n of notes){ if(n[2]<lo)lo=n[2]; if(n[2]>hi)hi=n[2]; }
  lo-=2; hi+=2; const end=data.end>0?data.end:1;
  return {lo,hi,end,w,h,rowH:h/(hi-lo+1)};
}
const timeToX=(t,L)=>Math.min(L.w,Math.max(0,t/L.end*L.w));
const xToTime=(x,L)=>Math.min(L.end,Math.max(0,x/L.w*L.end));
function niceStep(L){
  for(const s of [1,2,5,10,15,30,60,120,300]) if(s/L.end*L.w>=64) return s;
  return 600;
}
function drawRoll(g,L,data,col){
  g.clearRect(0,0,L.w,L.h);
  g.fillStyle=col.rule; g.globalAlpha=0.5;
  for(let p=L.lo;p<=L.hi;p++){ if([1,3,6,8,10].includes(p%12)) g.fillRect(0,L.h-(p-L.lo+1)*L.rowH,L.w,L.rowH); }
  g.globalAlpha=1;
  const bars=data.bars||[];
  if(bars.length){
    g.strokeStyle=col.rule; g.lineWidth=1; g.fillStyle=col.muted; g.font="10px sans-serif";
    let lastLabel=-99;
    bars.forEach((t,i)=>{
      const x=Math.round(timeToX(t,L))+0.5;
      g.beginPath(); g.moveTo(x,0); g.lineTo(x,L.h); g.stroke();
      if(x-lastLabel>=34){g.fillText(String(i+1),x+2,L.h-4);lastLabel=x;}
    });
  }
  g.fillStyle=col.note;
  for(const [on,off,p,v] of data.notes){
    g.globalAlpha=0.35+0.65*v/127;
    g.fillRect(timeToX(on,L),L.h-(p-L.lo+1)*L.rowH,Math.max(1,(off-on)/L.end*L.w),Math.max(1,L.rowH-0.5));
  }
  g.globalAlpha=1;
  const chords=data.chords||[]; let lastX=-99;
  g.fillStyle=col.ink; g.font="bold 11px sans-serif";
  for(const [t,label] of chords){ const x=timeToX(t,L); if(x-lastX>=30){g.fillText(label,x+2,12);lastX=x;} }
  const step=niceStep(L); g.fillStyle=col.muted; g.font="10px sans-serif";
  for(let t=step;t<L.end;t+=step) g.fillText(fmtDur(t),timeToX(t,L)+2,L.h-16);
}
function drawHead(g,L,t,col){
  g.clearRect(0,0,L.w,L.h);
  const x=timeToX(t,L); g.fillStyle=col.head; g.fillRect(Math.max(0,x-1),0,2,L.h);
}
/*PURE-END*/

/* ------------------------------------------------------------------ DOM / network */
const ROLLS={}, BLOBS={}, TITLE=document.title;
const prevState={};
let last="", pollTimer=0;
const $=s=>document.querySelector(s);
const colors=()=>{const c=getComputedStyle(document.documentElement),v=n=>c.getPropertyValue(n).trim();return{rule:v("--rule"),note:v("--note"),muted:v("--muted"),ink:v("--ink"),head:v("--head")};};
const store={get(k){try{return localStorage.getItem(k);}catch(e){return null;}},set(k,v){try{localStorage.setItem(k,v);}catch(e){}}};

/* theme */
function applyTheme(t){ if(t==="light"||t==="dark") document.documentElement.dataset.theme=t; else delete document.documentElement.dataset.theme; }
applyTheme(store.get("pf_theme"));
$("#theme").onclick=()=>{ const cur=document.documentElement.dataset.theme||"auto"; const nxt={auto:"light",light:"dark",dark:"auto"}[cur]; store.set("pf_theme",nxt); applyTheme(nxt); $("#msg").textContent="화면 색상: "+({auto:"기기 설정 따라가기",light:"밝게",dark:"어둡게"})[nxt]; setTimeout(()=>drawAllRolls(),50); };

/* options */
const OPT_FIELDS={piano_source:"#src",quality:"#q",piano:"#piano",notes:"#notes",room:"#room",device:"#dev",piano_type:"#ptype",texture:"#texture",mech:"#mech"};
function readOptions(){const o={};for(const k in OPT_FIELDS)o[k]=$(OPT_FIELDS[k]).value;o.pedal=$("#pedal").checked;o.snap=$("#snap").checked;o.no_cache=$("#nocache").checked;return o;}
function applyOptions(o){for(const k in OPT_FIELDS){ if(o[k]!=null && [...$(OPT_FIELDS[k]).options].some(x=>x.value===o[k])) $(OPT_FIELDS[k]).value=o[k]; }
  if(o.pedal!=null)$("#pedal").checked=!!o.pedal; if(o.snap!=null)$("#snap").checked=!!o.snap; if(o.no_cache!=null)$("#nocache").checked=!!o.no_cache;}
try{const saved=JSON.parse(store.get("pf_opts")||"null"); if(saved) applyOptions(saved);}catch(e){}
document.querySelectorAll("[data-preset]").forEach(b=>b.onclick=()=>{applyOptions(PRESETS[b.dataset.preset]);store.set("pf_opts",JSON.stringify(readOptions()));$("#msg").textContent=`'${b.textContent}' 설정을 적용했습니다.`;});
document.querySelectorAll("select,#pedal,#snap,#nocache").forEach(el=>el.addEventListener("change",()=>store.set("pf_opts",JSON.stringify(readOptions()))));

/* file picking: click, drag & drop, several songs */
let picked={audio:[],midi:null};
function renderPicked(){
  const rows=[];
  picked.audio.forEach((f,i)=>{const e=checkFile(f,"audio");rows.push(`<li class="${e?"bad":""}"><span>🎵 ${esc(f.name)} <span class="meta">${fmtBytes(f.size)}${e?" — "+esc(e):""}</span></span><button class="ghost small" data-rm="a${i}" aria-label="빼기">✕</button></li>`);});
  if(picked.midi){const e=checkFile(picked.midi,"midi");rows.push(`<li class="${e?"bad":""}"><span>🎼 참조 MIDI: ${esc(picked.midi.name)} <span class="meta">${fmtBytes(picked.midi.size)}${e?" — "+esc(e):""}</span></span><button class="ghost small" data-rm="m" aria-label="빼기">✕</button></li>`);}
  $("#picked").innerHTML=rows.join("");
  $("#picked").querySelectorAll("[data-rm]").forEach(b=>b.onclick=()=>{const k=b.dataset.rm; if(k==="m")picked.midi=null; else picked.audio.splice(+k.slice(1),1); renderPicked();});
  const bad=picked.audio.some(f=>checkFile(f,"audio"))||(picked.midi&&checkFile(picked.midi,"midi"));
  $("#go").disabled=!picked.audio.length||!!bad;
  if(picked.midi&&picked.audio.length>1) $("#msg").textContent="참조 MIDI는 곡이 1개일 때만 쓸 수 있어 이번에는 무시됩니다.";
}
function pick(files){
  const s=splitPicked([...files]);
  for(const f of s.audio) if(!picked.audio.some(x=>x.name===f.name&&x.size===f.size)) picked.audio.push(f);
  if(s.midi.length) picked.midi=s.midi[0];
  $("#msg").textContent=s.rejected.length?`지원하지 않는 파일 ${s.rejected.length}개는 건너뛰었습니다: ${s.rejected.map(f=>f.name).join(", ")}`:"";
  renderPicked();
}
$("#audio").onchange=e=>{pick(e.target.files);e.target.value="";};
const drop=$("#drop");
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("over");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("over");}));
drop.addEventListener("drop",e=>{if(e.dataTransfer&&e.dataTransfer.files)pick(e.dataTransfer.files);});
renderPicked();

function xhrUpload(file,kind,onProgress){
  return new Promise((resolve,reject)=>{
    const x=new XMLHttpRequest();
    x.open("POST",api(`upload?kind=${kind}&name=${encodeURIComponent(file.name)}`));
    x.upload.onprogress=e=>{if(e.lengthComputable)onProgress(e.loaded/e.total);};
    x.onload=()=>{let j={};try{j=JSON.parse(x.responseText);}catch(_){}
      if(x.status>=200&&x.status<300) resolve(j); else reject(new Error(j.error||("서버 응답 "+x.status)));};
    x.onerror=()=>reject(new Error("네트워크 오류 — 연결을 확인하세요"));
    x.send(file);
  });
}
$("#go").onclick=async()=>{
  const list=picked.audio.slice(), midi=list.length===1?picked.midi:null, options=readOptions();
  store.set("pf_opts",JSON.stringify(options));
  $("#go").disabled=true; const failures=[]; let started=0;
  for(let i=0;i<list.length;i++){
    const f=list[i], tag=list.length>1?`(${i+1}/${list.length}) `:"";
    try{
      const prog=p=>{$("#msg").textContent=`${tag}${f.name} 올리는 중 ${Math.round(p*100)}%`;};
      const reference=midi?(await xhrUpload(midi,"reference",()=>{})).id:null;
      const up=await xhrUpload(f,"audio",prog);
      $("#msg").textContent=`${tag}${f.name} 대기열에 추가하는 중…`;
      const r=await fetch(api("jobs"),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({audio:up.id,reference,options})});
      const j=await r.json(); if(!r.ok) throw new Error(j.error);
      started++;
    }catch(e){failures.push(`${f.name}: ${e.message}`);}
  }
  picked={audio:[],midi:null}; renderPicked();
  $("#msg").textContent=failures.length?`${started}곡을 시작했고 ${failures.length}곡은 실패했습니다 — ${failures.join(" / ")}`:`${started}곡을 대기열에 추가했습니다.`;
  refresh();
};

/* players */
async function fetchBlob(url,onP){
  if(BLOBS[url]) return BLOBS[url];
  const r=await fetch(url); if(!r.ok) throw new Error(r.status);
  const total=+r.headers.get("Content-Length")||0, reader=r.body.getReader(), parts=[]; let got=0;
  for(;;){const {done,value}=await reader.read(); if(done)break; parts.push(value); got+=value.length; onP(total?Math.round(100*got/total):Math.round(got/104857.6)/10);}
  BLOBS[url]=URL.createObjectURL(new Blob(parts,{type:r.headers.get("Content-Type")||"audio/mpeg"}));
  return BLOBS[url];
}
function setupAB(box,autoplay){
  const au=box.querySelector("audio"), btn=box.querySelector("[data-ab-load]"), seg=box.querySelector("[data-ab-seg]");
  box._srcs={piano:BLOBS[box.dataset.piano]}; if(box.dataset.orig&&BLOBS[box.dataset.orig]){box._srcs.orig=BLOBS[box.dataset.orig];seg.hidden=false;}
  box._cur="piano"; au.src=box._srcs.piano; au.hidden=false; btn.hidden=true; box.querySelector("[data-ab-info]").textContent="";
  if(autoplay) au.play().catch(()=>{});
}
async function loadAB(box,autoplay){
  const info=box.querySelector("[data-ab-info]"), btn=box.querySelector("[data-ab-load]");
  if(!btn.hidden) btn.disabled=true;
  try{
    await fetchBlob(box.dataset.piano,p=>info.textContent=`피아노 음원 내려받는 중 ${p}${p<=100&&Number.isInteger(p)?"%":" MB"}`);
    if(box.dataset.orig) await fetchBlob(box.dataset.orig,p=>info.textContent=`원곡 내려받는 중 ${p}${p<=100&&Number.isInteger(p)?"%":" MB"}`);
    setupAB(box,autoplay);
  }catch(e){info.textContent="내려받기 실패: "+e.message; btn.disabled=false;}
}
function switchAB(box,which){
  const au=box.querySelector("audio"); if(!box._srcs||!box._srcs[which]||box._cur===which) return;
  const t=au.currentTime, playing=!au.paused; box._cur=which; au.src=box._srcs[which];
  au.addEventListener("loadedmetadata",()=>{try{au.currentTime=Math.min(t,au.duration||t);}catch(e){} if(playing)au.play().catch(()=>{});},{once:true});
  box.querySelectorAll("[data-ab]").forEach(b=>b.classList.toggle("on",b.dataset.ab===which));
}
async function preparePlayer(box,autoplay){
  const url=box.dataset.src, btn=box.querySelector("button"), info=box.querySelector("span"), au=box.querySelector("audio");
  btn.disabled=true;
  try{ au.src=await fetchBlob(url,p=>info.textContent=`내려받는 중 ${p}${Number.isInteger(p)&&p<=100?"%":" MB"}`); au.hidden=false; btn.hidden=true; info.textContent=""; if(autoplay)au.play().catch(()=>{}); }
  catch(e){info.textContent="내려받기 실패: "+e.message; btn.disabled=false;}
}

/* piano roll with playhead */
async function rollData(id){ if(!ROLLS[id]) ROLLS[id]=await (await fetch(api(`jobs/${id}/roll`))).json(); return ROLLS[id]; }
async function drawRollFor(box){
  const base=box.querySelector("canvas.roll"), head=box.querySelector("canvas.head"); if(!base) return;
  const data=await rollData(box.dataset.job), dpr=window.devicePixelRatio||1, w=base.clientWidth, h=base.clientHeight;
  for(const c of [base,head]){c.width=w*dpr;c.height=h*dpr;c.getContext("2d").setTransform(dpr,0,0,dpr,0,0);}
  const L=rollLayout(data,w,h); box._layout=L;
  if(!L){const g=base.getContext("2d");g.fillStyle=colors().muted;g.font="14px sans-serif";g.fillText("검출된 노트가 없습니다",12,24);return;}
  drawRoll(base.getContext("2d"),L,data,colors());
  const au=box.querySelector("audio"); drawHead(head.getContext("2d"),L,au.currentTime||0,colors());
}
function bindRoll(box){
  const au=box.querySelector("audio"), head=box.querySelector("canvas.head"); let raf=0;
  const tick=()=>{ if(box._layout) drawHead(head.getContext("2d"),box._layout,au.currentTime,colors()); if(!au.paused&&!au.ended) raf=requestAnimationFrame(tick); };
  ["play","seeked","pause","ended","timeupdate"].forEach(ev=>au.addEventListener(ev,()=>{cancelAnimationFrame(raf);tick();}));
  head.addEventListener("click",e=>{
    if(au.hidden){loadAB(box,true);return;}
    if(!box._layout) return; const r=head.getBoundingClientRect();
    au.currentTime=xToTime((e.clientX-r.left)/r.width*box._layout.w,box._layout); tick();
  });
}
function drawAllRolls(){document.querySelectorAll(".ab").forEach(b=>drawRollFor(b).catch(()=>{}));}

/* sheet music (OpenSheetMusicDisplay, loaded only when a score is opened) */
let osmdReady=null;
function loadScript(src){return new Promise((res,rej)=>{const s=document.createElement("script");s.src=src;s.onload=()=>res();s.onerror=()=>{s.remove();rej(new Error(src));};document.head.appendChild(s);});}
function loadOSMD(){
  if(window.opensheetmusicdisplay) return Promise.resolve();
  if(!osmdReady) osmdReady=(async()=>{
    for(const u of OSMD_URLS){try{await loadScript(u); if(window.opensheetmusicdisplay) return;}catch(e){}}
    osmdReady=null; throw new Error("악보 표시 프로그램을 불러오지 못했습니다(인터넷 연결 필요). 'MusicXML 내려받기'로 받아 MuseScore 등에서 여세요.");
  })();
  return osmdReady;
}
async function openScore(box){
  const info=box.querySelector("[data-score-info]"), view=box.querySelector(".osmd");
  box.hidden=false; if(box._osmd) return;
  info.textContent="악보 불러오는 중…";
  try{
    await loadOSMD();
    const r=await fetch(api(`jobs/${box.dataset.job}/files/piano.musicxml?inline=1`)); if(!r.ok) throw new Error("서버 응답 "+r.status);
    const xml=await r.text();
    const osmd=new opensheetmusicdisplay.OpenSheetMusicDisplay(view,{autoResize:true,backend:"svg",drawTitle:true,drawPartNames:false});
    await osmd.load(xml); osmd.zoom=window.innerWidth<600?0.55:0.8; osmd.render(); box._osmd=osmd; info.textContent="";
  }catch(e){info.textContent="악보를 표시하지 못했습니다: "+e.message;}
}
function wireScore(el){
  el.querySelectorAll("[data-score]").forEach(b=>b.onclick=()=>{const box=document.getElementById("score-"+b.dataset.score); if(!box) return;
    if(!box.hidden&&box._osmd){box.hidden=true;b.textContent="🎼 악보 보기";return;} b.textContent="🎼 악보 닫기"; openScore(box);});
  el.querySelectorAll(".scorebox").forEach(box=>{
    box.querySelectorAll("[data-zoom]").forEach(z=>z.onclick=()=>{if(!box._osmd)return; box._osmd.zoom=Math.min(2,Math.max(0.3,box._osmd.zoom+(z.dataset.zoom==="+"?0.1:-0.1))); box._osmd.render();});
    const pr=box.querySelector("[data-print]"); if(pr) pr.onclick=()=>{if(!box._osmd)return; box.classList.add("print-target"); window.print(); setTimeout(()=>box.classList.remove("print-target"),500);};
  });
}

/* job cards */
async function post(path,body){
  const r=await fetch(api(path),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  const j=await r.json().catch(()=>({})); if(!r.ok) throw new Error(j.error||("서버 응답 "+r.status)); return j;
}
function wire(el){
  el.querySelectorAll(".ab").forEach(box=>{
    box.querySelector("[data-ab-load]").onclick=()=>loadAB(box,true);
    box.querySelectorAll("[data-ab]").forEach(b=>b.onclick=()=>switchAB(box,b.dataset.ab));
    bindRoll(box);
    if(BLOBS[box.dataset.piano]&&(!box.dataset.orig||BLOBS[box.dataset.orig])) setupAB(box,false);
    requestAnimationFrame(()=>drawRollFor(box).catch(()=>{}));
  });
  el.querySelectorAll(".player").forEach(p=>{p.querySelector("[data-play-load]").onclick=()=>preparePlayer(p,true); if(BLOBS[p.dataset.src])preparePlayer(p,false);});
  wireScore(el);
  const act=(sel,fn)=>el.querySelectorAll(sel).forEach(b=>b.onclick=async()=>{b.disabled=true;try{await fn(b);}catch(e){alert(e.message);}finally{b.disabled=false;}last="";refresh();});
  act("[data-retry]",b=>post(`jobs/${b.dataset.retry}/retry`));
  el.querySelectorAll("[data-copyprompt]").forEach(b=>b.onclick=async()=>{
    try{const txt=await (await fetch(api(`jobs/${b.dataset.copyprompt}/files/prompt.txt?inline=1`))).text();
      if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(txt);$("#msg").textContent="AI 음악 프롬프트를 복사했습니다.";}
      else{prompt("복사해서 쓰세요:",txt);}}
    catch(e){alert("프롬프트를 가져오지 못했습니다: "+e.message);}});
  act("[data-cancel]",b=>post(`jobs/${b.dataset.cancel}/cancel`));
  act("[data-delete]",async b=>{if(!confirm("이 곡과 결과 파일을 모두 삭제할까요? 되돌릴 수 없습니다.")){return;} const r=await post(`jobs/${b.dataset.delete}/delete`); $("#msg").textContent=`삭제했습니다 (${fmtBytes(r.freed_bytes)} 확보).`;});
  act("[data-cleanup]",async b=>{if(!confirm("분리·정규화 중간 파일을 지웁니다. 결과 파일(MIDI·WAV·리포트)은 그대로이지만 이 곡은 보정할 수 없게 됩니다. 계속할까요?")){return;} const r=await post(`jobs/${b.dataset.cleanup}/cleanup`); $("#msg").textContent=`중간 파일을 정리했습니다 (${fmtBytes(r.freed_bytes)} 확보).`;});
  el.querySelectorAll("[data-refine]").forEach(b=>b.onclick=async()=>{
    const id=b.dataset.refine, arrange=!!el.querySelector(`[data-arr="${id}"]`)?.checked; b.disabled=true;
    try{await post(`jobs/${id}/refine`,{arrange});}catch(e){alert("보정을 시작하지 못했습니다: "+e.message);}finally{b.disabled=false;} last=""; refresh();});
  el.querySelectorAll("details.joblog").forEach(d=>d.addEventListener("toggle",async()=>{if(d.open)d.querySelector("pre").textContent=await (await fetch(api(`log?job=${d.dataset.log}`))).text();}));
}
function liveUpdate(node,job){
  const t=node.querySelector("[data-livetext]"); if(t) t.textContent=liveText(job);
  const b=node.querySelector("[data-livebar]"); if(b) b.style.width=overallPct(job)+"%";
}
function renderJobs(jobs){
  const box=$("#jobs");
  if(!jobs.length){box.innerHTML='<p class="hint">아직 변환한 파일이 없습니다. 위에서 오디오 파일을 골라 시작하세요.</p>';return;}
  const old=new Map([...box.querySelectorAll("article.job")].map(e=>[e.id,e]));
  const nodes=jobs.map(j=>{
    const sig=stableSig(j), cur=old.get("j-"+j.id);
    if(cur&&cur.dataset.sig===sig){liveUpdate(cur,j);return cur;}  // untouched card: players keep playing, buttons stay put
    const t=document.createElement("template"); t.innerHTML=card(j).trim();
    const n=t.content.firstElementChild; n.dataset.sig=sig; wire(n);
    if(cur) cur.replaceWith(n);
    return n;});
  const same=nodes.length===box.children.length&&nodes.every((n,i)=>box.children[i]===n);
  if(!same) box.replaceChildren(...nodes);
}
function notifyChanges(jobs){
  let finished=0;
  for(const j of jobs){
    const was=prevState[j.id]; prevState[j.id]=j.state;
    if((was==="running"||was==="queued")&&(j.state==="done"||j.state==="failed")) finished++;
  }
  if(finished&&document.hidden){document.title=(jobs.some(j=>j.state==="failed")?"⚠️ ":"✅ ")+"변환 끝 · "+TITLE; if(navigator.vibrate)navigator.vibrate(200);}
}
document.addEventListener("visibilitychange",()=>{if(!document.hidden)document.title=TITLE;});
async function refresh(){
  clearTimeout(pollTimer);
  let delay=5000;
  try{
    const jobs=await (await fetch(api("jobs"))).json();
    notifyChanges(jobs);
    const sig=JSON.stringify(jobs.map(stableSig));
    if(sig!==last){last=sig;}
    renderJobs(jobs);
    const running=jobs.some(j=>j.state==="running"||j.refine_state==="running"), queued=jobs.some(j=>j.state==="queued"||j.refine_state==="queued");
    delay=running?1000:queued?2000:8000;
    if((running||queued)&&$("details[open] #log")) $("#log").textContent=await (await fetch(api("log"))).text();
  }catch(e){delay=5000;}
  pollTimer=setTimeout(refresh,delay);
}

/* footer: health, storage, access key */
async function health(){
  try{const h=await (await fetch(api("health"))).json();
    const gpu=h.gpu?`GPU ${h.gpu.name} (${h.gpu.free_gb}/${h.gpu.total_gb} GB 여유)`:`장치 ${h.device}`;
    $("#health").textContent=`v${h.version} · ${gpu} · 디스크 ${h.disk_free_gb} GB 여유 · 피아노 ${h.soundfont_ok?h.soundfont:"없음"} · 2차 전사 모델 ${h.transkun?"Transkun":"없음"} · FluidSynth ${h.fluidsynth_lib?"있음":"없음"} · 대기 ${h.queued}곡`;}
  catch(e){$("#health").textContent="서버에 연결하지 못했습니다.";}
}
async function showStorage(){
  try{const s=await (await fetch(api("storage"))).json(), row=(a,b)=>`<tr><td>${a}</td><td>${fmtBytes(b)}</td></tr>`;
    $("#storebox").innerHTML=`<table class="store">${row("남은 디스크",s.free_bytes)}${row("완료된 곡 결과",s.jobs_bytes)}${row("업로드한 원본",s.uploads_bytes)}${row("중간 파일(캐시)",s.cache_bytes)}${row("└ 더는 안 쓰는 것",s.orphan_bytes)}${row("모델 가중치",s.models_bytes)}${row("피아노 음색",s.soundfonts_bytes)}</table>`;}
  catch(e){$("#storebox").textContent="사용량을 불러오지 못했습니다.";}
}
$("#storage").addEventListener("toggle",e=>{if(e.target.open)showStorage();});
$("#purge").onclick=async()=>{try{const r=await post("storage/purge");$("#msg").textContent=`안 쓰는 중간 파일을 정리했습니다 (${fmtBytes(r.freed_bytes)} 확보).`;showStorage();health();}catch(e){alert(e.message);}};
$("#rotate").onclick=async()=>{
  if(!confirm("접속 키를 새로 만들면 지금까지 쓰던 주소(다른 기기 포함)가 모두 막힙니다. 계속할까요?")) return;
  try{const r=await post("rotate-token"); TOKEN=r.token; const url=location.origin+location.pathname+"?k="+encodeURIComponent(TOKEN);
    prompt("새 주소입니다. 복사해 두세요 (이 페이지는 새 주소로 이동합니다):",url); location.href=url;}
  catch(e){alert("바꾸지 못했습니다: "+e.message);}
};
$("#log").parentElement.addEventListener("toggle",async e=>{if(e.target.open)$("#log").textContent=await (await fetch(api("log"))).text();});
window.addEventListener("resize",()=>{clearTimeout(window._rs);window._rs=setTimeout(drawAllRolls,150);});
$("#jobs").innerHTML="";
$("#zip").href=api("project.zip");
health(); refresh(); setInterval(health,30000);
</script>
</body>
</html>
'''

# =========================================================================== embedded project
# 아래는 PianoForge 프로젝트 파일 원문이다. write_project()가 PROJECT 폴더에 그대로 쓴다.
# ---------------------------------------------------------------------------
# BUILD ARTIFACT: generated source bundle (do not hand-edit individual files)
# ---------------------------------------------------------------------------
SOURCES_BUNDLE_SHA256 = "a96a89e625a80f34146ca3cdbe17bada6b2bfc6513443aabca73437672b6afef"
SOURCES_BUNDLE_B64 = """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"""

def _load_sources_bundle() -> dict[str, str]:
    raw = base64.b64decode(SOURCES_BUNDLE_B64.encode("ascii"))
    if hashlib.sha256(raw).hexdigest() != SOURCES_BUNDLE_SHA256:
        raise RuntimeError("PianoForge source bundle integrity check failed")
    out: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        for name in zf.namelist():
            if not name.startswith("pianoforge/") or name.endswith("/"):
                continue
            rel = name[len("pianoforge/"):]
            if not rel or rel.startswith("../") or "/../" in rel:
                raise RuntimeError(f"unsafe source bundle path: {name}")
            out[rel] = zf.read(name).decode("utf-8")
    return out

SOURCES: dict[str, str] = _load_sources_bundle()
SOURCES_BUILD_MANIFEST = _sources_build_manifest()

if __name__ == "__main__":
    APP = main()
