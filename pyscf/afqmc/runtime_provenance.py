from __future__ import annotations

import os
import platform
import socket
import subprocess
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

_FIELD_NAME_WIDTH = 15
_FIELD_CONTINUATION_PREFIX = f" {'':<{_FIELD_NAME_WIDTH}s}   "
_PROVENANCE_WRAP_WIDTH = 88
_CPU_BACKEND_NOTE = (
    "If you have access to a supported GPU, AFQMC can benefit substantially "
    "from GPU acceleration and is often much faster on GPU than CPU for "
    "realistic workloads."
)


@dataclass(frozen=True)
class HostInfo:
    hostname: str
    system: str
    release: str
    machine: str
    processor: str
    cpu_count: int | None


@dataclass(frozen=True)
class GitInfo:
    root: Path | None
    branch: str | None
    commit: str | None
    status_lines: tuple[str, ...]


@dataclass(frozen=True)
class JaxInfo:
    version: str | None
    jaxlib_version: str | None
    backend: str | None
    local_device_count: int | None
    global_device_count: int | None
    process_index: int | None
    process_count: int | None
    device_summary: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeProvenance:
    run_started: datetime
    host: HostInfo
    git: GitInfo
    jax: JaxInfo


def _format_field(name: str, value: object) -> str:
    return f" {name:<{_FIELD_NAME_WIDTH}s} = {value}"


def _format_continuation(text: str, *, width: int = _PROVENANCE_WRAP_WIDTH) -> tuple[str, ...]:
    wrap_width = max(1, width - len(_FIELD_CONTINUATION_PREFIX))
    return tuple(
        f"{_FIELD_CONTINUATION_PREFIX}{line}"
        for line in textwrap.wrap(text, width=wrap_width)
    )


def _format_timestamp(dt: datetime) -> str:
    tz_name = dt.strftime("%Z")
    offset = dt.strftime("%z")
    if len(offset) == 5:
        offset = f"{offset[:3]}:{offset[3:]}"

    out = dt.strftime("%Y-%m-%d %H:%M:%S")
    if tz_name:
        out = f"{out} {tz_name}"
    if offset:
        out = f"{out} (UTC{offset})"
    return out


def collect_host_info() -> HostInfo:
    uname = platform.uname()
    return HostInfo(
        hostname=socket.gethostname(),
        system=uname.system,
        release=uname.release,
        machine=uname.machine,
        processor=uname.processor,
        cpu_count=os.cpu_count(),
    )


def _run_git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _find_git_root(repo_root: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root).expanduser().resolve())
    else:
        candidates.extend([Path(__file__).resolve().parent, Path.cwd().resolve()])

    seen: set[Path] = set()
    for start in candidates:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / ".git").exists():
                return candidate

    for start in candidates:
        root = _run_git(start, "rev-parse", "--show-toplevel")
        if root is not None:
            return Path(root).expanduser().resolve()

    return None


def collect_git_info(repo_root: Path | None = None) -> GitInfo:
    root = _find_git_root(repo_root)
    if root is None:
        return GitInfo(root=None, branch=None, commit=None, status_lines=())

    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--short")
    status_lines = tuple(line for line in (status or "").splitlines() if line.strip())
    return GitInfo(root=root, branch=branch, commit=commit, status_lines=status_lines)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _summarize_jax_devices(devices: list[Any]) -> tuple[str, ...]:
    groups = Counter()
    for device in devices:
        platform_name = getattr(device, "platform", "unknown")
        device_kind = getattr(device, "device_kind", platform_name)
        groups[(platform_name, device_kind)] += 1
    return tuple(
        f"{count} x {platform_name} ({device_kind})"
        for (platform_name, device_kind), count in sorted(groups.items())
    )


def collect_jax_info() -> JaxInfo:
    try:
        import jax
    except Exception:
        return JaxInfo(
            version=_package_version("jax"),
            jaxlib_version=_package_version("jaxlib"),
            backend=None,
            local_device_count=None,
            global_device_count=None,
            process_index=None,
            process_count=None,
            device_summary=(),
        )

    devices = list(jax.devices())
    backend = devices[0].platform if devices else None
    return JaxInfo(
        version=getattr(jax, "__version__", None) or _package_version("jax"),
        jaxlib_version=_package_version("jaxlib"),
        backend=backend,
        local_device_count=jax.local_device_count(),
        global_device_count=jax.device_count(),
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        device_summary=_summarize_jax_devices(devices),
    )


def collect_runtime_provenance(repo_root: Path | None = None) -> RuntimeProvenance:
    return RuntimeProvenance(
        run_started=datetime.now().astimezone(),
        host=collect_host_info(),
        git=collect_git_info(repo_root),
        jax=collect_jax_info(),
    )


def format_runtime_provenance(
    info: RuntimeProvenance,
    *,
    max_status_lines: int = 8,
) -> tuple[str, ...]:
    lines = [
        "******** Runtime ********",
        _format_field("run_started", _format_timestamp(info.run_started)),
        _format_field("hostname", info.host.hostname),
        _format_field("system", f"{info.host.system} {info.host.release}".strip()),
        _format_field("machine", info.host.machine),
    ]
    if info.host.processor:
        lines.append(_format_field("processor", info.host.processor))
    if info.host.cpu_count is not None:
        lines.append(_format_field("cpu_count", info.host.cpu_count))
    if info.jax.version is not None:
        lines.append(_format_field("jax_version", info.jax.version))
    if info.jax.jaxlib_version is not None:
        lines.append(_format_field("jaxlib_version", info.jax.jaxlib_version))
    if info.jax.backend is not None:
        lines.append(_format_field("jax_backend", info.jax.backend))
        if info.jax.backend == "cpu":
            lines.append(_format_field("jax_note", "running on CPU"))
            lines.extend(_format_continuation(_CPU_BACKEND_NOTE))
    if info.jax.local_device_count is not None and info.jax.global_device_count is not None:
        lines.append(
            _format_field(
                "jax_devices",
                f"{info.jax.local_device_count} local / {info.jax.global_device_count} global",
            )
        )
    if info.jax.process_index is not None and info.jax.process_count is not None:
        lines.append(
            _format_field(
                "jax_process",
                f"{info.jax.process_index} / {info.jax.process_count}",
            )
        )
    if info.jax.device_summary:
        for i, summary in enumerate(info.jax.device_summary):
            name = "jax_device_kind" if i == 0 else ""
            lines.append(_format_field(name, summary) if name else f" {'':15s}   {summary}")

    if info.git.root is None:
        lines.append(_format_field("git_commit", "unavailable"))
    else:
        lines.append(_format_field("git_root", info.git.root))
        lines.append(_format_field("git_branch", info.git.branch or "unknown"))
        lines.append(_format_field("git_commit", info.git.commit or "unknown"))
        if not info.git.status_lines:
            lines.append(_format_field("git_status", "clean"))
        else:
            n_status = len(info.git.status_lines)
            n_shown = min(n_status, max_status_lines)
            extra = "" if n_status == n_shown else f"; first {n_shown} shown"
            lines.append(_format_field("git_status", f"dirty ({n_status} entries{extra})"))
            lines.extend(f"  {line}" for line in info.git.status_lines[:max_status_lines])

    lines.append("")
    return tuple(lines)


def print_runtime_provenance(repo_root: Path | None = None) -> None:
    info = collect_runtime_provenance(repo_root)
    for line in format_runtime_provenance(info):
        print(line)


def dump_runtime_provenance(log: Any, repo_root: Path | None = None) -> None:
    info = collect_runtime_provenance(repo_root)
    for line in format_runtime_provenance(info):
        log.info("%s", line)
