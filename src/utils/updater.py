"""
Update checker for the portable Windows installation.

Runs early in main() on every invocation. Hits the GitHub releases API
behind a cooldown, prompts when a newer version is available, and (on
acceptance or when --auto_update is set) re-runs the official install
script to refresh the bundle in place.

Designed to be cheap and non-fatal: any failure is silently ignored so
the user's actual command keeps working.
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error

REPO = "NevermindNilas/TheAnimeScripter"
INSTALL_SCRIPT_URL = "https://tas.nevermindnilas.dev/install.ps1"
RELEASE_API = f"https://api.github.com/repos/{REPO}/releases/latest"
HTTP_TIMEOUT = 3.0


def _install_dir():
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def _state_path():
    return os.path.join(_install_dir(), "tas-update-state.json")


def _read_installed_version():
    try:
        from src.version import __version__
        return str(__version__)
    except Exception:
        return None


def _load_state():
    try:
        with open(_state_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    try:
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def _is_newer(latest, installed):
    def parse(v):
        return tuple(int(x) for x in v.lstrip("vV").split(".") if x.isdigit())
    try:
        return parse(latest) > parse(installed)
    except Exception:
        return latest != installed


def _fetch_latest_release():
    req = urllib.request.Request(
        RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "TheAnimeScripter-Updater",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _write_finalizer(install_path, current_pid):
    fd, path = tempfile.mkstemp(suffix=".cmd", prefix="tas-update-")
    os.close(fd)
    script = (
        '@echo off\r\n'
        'setlocal\r\n'
        'echo.\r\n'
        'echo Waiting for TheAnimeScripter to close...\r\n'
        ':wait\r\n'
        f'tasklist /FI "PID eq {current_pid}" 2>nul | findstr /C:"{current_pid}" >nul\r\n'
        'if not errorlevel 1 (\r\n'
        '    timeout /t 1 /nobreak >nul\r\n'
        '    goto wait\r\n'
        ')\r\n'
        f'powershell -NoProfile -ExecutionPolicy Bypass -Command "iex \\"& {{ $(irm {INSTALL_SCRIPT_URL}) }} -InstallPath \'{install_path}\' -Force -AddToPath:$false\\""\r\n'
        'echo.\r\n'
        'echo Update complete. Press any key to close.\r\n'
        'pause >nul\r\n'
        'del "%~f0"\r\n'
    )
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write(script)
    return path


def _spawn_finalizer(finalizer_path):
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_CONSOLE = 0x00000010
    subprocess.Popen(
        ["cmd.exe", "/c", finalizer_path],
        creationflags=DETACHED_PROCESS | CREATE_NEW_CONSOLE,
        close_fds=True,
    )


def _prompt_yes_no(message, default=False):
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{message} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def maybe_check_for_update(force_recheck=False, skip=False):
    """
    Run early in main(). Hits the GitHub releases API on every invocation
    and prompts the user when a newer release is available. A declined
    prompt is remembered per-tag so the user is not nagged for the same
    release. force_recheck=True (from --check_updates) ignores the prior
    dismissal so the prompt appears again, but the user is still asked
    y/N — nothing is installed without explicit confirmation. skip=True
    turns the function into a no-op (used for --help/--version).
    """
    if skip or sys.platform != "win32":
        return

    if not os.path.isfile(os.path.join(_install_dir(), "tas.cmd")):
        return  # not a portable install — likely a dev checkout

    installed = _read_installed_version()
    if not installed:
        return

    try:
        release = _fetch_latest_release()
    except (urllib.error.URLError, TimeoutError, OSError):
        return

    latest_tag = (release.get("tag_name") or "").strip()
    release_url = release.get("html_url") or ""
    if not latest_tag:
        return

    latest_version = latest_tag.lstrip("vV")
    if not _is_newer(latest_version, installed):
        return

    state = _load_state()
    if not force_recheck and state.get("dismissed_for") == latest_tag:
        return

    if not sys.stdin.isatty():
        return

    print()
    print(f"[TAS] Update available: {latest_tag} (installed v{installed})")
    if release_url:
        print(f"[TAS] Release notes: {release_url}")

    if not _prompt_yes_no("[TAS] Install update now?", default=False):
        state["dismissed_for"] = latest_tag
        _save_state(state)
        print("[TAS] Skipping update. Will not ask again until a newer release.")
        return

    print(f"[TAS] Preparing update to {latest_tag}...")
    try:
        finalizer = _write_finalizer(_install_dir(), os.getpid())
        _spawn_finalizer(finalizer)
    except Exception as exc:
        print(f"[TAS] Could not start updater: {exc}")
        return

    print("[TAS] A new window will finalize the update once this process exits.")
    sys.exit(0)
