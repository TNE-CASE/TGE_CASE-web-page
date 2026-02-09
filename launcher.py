import os
import sys
import time
import socket
import webbrowser
import traceback
from pathlib import Path

HOST = "127.0.0.1"
APP_NAME = "TGECase"
PORT_FILE = "tgecase_port.txt"
LOG_FILE = "tgecase.log"
LOCK_FILE = "tgecase.lock"  # non-windows single-instance

# -----------------------
# Paths / AppData helpers
# -----------------------
def get_app_data_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    p = base / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p

def runtime_roots() -> list[Path]:
    roots: list[Path] = []

    # PyInstaller runtime extraction dir (often most reliable)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))

    # Executable dir
    exe_path = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve()
    exe_dir = exe_path.parent
    roots.append(exe_dir)

    # _internal next to exe
    roots.append(exe_dir / "_internal")

    # macOS bundle: Contents/MacOS -> Contents/Resources
    # exe_dir = .../TGECase.app/Contents/MacOS
    if exe_dir.name == "MacOS" and exe_dir.parent.name == "Contents":
        resources = exe_dir.parent / "Resources"
        roots.append(resources)
        roots.append(resources / "_internal")

    # de-dup + existing only
    uniq = []
    seen = set()
    for r in roots:
        rp = str(r)
        if rp in seen:
            continue
        seen.add(rp)
        if r and r.exists():
            uniq.append(r)
    return uniq

def pick_runtime_root() -> Path:
    roots = runtime_roots()

    # prefer root that contains app/Total.py or app folder
    for r in roots:
        if (r / "app" / "Total.py").exists() or (r / "app").exists():
            return r
    return roots[0] if roots else Path.cwd()

# -----------------------
# Single instance
# -----------------------
def acquire_single_instance_lock(app_data: Path) -> bool:
    if os.name == "nt":
        # Windows mutex (best-effort)
        try:
            import ctypes
            from ctypes import wintypes
            MUTEX_NAME = "Global\\TGECase_SingleInstance"
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            CreateMutexW = k.CreateMutexW
            CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
            CreateMutexW.restype = wintypes.HANDLE
            GetLastError = k.GetLastError
            ERROR_ALREADY_EXISTS = 183
            h = CreateMutexW(None, False, MUTEX_NAME)
            if not h:
                return True
            return GetLastError() != ERROR_ALREADY_EXISTS
        except Exception:
            return True

    # mac/linux: simple lock file
    lock_path = app_data / LOCK_FILE
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return True

def release_lock(app_data: Path):
    if os.name == "nt":
        return
    try:
        (app_data / LOCK_FILE).unlink(missing_ok=True)
    except Exception:
        pass

# -----------------------
# Ports / resolve Total.py
# -----------------------
def find_free_port() -> int:
    s = socket.socket()
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port

def read_port(app_data: Path) -> int | None:
    p = app_data / PORT_FILE
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def write_port(app_data: Path, port: int):
    (app_data / PORT_FILE).write_text(str(port), encoding="utf-8")

def resolve_total_py(rt: Path) -> Path:
    candidates = [
        rt / "app" / "Total.py",
        rt / "_internal" / "app" / "Total.py",   # some layouts
        rt / "optimize" / "Total.py",            # dev fallback
        rt / "Total.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    # last-chance: search a bit
    for base in runtime_roots():
        for rel in [Path("app/Total.py"), Path("_internal/app/Total.py"), Path("optimize/Total.py")]:
            p = base / rel
            if p.exists():
                return p.resolve()

    raise FileNotFoundError(
        f"Total.py not found. runtime_root={rt}. Checked typical locations under: "
        + ", ".join(str(r) for r in runtime_roots())
    )

# -----------------------
# Main
# -----------------------
def main():
    app_data = get_app_data_dir()
    rt = pick_runtime_root()
    os.chdir(rt)

    log_path = app_data / LOG_FILE
    f = open(log_path, "a", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f

    print("\n=== TGECase launch ===")
    print("Runtime root:", rt)
    print("App data dir:", app_data)

    if not acquire_single_instance_lock(app_data):
        port = read_port(app_data) or 8501
        url = f"http://{HOST}:{port}"
        print("Another instance detected -> opening:", url)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        f.flush()
        return

    try:
        total_py = resolve_total_py(rt)
        port = find_free_port()
        write_port(app_data, port)
        url = f"http://{HOST}:{port}"

        print("Total.py:", total_py)
        print("Chosen port:", port)

        # Streamlit config overrides
        os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

        # Ensure help()/quit()/exit exist (some libs expect these)
        try:
            import site  # noqa: F401
        except Exception:
            pass
        try:
            import builtins, pydoc
            if not hasattr(builtins, "help"):
                builtins.help = pydoc.help
            if not hasattr(builtins, "quit"):
                builtins.quit = lambda *a, **k: None
            if not hasattr(builtins, "exit"):
                builtins.exit = lambda *a, **k: None
        except Exception:
            pass

        # open browser a bit later
        import threading
        def open_later():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=open_later, daemon=True).start()

        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit", "run", str(total_py),
            "--server.headless=true",
            f"--server.address={HOST}",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
            "--global.developmentMode=false",
        ]
        print("Starting Streamlit:", " ".join(sys.argv))
        f.flush()
        stcli.main()

    except SystemExit:
        print("Streamlit exited (SystemExit).")
    except Exception:
        print("STREAMLIT CRASH:\n", traceback.format_exc())
    finally:
        try:
            (app_data / PORT_FILE).unlink(missing_ok=True)
        except Exception:
            pass
        release_lock(app_data)
        f.flush()

if __name__ == "__main__":
    main()
