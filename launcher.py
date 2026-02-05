import os, sys, time, socket, webbrowser, traceback
from pathlib import Path

HOST = "127.0.0.1"
MUTEX_NAME = "Global\\TGECase_SingleInstance"
PORT_FILE = "tgecase_port.txt"
LOG_FILE = "tgecase.log"

def acquire_mutex() -> bool:
    try:
        import ctypes
        from ctypes import wintypes
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

def find_free_port() -> int:
    s = socket.socket()
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port

def resolve_total_py() -> Path:
    exe_dir = Path(sys.argv[0]).resolve().parent
    candidates = [
        exe_dir / "_internal" / "app" / "Total.py",
        exe_dir / "app" / "Total.py",
        exe_dir / "optimize" / "Total.py",  # dev fallback
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Total.py not found (checked _internal/app, app, optimize).")

def read_port(exe_dir: Path) -> int | None:
    p = exe_dir / PORT_FILE
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def write_port(exe_dir: Path, port: int):
    (exe_dir / PORT_FILE).write_text(str(port), encoding="utf-8")

def main():
    exe_dir = Path(sys.argv[0]).resolve().parent
    os.chdir(exe_dir)

    log_path = exe_dir / LOG_FILE
    f = open(log_path, "a", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f

    print("\n=== TGECase launch ===")
    print("Exe dir:", exe_dir)

    if not acquire_mutex():
        port = read_port(exe_dir) or 8501
        webbrowser.open(f"http://{HOST}:{port}")
        print("Another instance detected. Opened browser and exiting.")
        f.flush()
        return

    total_py = resolve_total_py()
    port = find_free_port()
    write_port(exe_dir, port)
    url = f"http://{HOST}:{port}"

    print("Total.py:", total_py)
    print("Chosen port:", port)
    f.flush()

    # Streamlit config override
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    # Browser'ı biraz gecikmeyle aç (server ayağa kalksın diye)
    import threading
    def open_later():
        time.sleep(1.2)
        webbrowser.open(url)
    threading.Thread(target=open_later, daemon=True).start()

    # --- Fix for frozen apps: ensure `help()` exists (some libs like gurobipy expect it) ---
    try:
        import site  # this normally defines builtins.help, quit, exit, etc.
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



    try:
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
        stcli.main()  # <-- MAIN THREAD'DE ÇALIŞIYOR
    except SystemExit:
        print("Streamlit exited (SystemExit).")
    except Exception:
        print("STREAMLIT CRASH:\n", traceback.format_exc())
    finally:
        try:
            (exe_dir / PORT_FILE).unlink(missing_ok=True)
        except Exception:
            pass
        f.flush()

if __name__ == "__main__":
    main()
