# TGE Case – Packaging Kit (Windows .exe + macOS .app via GitHub Actions)

This kit assumes your main Streamlit app is **`optimize/Total.py`** (it contains Puzzle Mode + Scenario Events UI).

## Why this approach works on huge repos
PyInstaller often breaks on large Streamlit codebases because of hidden imports.
So we **do not** freeze your whole app as Python modules. Instead we:

- bundle your whole `optimize/` folder as plain source into `dist/TGECase/app/`
- build only a small `launcher.exe`
- launcher runs: `streamlit run app/Total.py`

That makes packaging much more robust.

---

## Windows: build .exe locally

1) Copy these files into your repo root (same level as `optimize/`):
- `launcher.py`
- `build_windows.ps1` (or `build_windows.bat`)

2) Run:
```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

3) Output:
- `dist\TGECase\TGECase.exe`
- Logs: `dist\TGECase\tgecase.log`

---

## macOS: build .app without owning a Mac (GitHub Actions)

1) Create this file in your repo:
- `.github/workflows/build.yml`  (copy kit's `build.yml` there)

2) Push your repo to GitHub.

3) Trigger a build:
- Actions → **build-executables** → **Run workflow**
- Or create a tag like `v1.0.0` and push it.

4) Download artifacts from the workflow run:
- `TGECase-Windows.zip`
- `TGECase-macOS.zip` (or `TGECase-macOS-folder.zip`)

---

## Notes
- PyInstaller `--collect-all` is used to include Streamlit/Plotly/Altair/PyDeck assets and submodules. citeturn0search7
- The `--add-data` separator differs by OS: `;` on Windows, `:` on macOS/Linux. citeturn0search17turn0search4
- If you build the macOS app for sharing broadly, Gatekeeper may warn unless you sign/notarize.
