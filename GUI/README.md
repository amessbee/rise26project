# U.S. Infrastructure Stress Monitor — Local Package

This rebuild is designed to run reliably on Windows without a virtual environment,
FastAPI, pandas or any `pip install`.

The website reads precomputed deterministic projections generated from the supplied
frozen project history and original projection functions. The original model files,
metadata and training scripts are retained in `models/`.

## Start on Windows by double-clicking

1. Extract the ZIP.
2. Open the extracted `stress_monitor_local_complete` folder.
3. Double-click `START_LOCALHOST.bat`.
4. Your browser should open automatically.
5. Otherwise open: `http://127.0.0.1:8000`

Keep the command window open while using the site. Press `Ctrl+C` to stop it.

## Start from Git Bash

```bash
cd /d/RISE_Project/stress_monitor_local_complete
./start_git_bash.sh
```

Then open:

```text
http://127.0.0.1:8000
```

## Start manually

```bash
py server.py
```

or:

```bash
python server.py
```

No virtual environment is required.

## Local endpoints

- Website: `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/healthz`
- Example API output: `http://127.0.0.1:8000/api/project?state=CA&year=2030`
- Bootstrap metadata: `http://127.0.0.1:8000/api/bootstrap`

## If port 8000 is busy

Windows Command Prompt:

```bat
set PORT=8080
py server.py
```

Git Bash:

```bash
PORT=8080 py server.py
```

Then open `http://127.0.0.1:8080`.

## Important interpretation

Years after 2024 are damped recursive baseline scenarios. They are not direct
forecasts with known future weather, population, demand or infrastructure. Displayed
ranges are approximate and based on the supplied backtest RMSE values.

## Package structure

```text
index.html
server.py
START_LOCALHOST.bat
start_git_bash.sh
static/
  app.js
  data.js
  styles.css
data/
  projections.json
models/
  original frozen model files and project metadata
```


## Corrected model diagrams

The `HOW EACH FINAL MODEL WORKS` section now uses four hand-built SVG diagrams
in the exact 2x2 order from the supplied reference:

```text
SAIDI       SAIFI
Drought     Compliance
```

The recreated vector assets are stored in `static/diagrams/`.
