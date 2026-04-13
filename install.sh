#!/usr/bin/env bash
# Binance USDT-M Futures Trading Bot — interactive installer.
# Idempotent: safe to re-run. Use --force to overwrite an existing .env/config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

FORCE=0
AI_FIX_CLI=0
UPDATE_MODE=0
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        -a|--ai-fix) AI_FIX_CLI=1 ;;
        -u|--update) UPDATE_MODE=1 ;;
        -h|--help)
            cat <<EOF
Usage: ./install.sh [--force] [--ai-fix] [--update]

Creates a Python venv, installs dependencies, and interactively asks for:
  - Binance API key / secret
  - Telegram bot token / chat id
  - MiniMax Token Plan API key (the AI brain)
  - Default run mode (sim | live)

Writes .env (chmod 600) and config.yaml. Re-run with --force to overwrite.

--update    Pull latest code, show what changed, reinstall deps if needed.
--ai-fix    Enable AI-assisted install pipeline with step-by-step guidance.
--force     Overwrite existing .env and config.yaml.
EOF
            exit 0 ;;
    esac
done

color() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
info()  { echo "$(color '1;34' '[*]') $*"; }
warn()  { echo "$(color '1;33' '[!]') $*"; }
ok()    { echo "$(color '1;32' '[✓]') $*"; }
err()   { echo "$(color '1;31' '[x]') $*" >&2; }

# ---------- Termux detection -------------------------------------------------
# Termux on Android has its own filesystem under /data/data/com.termux, no
# sudo, no systemd, uses `pkg` as the package manager, and ships Python
# without a separate python3-venv package (venv is bundled).
IS_TERMUX=0
TERMUX_PREFIX="${PREFIX:-}"
if [[ "$TERMUX_PREFIX" == /data/data/com.termux* ]] || [ -d "/data/data/com.termux/files/usr" ]; then
    IS_TERMUX=1
    echo
    info "$(color '1;35' 'Termux environment detected')"
    echo "   If this is a fresh Termux install, run these first:"
    echo "     pkg update -y && pkg upgrade -y"
    echo "     pkg install -y python clang make rust binutils"
    echo "     pkg install -y libffi openssl pkg-config git curl"
    echo
fi

# Resolve PYTHON_BIN early so all helper functions can use it.
# The Python version check later validates the version.
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif [ "$IS_TERMUX" -eq 1 ] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"  # will fail at the version check with a clear error
fi

# ---------- Step progress + AI narration ------------------------------------
TOTAL_STEPS=9
CURRENT_STEP=0

step() {
    # Usage: step "Short title"
    # Prints a bold progress header: [Step 3/9] Short title
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo
    echo "$(color '1;36' "━━━ [Step ${CURRENT_STEP}/${TOTAL_STEPS}] $1 ━━━")"
}

# AI narration: explains the current step to the user in plain language.
# Only active when AI_FIX_ENABLED=1 and the key is available.
# Falls back silently to nothing if AI is unavailable.
ai_narrate() {
    # Args: $1 = step description for the AI
    if [ "${AI_FIX_ENABLED:-0}" -ne 1 ] || [ -z "${AI_FIX_KEY:-}" ]; then
        return 0
    fi
    local payload text
    payload=$(IS_TERMUX_ENV="$IS_TERMUX" AI_MODEL_ENV="$AI_FIX_MODEL" STEP_DESC="$1" \
              STEP_NUM="$CURRENT_STEP" TOTAL="$TOTAL_STEPS" $PYTHON_BIN - <<'PYEOF' 2>/dev/null) || return 0
import json, os
body = {
    "model": os.environ.get("AI_MODEL_ENV", "MiniMax-M2.7"),
    "max_tokens": 150,
    "system": (
        "You are a friendly install assistant guiding a user through setting up "
        "a Binance Futures grid trading bot. The user is on step "
        f"{os.environ['STEP_NUM']}/{os.environ['TOTAL']}. "
        + ("This is a Termux (Android) install. " if os.environ.get("IS_TERMUX_ENV") == "1" else "")
        + "Give a brief 1-2 sentence explanation of what this step does and why. "
        "Be encouraging. No markdown, no code blocks — plain text only."
    ),
    "messages": [{"role": "user", "content": os.environ["STEP_DESC"]}],
}
print(json.dumps(body))
PYEOF
    text=$(_minimax_call "$payload" 2>/dev/null) || return 0
    # Print the AI narration indented
    echo
    echo "   $(color '0;37' "$text")"
    echo
}

# ---------- Update mode (--update) ------------------------------------------
do_update() {
    echo
    echo "$(color '1;36' '━━━ Updating trading bot ━━━')"
    echo

    if [ ! -d ".git" ]; then
        err "Not a git repository. Clone it first:"
        err "  git clone https://github.com/MuTheBug/trading_bot.git"
        exit 1
    fi

    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    info "Current branch: $current_branch"

    # Check for local changes
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        warn "You have uncommitted local changes. Stashing them..."
        git stash push -m "install.sh --update auto-stash" 2>/dev/null || true
    fi

    # Fetch and show what's new
    info "Fetching latest changes..."
    local fetch_ok=0
    for attempt in 1 2 3 4; do
        if git fetch origin "$current_branch" 2>/dev/null; then
            fetch_ok=1
            break
        fi
        local delay=$((2 ** attempt))
        warn "Fetch failed (attempt $attempt/4), retrying in ${delay}s..."
        sleep "$delay"
    done

    if [ "$fetch_ok" -eq 0 ]; then
        err "Could not fetch from remote after 4 attempts."
        exit 1
    fi

    # Show changelog
    local ahead behind
    ahead=$(git rev-list --count "origin/$current_branch..HEAD" 2>/dev/null || echo 0)
    behind=$(git rev-list --count "HEAD..origin/$current_branch" 2>/dev/null || echo 0)

    if [ "$behind" -eq 0 ]; then
        ok "Already up to date! No new changes."
        echo
        return 0
    fi

    info "$behind new commit(s) available:"
    echo
    git --no-pager log --oneline --no-decorate "HEAD..origin/$current_branch" 2>/dev/null | \
        head -20 | while IFS= read -r line; do
            echo "   $(color '0;33' '•') $line"
        done
    echo

    # Show changed files summary
    info "Files changed:"
    git --no-pager diff --stat "HEAD..origin/$current_branch" 2>/dev/null | \
        head -30 | while IFS= read -r line; do
            echo "   $line"
        done
    echo

    # Check if requirements.txt changed
    local reqs_changed=0
    if git diff --name-only "HEAD..origin/$current_branch" 2>/dev/null | grep -q "requirements.txt"; then
        reqs_changed=1
        warn "requirements.txt changed — dependencies will be reinstalled"
    fi

    # Pull
    info "Pulling changes..."
    if ! git pull origin "$current_branch" 2>/dev/null; then
        err "Pull failed. You may need to resolve conflicts manually."
        exit 1
    fi
    ok "Code updated to latest version"

    # Reinstall deps if requirements changed
    if [ "$reqs_changed" -eq 1 ]; then
        info "Reinstalling dependencies..."
        if [ -d ".venv" ]; then
            # shellcheck disable=SC1091
            source .venv/bin/activate
            pip install --prefer-binary -r requirements.txt 2>&1 | tail -5
            ok "Dependencies updated"
        else
            warn ".venv not found — run ./install.sh without --update first"
        fi
    fi

    # Verify imports
    info "Verifying imports..."
    if [ -d ".venv" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    if $PYTHON_BIN -c "import src.config, src.bot, src.exchange.simulator, src.strategy.ai_strategy" 2>/dev/null; then
        ok "imports OK"
    else
        warn "import check failed after update — run full ./install.sh to fix"
    fi

    # AI summary of the update
    if [ "${AI_FIX_ENABLED:-0}" -eq 1 ] && [ -n "${AI_FIX_KEY:-}" ]; then
        local changelog
        changelog=$(git --no-pager log --oneline --no-decorate "HEAD~${behind}..HEAD" 2>/dev/null | head -10)
        local payload text
        payload=$(CHANGELOG="$changelog" AI_MODEL_ENV="$AI_FIX_MODEL" $PYTHON_BIN - <<'PYEOF' 2>/dev/null) || true
import json, os
body = {
    "model": os.environ.get("AI_MODEL_ENV", "MiniMax-M2.7"),
    "max_tokens": 200,
    "system": (
        "You are a friendly update assistant for a Binance Futures grid "
        "trading bot. Summarize the changes in 2-3 sentences. Be brief and "
        "highlight anything the user should know (new features, bug fixes, "
        "breaking changes). Plain text only, no markdown."
    ),
    "messages": [{"role": "user", "content": f"Commits pulled:\n{os.environ['CHANGELOG']}"}],
}
print(json.dumps(body))
PYEOF
        if [ -n "$payload" ]; then
            text=$(_minimax_call "$payload" 2>/dev/null) || true
            if [ -n "$text" ]; then
                echo
                echo "   $(color '1;35' 'AI Summary:') $text"
            fi
        fi
    fi

    echo
    ok "Update complete! Restart the bot to use the new version."
    echo "   ./run.sh"
    echo
    exit 0
}

# --------------------------------------------------------------------------
# AI-assisted install pipeline.
#
# When enabled, the installer has two AI touchpoints:
#
#   1. Preflight scan — before venv creation, it collects OS info (distro,
#      package manager, python version, gcc/make/rustc/cargo availability,
#      python3-venv module status) and sends it to MiniMax-M2.7, which
#      returns a minimal list of setup_commands to auto-run. This catches
#      common missing-deps issues (e.g. Debian without python3-venv, RHEL
#      without rustc for jiter) BEFORE the install tries and fails.
#
#   2. Retry loop — each failure-prone step is wrapped with `run_with_ai_fix`.
#      On failure, the error log + failing command is sent to MiniMax-M2.7,
#      which returns a list of fix_commands to auto-run, then the step is
#      retried up to AI_FIX_MAX_ATTEMPTS times.
#
# Suggested commands from the AI are auto-applied without a y/N prompt, but
# every command is printed before execution and filtered against a denylist
# of obviously destructive patterns (rm -rf /, mkfs, dd, shutdown, reboot,
# chmod 777 /, fork bombs). If any command in a plan matches the denylist,
# the whole plan is rejected and the step aborts.
#
# Uses plain `curl` to call the Anthropic-compatible /v1/messages endpoint,
# so there is no dependency on the `anthropic` SDK being installed yet.
# --------------------------------------------------------------------------
AI_FIX_ENABLED=0
AI_FIX_KEY="${ANTHROPIC_API_KEY:-}"
AI_FIX_URL="${ANTHROPIC_BASE_URL:-https://api.minimax.io/anthropic}"
AI_FIX_MODEL="${AI_MODEL:-MiniMax-M2.7}"
AI_FIX_MAX_ATTEMPTS=3

# Reject obviously destructive commands so auto-accept stays safe.
is_command_safe() {
    local cmd="$1"
    case "$cmd" in
        *"rm -rf /"*|*"rm -rf /*"*|*"rm -fr /"*|*"rm -r -f /"*) return 1 ;;
        *"rm --recursive --force /"*) return 1 ;;
        *"mkfs"*|*"mkfs."*) return 1 ;;
        *"dd if="*"of=/dev/sd"*|*"dd if="*"of=/dev/nvme"*|*"dd if="*"of=/dev/hd"*) return 1 ;;
        *"shutdown "*|*"reboot"*|*"halt "*|*"poweroff"*|*"init 0"*|*"init 6"*) return 1 ;;
        *':(){ :|:&'*) return 1 ;;  # fork bomb
        *"chmod -R 777 /"*|*"chmod 777 /"*) return 1 ;;
        *"> /dev/sd"*|*"> /dev/nvme"*|*"> /dev/hd"*) return 1 ;;
        *"chown -R"*" /"*" "*) return 1 ;;
        *"userdel"*|*"passwd -d root"*) return 1 ;;
        *"curl "*" | sh"*|*"curl "*" | bash"*|*"wget "*" | sh"*|*"wget "*" | bash"*)
            # Allow only the rustup.rs canonical installer, otherwise reject.
            case "$cmd" in
                *"https://sh.rustup.rs"*|*"rustup.rs"*) return 0 ;;
                *) return 1 ;;
            esac ;;
    esac
    return 0
}

# Low-level call to the Anthropic-compatible /v1/messages endpoint.
# Args: $1 = full JSON request body.
# Stdout: the assistant's text block on success.
# Returns non-zero on any failure (curl error, non-200, API error).
_minimax_call() {
    local body="$1"

    if [ -z "$AI_FIX_KEY" ]; then
        echo "ERR: AI_FIX_KEY not set" >&2
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERR: curl not installed; cannot call AI endpoint" >&2
        return 1
    fi

    # Sends both `x-api-key` (Anthropic-native) and `Authorization: Bearer`
    # (MiniMax-native) headers so the same code works against either proxy.
    local body_file err_file http_code curl_rc
    body_file="$(mktemp)"
    err_file="$(mktemp)"
    set +e
    http_code=$(printf '%s' "$body" | curl -sS --max-time 120 \
              -o "$body_file" \
              -w '%{http_code}' \
              "${AI_FIX_URL%/}/v1/messages" \
              -H "x-api-key: $AI_FIX_KEY" \
              -H "Authorization: Bearer $AI_FIX_KEY" \
              -H "anthropic-version: 2023-06-01" \
              -H "content-type: application/json" \
              --data-binary @- 2> "$err_file")
    curl_rc=$?
    set -e

    if [ "$curl_rc" -ne 0 ]; then
        {
            echo "ERR: curl failed (rc=$curl_rc) calling ${AI_FIX_URL%/}/v1/messages"
            head -c 500 "$err_file" 2>/dev/null
        } >&2
        rm -f "$body_file" "$err_file"
        return 1
    fi

    if [ "$http_code" != "200" ]; then
        {
            echo "ERR: HTTP $http_code from ${AI_FIX_URL%/}/v1/messages"
            echo "body: $(head -c 500 "$body_file" 2>/dev/null)"
        } >&2
        rm -f "$body_file" "$err_file"
        return 1
    fi

    RESP_FILE="$body_file" $PYTHON_BIN - <<'PYEOF'
import json, os, sys
with open(os.environ["RESP_FILE"], "r", errors="replace") as f:
    raw = f.read()
if not raw.strip():
    print("ERR: empty response body", file=sys.stderr)
    sys.exit(1)
try:
    obj = json.loads(raw)
except Exception as e:
    print(f"ERR: response not JSON: {e}", file=sys.stderr)
    print(f"raw: {raw[:400]}", file=sys.stderr)
    sys.exit(1)
if isinstance(obj, dict) and obj.get("type") == "error":
    print(f"ERR: API error: {obj.get('error', obj)}", file=sys.stderr)
    sys.exit(1)
if "error" in obj and "content" not in obj:
    print(f"ERR: API error: {obj['error']}", file=sys.stderr)
    sys.exit(1)
# Some API proxies wrap the content in "choices" (OpenAI-style) instead of
# "content" (Anthropic-style). Handle both.
content = obj.get("content") or []
if not content and "choices" in obj:
    for ch in obj.get("choices", []):
        msg = ch.get("message", {})
        if msg.get("content"):
            sys.stdout.write(msg["content"])
            sys.exit(0)
text = ""
for b in content:
    if isinstance(b, dict) and b.get("type") == "text":
        text += b.get("text", "") or ""
    elif isinstance(b, str):
        text += b
if not text.strip():
    # Dump the response keys so the user can report the real structure.
    keys = list(obj.keys()) if isinstance(obj, dict) else str(type(obj))
    content_types = [
        b.get("type", "?") if isinstance(b, dict) else type(b).__name__
        for b in content
    ] if content else []
    print(f"ERR: API returned 200 but assistant text was empty", file=sys.stderr)
    print(f"  response keys: {keys}", file=sys.stderr)
    print(f"  content block types: {content_types}", file=sys.stderr)
    print(f"  raw (first 500 chars): {raw[:500]}", file=sys.stderr)
    sys.exit(1)
sys.stdout.write(text)
PYEOF
    local rc=$?
    rm -f "$body_file" "$err_file"
    return $rc
}

# Pull an embedded JSON object out of a text blob (tolerates prose around it).
# Usage: _extract_json_obj "$text_var"
# Stdout: extracted JSON object. Returns non-zero on failure.
_extract_json_obj() {
    _EXTRACT_INPUT="$1" $PYTHON_BIN - <<'PYEOF'
import json, os, sys
text = os.environ.get("_EXTRACT_INPUT", "").strip()
if not text:
    print("ERR: empty input to _extract_json_obj", file=sys.stderr)
    sys.exit(1)
i, j = text.find("{"), text.rfind("}")
if i != -1 and j != -1 and j > i:
    text = text[i:j+1]
try:
    obj = json.loads(text)
except Exception as e:
    print(f"ERR: not valid JSON: {e}", file=sys.stderr)
    print(f"  input (first 300 chars): {text[:300]}", file=sys.stderr)
    sys.exit(1)
if not isinstance(obj, dict):
    print("ERR: extracted value is not a JSON object", file=sys.stderr)
    sys.exit(1)
print(json.dumps(obj))
PYEOF
}

# Collect a system snapshot for the preflight AI call. Pure Python stdlib.
preflight_collect() {
    $PYTHON_BIN - <<'PYEOF'
import json, os, platform, shutil, subprocess

def has(c):
    return shutil.which(c) is not None

def run(args, timeout=5):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()[:500]
    except Exception as e:
        return f"err: {e}"

os_release = {}
try:
    with open("/etc/os-release") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os_release[k] = v.strip('"').strip("'")
except OSError:
    pass

# Termux detection (Android). Termux has its own filesystem under
# /data/data/com.termux, no sudo, no /etc/os-release, and uses `pkg` (a
# wrapper around apt against the Termux repo) as its package manager.
termux_prefix = os.environ.get("PREFIX", "")
is_termux = termux_prefix.startswith("/data/data/com.termux") or os.path.isdir("/data/data/com.termux/files/usr")

if is_termux:
    pkg_mgr = "pkg"
else:
    pkg_mgr = next(
        (m for m in ("apt-get", "dnf", "yum", "apk", "pacman", "zypper", "brew", "pkg")
         if has(m)),
        None,
    )

venv_ok = False
venv_err = None
try:
    r = subprocess.run(
        ["python3", "-c", "import venv, ensurepip"],
        capture_output=True, text=True, timeout=5,
    )
    venv_ok = r.returncode == 0
    if not venv_ok:
        venv_err = (r.stderr or r.stdout).strip()[:300]
except Exception as e:
    venv_err = str(e)

info = {
    "uname": platform.platform(),
    "arch": platform.machine(),
    "distro": (
        "Termux (Android)" if is_termux
        else (os_release.get("PRETTY_NAME") or os_release.get("NAME") or "unknown")
    ),
    "distro_id": "termux" if is_termux else os_release.get("ID", "unknown"),
    "distro_id_like": os_release.get("ID_LIKE", ""),
    "version_id": os_release.get("VERSION_ID", ""),
    "is_termux": is_termux,
    "termux_prefix": termux_prefix,
    "python_version": platform.python_version(),
    "python_executable": shutil.which("python3") or "",
    "pip_version": run(["python3", "-m", "pip", "--version"]),
    "pkg_manager": pkg_mgr,
    "has_python3_venv_module": venv_ok,
    "venv_error": venv_err,
    "has_curl": has("curl"),
    "has_git": has("git"),
    "has_make": has("make"),
    "has_gcc": has("gcc"),
    "has_clang": has("clang"),
    "has_gxx": has("g++"),
    "has_rustc": has("rustc"),
    "has_cargo": has("cargo"),
    "has_libssl_headers": os.path.exists("/usr/include/openssl/ssl.h"),
    "has_libffi_headers": os.path.exists("/usr/include/ffi.h"),
    "is_root": (os.geteuid() == 0) if hasattr(os, "geteuid") else False,
    "has_sudo": has("sudo"),
}
print(json.dumps(info))
PYEOF
}

# Run the preflight: collect -> send to AI -> auto-execute suggested prep cmds.
preflight_run() {
    if [ "$AI_FIX_ENABLED" -ne 1 ] || [ -z "$AI_FIX_KEY" ]; then
        return 0
    fi

    info "Running preflight scan (OS + toolchain)..."
    local info_file
    info_file="$(mktemp)"
    if ! preflight_collect > "$info_file" 2>/dev/null; then
        warn "Preflight collection failed — skipping AI preflight"
        rm -f "$info_file"
        return 0
    fi

    # Show the scan summary to the user. Use an env var (not stdin) so the
    # `python3 - <<'PYEOF'` heredoc isn't fighting a `< file` redirect for
    # Python's stdin.
    INFO_FILE="$info_file" $PYTHON_BIN - <<'PYEOF'
import json, os
with open(os.environ["INFO_FILE"]) as f:
    d = json.load(f)
def tick(x): return "\u2713" if x else "\u2717"
lines = [
    f"   distro:       {d.get('distro','?')}  [{d.get('distro_id','?')}]",
    f"   arch:         {d.get('arch','?')}",
    f"   python:       {d.get('python_version','?')}",
    f"   pkg manager:  {d.get('pkg_manager','?')}",
    f"   termux:       {d.get('is_termux', False)}",
    f"   venv module:  {'OK' if d.get('has_python3_venv_module') else 'MISSING'}",
    f"   curl:  {tick(d.get('has_curl'))}   git: {tick(d.get('has_git'))}   sudo: {tick(d.get('has_sudo'))}",
    f"   gcc:   {tick(d.get('has_gcc'))}   make: {tick(d.get('has_make'))}   g++:  {tick(d.get('has_gxx'))}",
    f"   rustc: {tick(d.get('has_rustc'))}   cargo: {tick(d.get('has_cargo'))}",
]
print("\n".join(lines))
PYEOF

    # Build the AI request.
    local payload
    payload=$(INFO_FILE="$info_file" AI_MODEL_ENV="$AI_FIX_MODEL" $PYTHON_BIN - <<'PYEOF'
import json, os
with open(os.environ["INFO_FILE"]) as f:
    info = json.load(f)

system = (
    "You are a Linux install preflight assistant. The user is about to run "
    "a Python 3.11+ project installer that will:\n"
    "  (1) create a venv with `python3 -m venv .venv`\n"
    "  (2) upgrade pip + setuptools + wheel inside it\n"
    "  (3) pip install --prefer-binary a set of libraries including "
    "`anthropic` (which transitively pulls `jiter`, a Rust-backed JSON "
    "parser whose wheels may be missing on some arches), `python-binance`, "
    "`pandas`, `numpy`, `python-telegram-bot`, `loguru`, `pydantic`, "
    "`pydantic-settings`, `PyYAML`, `aiohttp`.\n\n"
    "Based on the system report in the user message, return the minimal "
    "sequence of shell commands needed to prepare this system so the "
    "installer will succeed without falling back to source builds. If the "
    "environment already looks ready, return setup_commands=[].\n\n"
    "Rules:\n"
    "- Use the exact package manager reported in `pkg_manager` (apt-get, "
    "dnf, yum, apk, pacman, zypper, brew, pkg).\n"
    "- If `is_termux` is true: this is Android Termux. Use `pkg install -y "
    "<name>` (NOT apt-get), do NOT use sudo (Termux runs as the user, "
    "there is no sudo and no root). Termux package names: `python` for "
    "Python, `python-pip`, `clang` for the C compiler (gcc is aliased), "
    "`make`, `binutils`, `libffi`, `openssl`, `rust` for rustc/cargo, "
    "`pkg-config`. There is no python3-venv package — the venv module is "
    "bundled with the `python` package on Termux. If venv is missing, "
    "reinstall `python` via `pkg install -y python`.\n"
    "- Otherwise: prefix commands with `sudo` unless `is_root` is true.\n"
    "- Non-interactive flags only: `apt-get -y`, `dnf -y`, "
    "`apk add --no-cache`, `pacman --noconfirm -S`, `pkg install -y`, etc.\n"
    "- Never suggest destructive commands: rm -rf /, mkfs, dd to a block "
    "device, shutdown, reboot, chmod 777 on system dirs, fork bombs, "
    "curl-pipe-sh from untrusted hosts.\n"
    "- Do NOT include `python3 -m venv` or `pip install` — the installer "
    "handles those itself.\n"
    "- Keep the list minimal (ideally 1-3 commands).\n\n"
    "Respond with a SINGLE JSON object, no markdown, no prose:\n"
    "{\n"
    '  "diagnosis": "<one sentence>",\n'
    '  "setup_commands": ["cmd1", "cmd2", ...],\n'
    '  "confidence": 0.0-1.0\n'
    "}"
)

body = {
    "model": os.environ.get("AI_MODEL_ENV", "MiniMax-M2.7"),
    "max_tokens": 1024,
    "system": system,
    "messages": [{"role": "user", "content": json.dumps(info)}],
}
print(json.dumps(body))
PYEOF
    ) || { rm -f "$info_file"; return 0; }
    rm -f "$info_file"

    info "Asking MiniMax-M2.7 for an environment prep plan..."
    local text
    if ! text=$(_minimax_call "$payload"); then
        warn "MiniMax preflight call failed — continuing without AI prep"
        return 0
    fi

    local plan
    if ! plan=$(_extract_json_obj "$text"); then
        warn "Preflight response was not valid JSON — continuing"
        return 0
    fi

    local diag cmds
    diag=$(printf '%s' "$plan" | $PYTHON_BIN -c \
        'import sys,json;print(json.load(sys.stdin).get("diagnosis",""))' \
        2>/dev/null || echo "")
    cmds=$(printf '%s' "$plan" | $PYTHON_BIN -c \
'import sys,json
obj = json.load(sys.stdin)
for c in obj.get("setup_commands", []) or []:
    c = str(c).strip()
    if c:
        print(c)' 2>/dev/null || true)

    echo
    echo "   $(color '1;36' 'Diagnosis:') ${diag:-(none)}"
    if [ -z "$cmds" ]; then
        ok "AI reports environment is ready — no preflight changes needed"
        return 0
    fi

    echo "   $(color '1;36' 'Preflight commands (auto-accept):')"
    while IFS= read -r c; do
        [ -z "$c" ] && continue
        echo "     \$ $c"
    done <<< "$cmds"

    # Safety filter: reject the whole plan if any command is denylisted.
    while IFS= read -r c; do
        [ -z "$c" ] && continue
        if ! is_command_safe "$c"; then
            err "Refusing to auto-execute unsafe command: $c"
            warn "Aborting AI preflight. Install will proceed without prep."
            return 0
        fi
    done <<< "$cmds"

    # Auto-execute.
    local any_failed=0
    while IFS= read -r pcmd; do
        [ -z "$pcmd" ] && continue
        info "Running: $pcmd"
        if ! bash -c "$pcmd"; then
            warn "Preflight command failed: $pcmd (continuing)"
            any_failed=1
        fi
    done <<< "$cmds"

    if [ "$any_failed" -eq 0 ]; then
        ok "Preflight AI prep complete"
    else
        warn "Preflight had failures — continuing and relying on retry loop"
    fi
}

ai_fix_suggest() {
    # Args: $1=step_label $2=failed_command_str $3=exit_code $4=log_file
    # Writes a JSON object {diagnosis, fix_commands, retry, confidence} to
    # stdout on success; returns non-zero on failure.
    local step="$1" cmd="$2" rc="$3" log="$4"

    local payload
    payload=$(AI_STEP="$step" AI_CMD="$cmd" AI_RC="$rc" AI_LOG_FILE="$log" \
              AI_MODEL_ENV="$AI_FIX_MODEL" IS_TERMUX="$IS_TERMUX" $PYTHON_BIN - <<'PYEOF'
import json, os, platform
log_path = os.environ["AI_LOG_FILE"]
try:
    with open(log_path, "r", errors="replace") as f:
        log_tail = f.read()[-4000:]
except OSError:
    log_tail = "(no log)"

user_msg = {
    "step": os.environ["AI_STEP"],
    "command": os.environ["AI_CMD"],
    "exit_code": int(os.environ.get("AI_RC", "1")),
    "stderr_tail": log_tail,
    "os": platform.platform(),
    "python_version": platform.python_version(),
    "cwd": os.getcwd(),
    "is_termux": os.environ.get("IS_TERMUX", "0") == "1",
    "termux_prefix": os.environ.get("PREFIX", ""),
}

system = (
    "You are an install-script repair assistant for a Python project on "
    "Linux. A step in ./install.sh just failed. Diagnose the root cause from "
    "the error log and propose a minimal sequence of shell commands that "
    "will fix it so the failed step can be retried.\n\n"
    "Rules:\n"
    "- Never suggest destructive commands (rm -rf /, mkfs, dd, shutdown, "
    "reboot, curl | sh from unknown hosts, chmod 777 on system dirs).\n"
    "- If `is_termux` is true: this is Android Termux. Use `pkg install -y "
    "<name>` (NOT apt-get), do NOT use sudo (Termux has no root). Termux "
    "package names: `python` (not python3), `python-pip`, `clang` (for C "
    "compiler), `make`, `binutils`, `libffi`, `openssl`, `rust` (for "
    "rustc/cargo), `pkg-config`. There is no python3-venv package — venv "
    "is bundled with the `python` package.\n"
    "- Otherwise: prefer apt/yum/dnf/apk install with sudo when a system "
    "dep is missing.\n"
    "- Or pip install inside the existing venv for Python packages.\n"
    "- Keep the fix minimal (1-4 commands).\n"
    "- If you cannot confidently fix it, return fix_commands=[] and "
    "retry=false.\n\n"
    "Respond with a SINGLE JSON object and nothing else — no markdown, no "
    "prose:\n"
    "{\n"
    '  "diagnosis": "<one sentence>",\n'
    '  "fix_commands": ["cmd1", "cmd2", ...],\n'
    '  "retry": true | false,\n'
    '  "confidence": 0.0-1.0\n'
    "}"
)

body = {
    "model": os.environ.get("AI_MODEL_ENV", "MiniMax-M2.7"),
    "max_tokens": 1024,
    "system": system,
    "messages": [{"role": "user", "content": json.dumps(user_msg)}],
}
print(json.dumps(body))
PYEOF
    ) || return 1

    local text
    if ! text=$(_minimax_call "$payload"); then
        return 1
    fi
    _extract_json_obj "$text"
}

run_with_ai_fix() {
    # Usage: run_with_ai_fix "step label" cmd arg1 arg2 ...
    # Runs the command; on failure, if AI auto-fix is enabled, asks MiniMax
    # for a fix, shows it to the user, optionally applies it, and retries.
    local step="$1"; shift
    local cmd_str="$*"
    local attempt=0
    while : ; do
        local log rc
        log="$(mktemp)"
        rc=0
        if "$@" 2>&1 | tee "$log"; then
            rm -f "$log"
            return 0
        else
            rc=${PIPESTATUS[0]:-1}
            [ "$rc" -eq 0 ] && rc=1
        fi

        if [ "$AI_FIX_ENABLED" -ne 1 ] || [ -z "$AI_FIX_KEY" ]; then
            rm -f "$log"
            return "$rc"
        fi

        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$AI_FIX_MAX_ATTEMPTS" ]; then
            err "AI auto-fix gave up after ${AI_FIX_MAX_ATTEMPTS} attempts."
            rm -f "$log"
            return "$rc"
        fi

        warn "Step '$step' failed (exit $rc). Asking MiniMax-M2.7 for a fix (${attempt}/${AI_FIX_MAX_ATTEMPTS})..."
        local suggestion
        if ! suggestion=$(ai_fix_suggest "$step" "$cmd_str" "$rc" "$log"); then
            err "Could not get an AI suggestion — aborting this step."
            rm -f "$log"
            return "$rc"
        fi
        rm -f "$log"

        local diag retry cmds
        diag=$(printf '%s' "$suggestion" | $PYTHON_BIN -c \
            'import sys,json;print(json.load(sys.stdin).get("diagnosis",""))' \
            2>/dev/null || echo "")
        retry=$(printf '%s' "$suggestion" | $PYTHON_BIN -c \
            'import sys,json;print("yes" if json.load(sys.stdin).get("retry", True) else "no")' \
            2>/dev/null || echo "yes")
        cmds=$(printf '%s' "$suggestion" | $PYTHON_BIN -c \
'import sys,json
obj = json.load(sys.stdin)
for c in obj.get("fix_commands", []) or []:
    c = str(c).strip()
    if c:
        print(c)' 2>/dev/null || true)

        echo
        echo "   $(color '1;36' 'Diagnosis:') ${diag:-(none)}"
        if [ -z "$cmds" ]; then
            warn "AI had no fix to suggest — aborting this step."
            return "$rc"
        fi
        echo "   $(color '1;36' 'Fix commands (auto-accept):')"
        while IFS= read -r c; do
            [ -z "$c" ] && continue
            echo "     \$ $c"
        done <<< "$cmds"

        # Safety filter: reject the whole plan if any command is denylisted.
        local unsafe=0
        while IFS= read -r c; do
            [ -z "$c" ] && continue
            if ! is_command_safe "$c"; then
                err "Refusing to auto-execute unsafe command: $c"
                unsafe=1
                break
            fi
        done <<< "$cmds"
        if [ "$unsafe" -eq 1 ]; then
            warn "Aborting this step."
            return "$rc"
        fi

        while IFS= read -r fix_cmd; do
            [ -z "$fix_cmd" ] && continue
            info "Running: $fix_cmd"
            if ! bash -c "$fix_cmd"; then
                warn "Fix command failed: $fix_cmd (will ask AI again on retry)"
            fi
        done <<< "$cmds"

        if [ "$retry" = "no" ]; then
            info "AI marked retry=false; not re-running this step."
            return 0
        fi
        info "Retrying step: $step"
    done
}

# ---------- Run update mode if requested (after all functions are defined) ---
if [ "$UPDATE_MODE" -eq 1 ]; then
    # Load AI key from .env if available for update summary
    if [ -z "$AI_FIX_KEY" ] && [ -f ".env" ]; then
        AI_FIX_KEY="${ANTHROPIC_API_KEY:-$(grep -oP 'ANTHROPIC_API_KEY=\K.*' .env 2>/dev/null || true)}"
    fi
    if [ -n "$AI_FIX_KEY" ]; then
        AI_FIX_ENABLED=1
    fi
    do_update
fi

# ===========================================================================
# FULL INSTALL FLOW
# ===========================================================================

# --- 1. AI-Guided Install (ask for key FIRST so all steps get AI help) ------
step "AI-Guided Install"
echo
info "$(color '1;35' 'This installer uses AI to guide you and auto-fix errors.')"
echo "   Every step is monitored. If something fails, MiniMax-M2.7 will"
echo "   diagnose the error and apply the fix automatically."
echo

# Try to load key from environment or existing .env
if [ -z "$AI_FIX_KEY" ] && [ -f ".env" ]; then
    AI_FIX_KEY="$(grep -oP 'ANTHROPIC_API_KEY=\K.*' .env 2>/dev/null || true)"
fi

if [ -n "$AI_FIX_KEY" ]; then
    AI_FIX_ENABLED=1
    ok "AI key loaded (model: $AI_FIX_MODEL)"
else
    echo "   You need a MiniMax API key for AI-guided install."
    echo "   Get one at: $(color '1;36' 'https://platform.minimax.io/user-center/basic-information/interface-key')"
    echo
    read -rsp "   MiniMax API Key: " AI_FIX_KEY
    echo
    if [ -n "$AI_FIX_KEY" ]; then
        AI_FIX_ENABLED=1
        ok "AI guide enabled (model: $AI_FIX_MODEL)"
    else
        warn "No key — installing without AI guidance (errors won't be auto-fixed)"
    fi
fi

# --- 2. Python version check ------------------------------------------------
step "Checking Python"
ai_narrate "Checking that Python 3.11+ is installed and working"
info "Checking Python..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if [ "$IS_TERMUX" -eq 1 ]; then
        err "python not found. Install it with: pkg install -y python"
    else
        err "python3 not found. Install Python 3.11+ and re-run."
    fi
    exit 1
fi
if [ "$IS_TERMUX" -eq 1 ] && [ "$PYTHON_BIN" = "python" ]; then
    info "Termux detected — using 'python' instead of 'python3'"
fi

PY_VER=$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_MAJOR=$($PYTHON_BIN -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$($PYTHON_BIN -c 'import sys; print(sys.version_info[1])')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    if [ "$IS_TERMUX" -eq 1 ]; then
        err "Python >= 3.11 required (found $PY_VER)."
        err "Update with: pkg upgrade -y python"
    else
        err "Python >= 3.11 required (found $PY_VER)."
    fi
    exit 1
fi
ok "Python $PY_VER ($([ "$IS_TERMUX" -eq 1 ] && echo 'Termux' || echo 'system'))"

# --- 3. Preflight AI environment prep -------------------------------------
step "Preflight Environment Scan"
ai_narrate "Scanning your system for missing build tools and libraries before we start installing"
preflight_run

# --- 4. venv ----------------------------------------------------------------
step "Creating Virtual Environment"
ai_narrate "Setting up an isolated Python environment so the bot's dependencies don't conflict with your system packages"
if [ ! -d ".venv" ]; then
    info "Creating virtualenv at .venv ..."
    if ! run_with_ai_fix "create virtualenv" $PYTHON_BIN -m venv .venv; then
        if [ "$IS_TERMUX" -eq 1 ]; then
            err "Could not create venv. Try:"
            err "  pkg install -y python"
            err "  (venv is bundled with the Termux python package)"
        else
            err "Could not create venv. On Debian/Ubuntu try:"
            err "  sudo apt install python3-venv"
        fi
        exit 1
    fi
    ok "venv created"
else
    ok "venv exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

step "Installing Dependencies"
ai_narrate "Installing all Python packages the bot needs: Binance API client, pandas for data, telegram for alerts, AI SDK, and more. This is the longest step."
info "Upgrading pip / setuptools / wheel..."
if ! run_with_ai_fix "upgrade pip toolchain" pip install --upgrade pip setuptools wheel; then
    err "pip toolchain upgrade failed."
    exit 1
fi
info "Installing dependencies (this may take a minute)..."
# --prefer-binary makes pip pick prebuilt wheels when available, avoiding
# source builds for Rust-backed packages like `jiter` (an anthropic
# transitive dep) on platforms where Rust/maturin isn't installed.
if ! run_with_ai_fix "install requirements" pip install --prefer-binary -r requirements.txt; then
    err "Dependency install failed. See the log above."
    exit 1
fi
ok "dependencies installed"

# --- 6. config.yaml ---------------------------------------------------------
step "Configuration Files"
ai_narrate "Setting up config.yaml with default trading parameters and preparing .env for your API credentials"
if [ -f "config.yaml" ] && [ "$FORCE" -eq 0 ]; then
    ok "config.yaml already present (use --force to overwrite)"
else
    cp config.example.yaml config.yaml
    ok "config.yaml created from config.example.yaml"
fi

# --- 7. .env prompt ---------------------------------------------------------
step "API Credentials"
ai_narrate "Now we need your Binance, Telegram, and MiniMax API keys. These are stored locally in .env with restricted permissions — never shared."
if [ -f ".env" ] && [ "$FORCE" -eq 0 ]; then
    ok ".env already present (use --force to re-prompt)"
else
    echo
    info "Let's set up your credentials."
    echo "   (Press Enter to leave a field blank and fill it in later by editing .env.)"
    echo

    read -rp "   Binance API Key: " BINANCE_API_KEY
    read -rsp "   Binance API Secret: " BINANCE_API_SECRET
    echo
    read -rp "   Use Binance Futures TESTNET? [y/N]: " USE_TESTNET
    TESTNET_VAL="false"
    case "${USE_TESTNET:-}" in
        y|Y|yes|YES) TESTNET_VAL="true" ;;
    esac

    read -rsp "   Telegram Bot Token (from @BotFather): " TG_TOKEN
    echo
    read -rp "   Telegram Chat ID (numeric, from @userinfobot): " TG_CHAT

    echo
    info "AI brain — MiniMax Token Plan (the bot uses M2.7 to decide trades)"
    echo "   Get a key at https://platform.minimax.io/user-center/basic-information/interface-key"
    if [ -n "$AI_FIX_KEY" ]; then
        echo "   (Press Enter to reuse the key you supplied for AI auto-fix.)"
        read -rsp "   MiniMax API Key: " MINIMAX_KEY_IN
        echo
        MINIMAX_KEY="${MINIMAX_KEY_IN:-$AI_FIX_KEY}"
    else
        read -rsp "   MiniMax API Key: " MINIMAX_KEY
        echo
    fi
    MINIMAX_URL="${AI_FIX_URL:-https://api.minimax.io/anthropic}"
    read -rp "   MiniMax base URL (default ${MINIMAX_URL}): " MINIMAX_URL_IN
    if [ -n "${MINIMAX_URL_IN:-}" ]; then
        MINIMAX_URL="${MINIMAX_URL_IN}"
    fi

    DEFAULT_MODE="sim"
    read -rp "   Default run mode [sim/live] (default sim): " MODE_IN
    case "${MODE_IN:-}" in
        live|LIVE) DEFAULT_MODE="live" ;;
        *) DEFAULT_MODE="sim" ;;
    esac

    umask 077
    cat > .env <<EOF
# Auto-generated by install.sh. Edit as needed.
BINANCE_API_KEY=${BINANCE_API_KEY}
BINANCE_API_SECRET=${BINANCE_API_SECRET}
BINANCE_TESTNET=${TESTNET_VAL}

TELEGRAM_BOT_TOKEN=${TG_TOKEN}
TELEGRAM_CHAT_ID=${TG_CHAT}

ANTHROPIC_API_KEY=${MINIMAX_KEY}
ANTHROPIC_BASE_URL=${MINIMAX_URL}

RUN_MODE=${DEFAULT_MODE}
EOF
    chmod 600 .env
    ok ".env written (chmod 600)"
fi

# --- make run.sh executable + runtime dirs ----------------------------------
chmod +x run.sh 2>/dev/null || true
mkdir -p logs state
ok "logs/ and state/ directories ready"

# --- 8. smoke-check imports ------------------------------------------------
step "Verifying Installation"
ai_narrate "Almost done! Checking that all Python modules import correctly and testing connectivity to your configured services."
info "Verifying imports..."
if $PYTHON_BIN -c "import src.config, src.bot, src.exchange.simulator, src.strategy.ai_strategy" 2>/dev/null; then
    ok "imports OK"
else
    warn "import check failed — run '$PYTHON_BIN -c \"import src.bot\"' to debug"
fi

# --- 9. optional connectivity check ----------------------------------------
step "Connectivity Check"
if [ -f ".env" ]; then
    read -rp "   Run a connectivity check against Binance + Telegram + MiniMax now? [Y/n]: " CONN
    case "${CONN:-Y}" in
        n|N|no|NO) ;;
        *)
            # shellcheck disable=SC1091
            set -a; source .env; set +a
            info "Pinging Binance Futures..."
            $PYTHON_BIN - <<'PYEOF' || warn "Binance ping failed"
import asyncio, os
try:
    from binance import AsyncClient
except ImportError:
    print("python-binance not installed"); raise SystemExit(1)
async def main():
    key = os.environ.get("BINANCE_API_KEY", "")
    sec = os.environ.get("BINANCE_API_SECRET", "")
    tn  = os.environ.get("BINANCE_TESTNET", "false").lower() == "true"
    if not key or not sec:
        print("(no key set, skipping auth ping)")
        c = await AsyncClient.create(testnet=tn)
    else:
        c = await AsyncClient.create(api_key=key, api_secret=sec, testnet=tn)
    try:
        await c.futures_ping()
        print("binance: OK")
    finally:
        await c.close_connection()
asyncio.run(main())
PYEOF

            if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
                info "Checking Telegram token..."
                if command -v curl >/dev/null 2>&1; then
                    TG_OUT=$(curl -s --max-time 10 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" || true)
                    case "$TG_OUT" in
                        *'"ok":true'*) ok "telegram: OK" ;;
                        *) warn "telegram: bad token or network — raw: ${TG_OUT:0:200}" ;;
                    esac
                fi
            fi

            if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
                info "Checking MiniMax (Anthropic-compatible) API..."
                $PYTHON_BIN - <<'PYEOF' || warn "MiniMax API check failed"
import os
try:
    from anthropic import Anthropic
except ImportError:
    print("anthropic SDK not installed"); raise SystemExit(1)
c = Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic"),
)
msg = c.messages.create(
    model=os.environ.get("AI_MODEL", "MiniMax-M2.7"),
    max_tokens=32,
    messages=[{"role": "user", "content": "Reply with the single word OK."}],
)
text = ""
for b in msg.content:
    if getattr(b, "type", None) == "text":
        text += getattr(b, "text", "")
print(f"minimax: {text.strip()[:40] or '(empty)'}")
PYEOF
            fi
            ;;
    esac
fi

if [ "$IS_TERMUX" -eq 1 ]; then
    cat <<'EOF'

──────────────────────────────────────────────
 ✅ Install complete (Termux).

 Next steps:

   ./run.sh                # runs in mode from .env (defaults to sim)
   ./run.sh --mode sim     # force simulation (paper trading, no real orders)
   ./run.sh --mode live    # real orders on Binance Futures

 Termux tips:
   • Acquire a wakelock so Android doesn't kill the bot:
       termux-wake-lock
   • Run in background with tmux or nohup:
       pkg install -y tmux
       tmux new -s bot './run.sh'
       (Ctrl+B, D to detach — tmux attach -t bot to reconnect)
   • Or use nohup:
       nohup ./run.sh > bot.log 2>&1 &
   • To stop: kill %1  or  tmux kill-session -t bot
   • If pip install fails with build errors, try:
       pkg install -y clang make rust binutils libffi openssl pkg-config
       then re-run ./install.sh

 Edit strategy parameters:  config.yaml
 Edit credentials:          .env
──────────────────────────────────────────────
EOF
else
    cat <<'EOF'

──────────────────────────────────────────────
 ✅ Install complete.

 Next steps:

   ./run.sh                # runs in mode from .env (defaults to sim)
   ./run.sh --mode sim     # force simulation (paper trading, no real orders)
   ./run.sh --mode live    # real orders on Binance Futures — READ THE DOCS FIRST

 Edit strategy parameters:  config.yaml
 Edit credentials:          .env
──────────────────────────────────────────────
EOF
fi
