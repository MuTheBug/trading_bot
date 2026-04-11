#!/usr/bin/env bash
# Binance USDT-M Futures Trading Bot — interactive installer.
# Idempotent: safe to re-run. Use --force to overwrite an existing .env/config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

FORCE=0
AI_FIX_CLI=0
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        -a|--ai-fix) AI_FIX_CLI=1 ;;
        -h|--help)
            cat <<EOF
Usage: ./install.sh [--force] [--ai-fix]

Creates a Python venv, installs dependencies, and interactively asks for:
  - Binance API key / secret
  - Telegram bot token / chat id
  - MiniMax Token Plan API key (the AI brain)
  - Default run mode (sim | live)

Writes .env (chmod 600) and config.yaml. Re-run with --force to overwrite.

--ai-fix    Enable AI-assisted error recovery without prompting. When a
            failure-prone step (venv, pip install) fails, the error log is
            sent to MiniMax-M2.7 which proposes shell commands to fix the
            issue. Every command is shown before it runs; nothing executes
            without your approval. Reads the key from ANTHROPIC_API_KEY if
            set, otherwise prompts you.
EOF
            exit 0 ;;
    esac
done

color() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
info()  { echo "$(color '1;34' '[*]') $*"; }
warn()  { echo "$(color '1;33' '[!]') $*"; }
ok()    { echo "$(color '1;32' '[✓]') $*"; }
err()   { echo "$(color '1;31' '[x]') $*" >&2; }

# --------------------------------------------------------------------------
# AI-assisted error recovery.
#
# When enabled, failure-prone steps are wrapped with `run_with_ai_fix`. On
# failure, the last ~4000 bytes of the log plus the failing command are sent
# to MiniMax-M2.7 via the Anthropic-compatible /v1/messages endpoint using
# plain curl (so we don't depend on the `anthropic` SDK being installed yet).
# The model returns a JSON object with a diagnosis and a list of shell
# commands to run. Nothing runs without the user's explicit approval.
# --------------------------------------------------------------------------
AI_FIX_ENABLED=0
AI_FIX_KEY="${ANTHROPIC_API_KEY:-}"
AI_FIX_URL="${ANTHROPIC_BASE_URL:-https://api.minimax.io/anthropic}"
AI_FIX_MODEL="${AI_MODEL:-MiniMax-M2.7}"
AI_FIX_MAX_ATTEMPTS=3

ai_fix_suggest() {
    # Args: $1=step_label $2=failed_command_str $3=exit_code $4=log_file
    # Writes a JSON object {diagnosis, fix_commands, retry, confidence} to
    # stdout on success; returns non-zero on failure.
    local step="$1" cmd="$2" rc="$3" log="$4"

    if ! command -v curl >/dev/null 2>&1; then
        echo "ERR: curl not installed; cannot call AI fix endpoint" >&2
        return 1
    fi

    local payload
    payload=$(AI_STEP="$step" AI_CMD="$cmd" AI_RC="$rc" AI_LOG_FILE="$log" \
              AI_MODEL_ENV="$AI_FIX_MODEL" python3 - <<'PYEOF'
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
}

system = (
    "You are an install-script repair assistant for a Python project on "
    "Linux. A step in ./install.sh just failed. Diagnose the root cause from "
    "the error log and propose a minimal sequence of shell commands that "
    "will fix it so the failed step can be retried.\n\n"
    "Rules:\n"
    "- Never suggest destructive commands (rm -rf /, mkfs, dd, shutdown, "
    "reboot, curl | sh from unknown hosts, chmod 777 on system dirs).\n"
    "- Prefer apt/yum/dnf/apk install with sudo when a system dep is "
    "missing, or pip install inside the existing venv for Python packages.\n"
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

    # Call the Anthropic-compatible /v1/messages endpoint. We send both
    # `x-api-key` (Anthropic-native) and `Authorization: Bearer` (MiniMax-
    # native) headers so the same code works regardless of which convention
    # the proxy expects. We capture the HTTP status via -w and write the
    # body to a temp file so we can surface the real error when it isn't a
    # clean 200.
    local body_file err_file http_code curl_rc
    body_file="$(mktemp)"
    err_file="$(mktemp)"
    set +e
    http_code=$(printf '%s' "$payload" | curl -sS --max-time 90 \
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

    # Extract the assistant text block and pull out the embedded JSON fix.
    RESP_FILE="$body_file" python3 - <<'PYEOF'
import json, os, sys
with open(os.environ["RESP_FILE"], "r", errors="replace") as f:
    raw = f.read()
if not raw.strip():
    print("ERR: empty response body from API", file=sys.stderr)
    sys.exit(1)
try:
    obj = json.loads(raw)
except Exception as e:
    print(f"ERR: API response not JSON: {e}", file=sys.stderr)
    print(f"raw: {raw[:400]}", file=sys.stderr)
    sys.exit(1)
if isinstance(obj, dict) and obj.get("type") == "error":
    print(f"ERR: API error: {obj.get('error', obj)}", file=sys.stderr)
    sys.exit(1)
if "error" in obj and "content" not in obj:
    print(f"ERR: API error: {obj['error']}", file=sys.stderr)
    sys.exit(1)
text = ""
for b in obj.get("content", []) or []:
    if isinstance(b, dict) and b.get("type") == "text":
        text += b.get("text", "") or ""
s = text.strip()
i = s.find("{"); j = s.rfind("}")
if i != -1 and j != -1 and j > i:
    s = s[i:j+1]
try:
    fix = json.loads(s)
except Exception as e:
    print(f"ERR: AI reply did not contain valid JSON: {e}", file=sys.stderr)
    print(f"text: {text[:400]}", file=sys.stderr)
    sys.exit(1)
if not isinstance(fix, dict):
    print("ERR: AI reply JSON was not an object", file=sys.stderr)
    sys.exit(1)
print(json.dumps(fix))
PYEOF
    local parse_rc=$?
    rm -f "$body_file" "$err_file"
    return $parse_rc
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
        diag=$(printf '%s' "$suggestion" | python3 -c \
            'import sys,json;print(json.load(sys.stdin).get("diagnosis",""))' \
            2>/dev/null || echo "")
        retry=$(printf '%s' "$suggestion" | python3 -c \
            'import sys,json;print("yes" if json.load(sys.stdin).get("retry", True) else "no")' \
            2>/dev/null || echo "yes")
        cmds=$(printf '%s' "$suggestion" | python3 -c \
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
        echo "   $(color '1;36' 'Proposed fix commands:')"
        while IFS= read -r c; do echo "     \$ $c"; done <<< "$cmds"
        read -rp "   Apply and retry this step? [y/N]: " APPLY
        case "${APPLY:-}" in
            y|Y|yes|YES) ;;
            *) warn "Skipped by user — aborting this step."; return "$rc" ;;
        esac

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

# --- 1. Python version check ------------------------------------------------
info "Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found. Install Python 3.11+ and re-run."
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    err "Python >= 3.11 required (found $PY_VER)."
    exit 1
fi
ok "Python $PY_VER"

# --- 1.5 AI-assisted error recovery opt-in ---------------------------------
if [ "$AI_FIX_CLI" -eq 1 ]; then
    if [ -z "$AI_FIX_KEY" ]; then
        echo
        info "AI auto-fix enabled via --ai-fix"
        read -rsp "   MiniMax API Key (for auto-fix): " AI_FIX_KEY
        echo
    fi
    if [ -n "$AI_FIX_KEY" ]; then
        AI_FIX_ENABLED=1
        ok "AI auto-fix enabled (model: $AI_FIX_MODEL)"
    else
        warn "No API key supplied — AI auto-fix disabled"
    fi
else
    echo
    info "AI-assisted error recovery"
    echo "   When a step fails (e.g. 'python3 -m venv' missing python3-venv,"
    echo "   or 'pip install' missing a system library), the installer can"
    echo "   send the error log to MiniMax-M2.7 and apply the fix it proposes."
    echo "   Every command is shown before it runs; nothing executes without"
    echo "   your approval."
    read -rp "   Enable AI auto-fix? [y/N]: " USE_AI_FIX
    case "${USE_AI_FIX:-}" in
        y|Y|yes|YES)
            if [ -z "$AI_FIX_KEY" ]; then
                read -rsp "   MiniMax API Key (for auto-fix): " AI_FIX_KEY
                echo
            fi
            if [ -n "$AI_FIX_KEY" ]; then
                AI_FIX_ENABLED=1
                ok "AI auto-fix enabled (model: $AI_FIX_MODEL)"
            else
                warn "No API key supplied — AI auto-fix disabled"
            fi
            ;;
        *) ;;
    esac
fi

# --- 2. venv ----------------------------------------------------------------
if [ ! -d ".venv" ]; then
    info "Creating virtualenv at .venv ..."
    if ! run_with_ai_fix "create virtualenv" python3 -m venv .venv; then
        err "Could not create venv. On Debian/Ubuntu try:"
        err "  sudo apt install python3-venv"
        exit 1
    fi
    ok "venv created"
else
    ok "venv exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

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

# --- 3. config.yaml ---------------------------------------------------------
if [ -f "config.yaml" ] && [ "$FORCE" -eq 0 ]; then
    ok "config.yaml already present (use --force to overwrite)"
else
    cp config.example.yaml config.yaml
    ok "config.yaml created from config.example.yaml"
fi

# --- 4. .env prompt ---------------------------------------------------------
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

# --- 5. make run.sh executable ---------------------------------------------
chmod +x run.sh 2>/dev/null || true

# --- 6. create runtime dirs ------------------------------------------------
mkdir -p logs state
ok "logs/ and state/ directories ready"

# --- 7. smoke-check imports ------------------------------------------------
info "Verifying imports..."
if python3 -c "import src.config, src.bot, src.exchange.simulator, src.strategy.ai_strategy" 2>/dev/null; then
    ok "imports OK"
else
    warn "import check failed — run 'python3 -c \"import src.bot\"' to debug"
fi

# --- 8. optional connectivity check ----------------------------------------
if [ -f ".env" ]; then
    read -rp "   Run a connectivity check against Binance + Telegram + MiniMax now? [Y/n]: " CONN
    case "${CONN:-Y}" in
        n|N|no|NO) ;;
        *)
            # shellcheck disable=SC1091
            set -a; source .env; set +a
            info "Pinging Binance Futures..."
            python3 - <<'PYEOF' || warn "Binance ping failed"
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
                python3 - <<'PYEOF' || warn "MiniMax API check failed"
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
