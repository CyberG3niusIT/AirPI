#!/usr/bin/env bash
# AirPI One-Line Installer
# Aufruf: curl -fsSL https://raw.githubusercontent.com/HerrMaschinist/AirPI/main/install.sh | bash
#
# Installiert AirPI (LLM Inference Server) vollautomatisch auf Raspberry Pi.
# Idempotent: kann mehrfach ausgeführt werden.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# FARBEN & FORMATIERUNG
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

OK()   { echo -e "${GREEN}✅ $*${RESET}"; }
WARN() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
ERR()  { echo -e "${RED}❌ $*${RESET}" >&2; }
INFO() { echo -e "${BLUE}ℹ  $*${RESET}"; }
STEP() { echo -e "\n${BOLD}${BLUE}$*${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AIRPI_INSTALL_DIR="/opt/airpi"
AIRPI_SRC_DIR="${AIRPI_INSTALL_DIR}/src"
AIRPI_VENV_DIR="${AIRPI_INSTALL_DIR}/venv"
AIRPI_MODELS_DIR="${AIRPI_INSTALL_DIR}/models"
AIRPI_REPO="https://github.com/HerrMaschinist/AirPI"
AIRPI_SERVICE_NAME="airpi"
AIRPI_PORT="11435"

TOTAL_STEPS=9

# ─────────────────────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────
step_banner() {
    local step="$1"
    local desc="$2"
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  Schritt ${step}/${TOTAL_STEPS}: ${desc}${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

die() {
    ERR "$1"
    echo ""
    if [[ -n "${2:-}" ]]; then
        echo -e "${YELLOW}💡 Hinweis: ${2}${RESET}"
    fi
    exit 1
}

run_as_root() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 1: PREFLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────
step_banner 1 "Preflight Checks"

# Root oder sudo
if [[ $EUID -ne 0 ]]; then
    if ! sudo -n true 2>/dev/null; then
        # sudo mit Passwort verfügbar?
        if ! sudo true 2>/dev/null; then
            die "Weder Root noch sudo-Rechte vorhanden." \
                "Führe das Script als root aus: sudo bash install.sh"
        fi
    fi
    INFO "Läuft als Benutzer $(whoami), verwende sudo für privilegierte Operationen."
else
    INFO "Läuft als root."
fi
OK "Rechte: OK"

# Raspberry Pi prüfen
PI_MODEL_FILE="/proc/device-tree/model"
ARCH=$(uname -m)
IS_PI=false
PI_MODEL_STRING=""

if [[ -f "$PI_MODEL_FILE" ]]; then
    PI_MODEL_STRING=$(tr -d '\0' < "$PI_MODEL_FILE")
    if echo "$PI_MODEL_STRING" | grep -qi "raspberry pi"; then
        IS_PI=true
    fi
fi

if [[ "$IS_PI" == "false" ]]; then
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "armhf" || "$ARCH" == "armv7l" ]]; then
        WARN "Kein Raspberry Pi erkannt, aber ARM-Architektur gefunden (${ARCH})."
        WARN "Installation wird fortgesetzt, aber AirPI ist für Raspberry Pi optimiert."
        IS_PI=true
        PI_MODEL_STRING="Unknown ARM Device"
    else
        die "Kein Raspberry Pi erkannt (Arch: ${ARCH})." \
            "AirPI ist für Raspberry Pi (aarch64/armhf) ausgelegt. Architektur: ${ARCH}"
    fi
fi
OK "Hardware: ${PI_MODEL_STRING:-ARM Device}"

# Python 3.11+ prüfen
PYTHON_BIN=""
for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VERSION=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 11 ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    die "Python 3.11+ nicht gefunden." \
        "Installiere Python 3.11: sudo apt-get install python3.11 python3.11-venv python3.11-dev"
fi
OK "Python: $($PYTHON_BIN --version)"

# RAM ermitteln
MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_MB=$((MEM_KB / 1024))
MEM_GB=$((MEM_MB / 1024))
INFO "Verfügbarer RAM: ${MEM_GB}GB (${MEM_MB}MB)"
if [[ "$MEM_MB" -lt 1800 ]]; then
    die "Zu wenig RAM: ${MEM_MB}MB. Mindestens 2GB benötigt." \
        "AirPI benötigt mindestens 2GB RAM für den kleinsten Modell-Quantisierungsgrad."
fi
OK "RAM: ${MEM_GB}GB — ausreichend"

# Disk-Space prüfen (>= 2GB frei)
INSTALL_PARENT=$(dirname "$AIRPI_INSTALL_DIR")
# Verzeichnis muss existieren für df
[[ ! -d "$INSTALL_PARENT" ]] && INSTALL_PARENT="/"
DISK_FREE_KB=$(df -k "$INSTALL_PARENT" | awk 'NR==2 {print $4}')
DISK_FREE_GB=$((DISK_FREE_KB / 1024 / 1024))
if [[ "$DISK_FREE_KB" -lt 2097152 ]]; then  # 2GB in KB
    die "Zu wenig Festplattenspeicher: ${DISK_FREE_GB}GB frei. Mindestens 2GB benötigt." \
        "Gib Speicher frei und starte den Installer erneut. Tipp: sudo du -sh /* 2>/dev/null | sort -hr | head"
fi
OK "Disk: ${DISK_FREE_GB}GB frei"

# Internet-Verbindung
if ! ping -c 1 -W 5 8.8.8.8 &>/dev/null && ! ping -c 1 -W 5 1.1.1.1 &>/dev/null; then
    die "Keine Internet-Verbindung." \
        "Stelle eine Netzwerkverbindung her und starte erneut. Tipp: ip route und ping 8.8.8.8"
fi
OK "Netzwerk: Verbunden"

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 2: PI-MODELL ERKENNEN & OPTIMIERUNGSFLAGS SETZEN
# ─────────────────────────────────────────────────────────────────────────────
step_banner 2 "Pi-Modell erkennen & CPU-Flags optimieren"

PI_GEN="unknown"
CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_NEON=ON"
CFLAGS_EXTRA=""

if echo "$PI_MODEL_STRING" | grep -qi "raspberry pi 5"; then
    PI_GEN="pi5"
    CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_NEON=ON"
    CFLAGS_EXTRA="-march=armv8.2-a+fp16+crypto -mtune=cortex-a76"
    OK "Raspberry Pi 5 (Cortex-A76) — optimale Performance, empfohlen für AirPI"
elif echo "$PI_MODEL_STRING" | grep -qi "raspberry pi 4"; then
    PI_GEN="pi4"
    CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_NEON=ON"
    CFLAGS_EXTRA="-march=armv8-a+crypto -mtune=cortex-a72"
    OK "Raspberry Pi 4 (Cortex-A72) — gute Performance"
elif echo "$PI_MODEL_STRING" | grep -qi "raspberry pi 3"; then
    PI_GEN="pi3"
    CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_NEON=ON"
    CFLAGS_EXTRA="-march=armv8-a -mtune=cortex-a53"
    WARN "Raspberry Pi 3 (Cortex-A53) erkannt — sehr langsame Inferenz erwartet."
    WARN "AirPI ist für Pi 4/5 optimiert. Pi 3 wird unterstützt, aber nicht empfohlen."
else
    WARN "Pi-Modell nicht eindeutig erkannt: '${PI_MODEL_STRING}'"
    WARN "Verwende generische ARM-Optimierungsflags."
    CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_NEON=ON"
    CFLAGS_EXTRA="-march=armv8-a"
fi

INFO "CMAKE_ARGS: ${CMAKE_ARGS}"
INFO "CFLAGS extra: ${CFLAGS_EXTRA:-'(keine)'}"

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 3: ABHÄNGIGKEITEN INSTALLIEREN
# ─────────────────────────────────────────────────────────────────────────────
step_banner 3 "System-Abhängigkeiten installieren"

INFO "Aktualisiere Paketlisten..."
run_as_root apt-get update -qq

INFO "Installiere Build-Werkzeuge und Python-Komponenten..."
run_as_root apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev

OK "System-Pakete installiert"

# Installationsverzeichnisse anlegen
INFO "Erstelle Verzeichnisstruktur in ${AIRPI_INSTALL_DIR}..."
run_as_root mkdir -p "${AIRPI_INSTALL_DIR}" "${AIRPI_MODELS_DIR}" "/etc/airpi"
# Ownership an aktuellen Benutzer (oder root wenn als root)
INSTALL_USER="${SUDO_USER:-$(whoami)}"
run_as_root chown -R "${INSTALL_USER}:${INSTALL_USER}" "${AIRPI_INSTALL_DIR}"
OK "Verzeichnisse: ${AIRPI_INSTALL_DIR}"

# Venv erstellen (idempotent)
if [[ ! -d "${AIRPI_VENV_DIR}" ]]; then
    INFO "Erstelle Python-Virtualenv in ${AIRPI_VENV_DIR}..."
    "$PYTHON_BIN" -m venv "${AIRPI_VENV_DIR}"
    OK "Virtualenv erstellt"
else
    INFO "Virtualenv existiert bereits, überspringe Erstellung."
fi

VENV_PIP="${AIRPI_VENV_DIR}/bin/pip"
VENV_PYTHON="${AIRPI_VENV_DIR}/bin/python"

# pip aktualisieren
"$VENV_PIP" install --upgrade pip --quiet
OK "pip aktualisiert: $("$VENV_PIP" --version)"

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 4: LLAMA-CPP-PYTHON INSTALLIEREN (PI-OPTIMIERT)
# ─────────────────────────────────────────────────────────────────────────────
step_banner 4 "llama-cpp-python kompilieren (ca. 15-25 Minuten)"

WARN "Dieser Schritt kompiliert llama.cpp nativ für deinen ${PI_MODEL_STRING:-ARM-Prozessor}."
WARN "Das dauert 15–25 Minuten. Bitte Geduld — der Pi ist beschäftigt!"
echo ""

# Prüfen ob bereits installiert und funktionsfähig
if "$VENV_PYTHON" -c "import llama_cpp; print('llama_cpp ok')" 2>/dev/null | grep -q "ok"; then
    OK "llama-cpp-python bereits installiert, überspringe Kompilierung."
else
    INFO "Starte Kompilierung mit optimierten Flags..."
    INFO "CMAKE_ARGS: ${CMAKE_ARGS}"

    # Fortschrittsanzeige während des Builds
    export CMAKE_ARGS="${CMAKE_ARGS}"
    export CFLAGS="${CFLAGS_EXTRA}"
    export CXXFLAGS="${CFLAGS_EXTRA}"

    # Compilierung mit Live-Output (nicht quiet, damit Nutzer Fortschritt sieht)
    if "$VENV_PIP" install \
        --no-binary llama_cpp_python \
        "llama-cpp-python>=0.3.0"; then
        OK "llama-cpp-python erfolgreich kompiliert und installiert"
    else
        die "llama-cpp-python Kompilierung fehlgeschlagen." \
            "Prüfe die Ausgabe oben. Häufige Ursachen: fehlende build-essential/cmake. Versuch: sudo apt-get install build-essential cmake libopenblas-dev"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 5: AIRPI REPO KLONEN
# ─────────────────────────────────────────────────────────────────────────────
step_banner 5 "AirPI Quellcode herunterladen"

clone_or_update_repo() {
    if [[ -d "${AIRPI_SRC_DIR}/.git" ]]; then
        INFO "Repo existiert, aktualisiere..."
        git -C "${AIRPI_SRC_DIR}" pull --ff-only || {
            WARN "git pull fehlgeschlagen (lokale Änderungen?), überspringe Update."
        }
    elif [[ -d "${AIRPI_SRC_DIR}" && "$(ls -A "${AIRPI_SRC_DIR}")" ]]; then
        INFO "Quellverzeichnis existiert bereits (kein git), überspringe Klon."
    else
        INFO "Klone AirPI von ${AIRPI_REPO}..."
        git clone "${AIRPI_REPO}" "${AIRPI_SRC_DIR}" || {
            WARN "git clone fehlgeschlagen, versuche ZIP-Download als Fallback..."
            download_zip_fallback
        }
    fi
}

download_zip_fallback() {
    local ZIP_URL="${AIRPI_REPO}/archive/refs/heads/main.zip"
    local ZIP_TMP="/tmp/airpi_download.zip"
    INFO "Lade ZIP von: ${ZIP_URL}"
    if command -v wget &>/dev/null; then
        wget --progress=bar:force -O "$ZIP_TMP" "$ZIP_URL" 2>&1 || die "ZIP-Download fehlgeschlagen." "Prüfe Netzwerkverbindung und ob ${AIRPI_REPO} erreichbar ist."
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$ZIP_TMP" "$ZIP_URL" || die "ZIP-Download fehlgeschlagen." "Prüfe Netzwerkverbindung."
    else
        die "Weder git, wget noch curl verfügbar." "sudo apt-get install git"
    fi

    INFO "Entpacke ZIP..."
    run_as_root apt-get install -y unzip -qq
    mkdir -p "${AIRPI_SRC_DIR}"
    unzip -q "$ZIP_TMP" -d "/tmp/airpi_extract/"
    # ZIP enthält Unterordner AirPI-main/
    cp -r /tmp/airpi_extract/AirPI-main/. "${AIRPI_SRC_DIR}/"
    rm -rf "$ZIP_TMP" "/tmp/airpi_extract/"
    OK "ZIP-Fallback: Quellcode entpackt"
}

clone_or_update_repo
OK "AirPI Quellcode: ${AIRPI_SRC_DIR}"

# Python-Abhängigkeiten aus requirements.txt installieren
if [[ -f "${AIRPI_SRC_DIR}/requirements.txt" ]]; then
    INFO "Installiere Python-Abhängigkeiten aus requirements.txt..."
    # llama-cpp-python überspringen (bereits mit optimierten Flags installiert)
    grep -v "llama-cpp-python" "${AIRPI_SRC_DIR}/requirements.txt" > /tmp/airpi_req_filtered.txt
    "$VENV_PIP" install -r /tmp/airpi_req_filtered.txt --quiet
    rm /tmp/airpi_req_filtered.txt
    OK "Python-Abhängigkeiten installiert"
fi

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 6: PASSENDES MODELL HERUNTERLADEN
# ─────────────────────────────────────────────────────────────────────────────
step_banner 6 "Sprachmodell herunterladen (basierend auf RAM: ${MEM_GB}GB)"

# Modell-Auswahl basierend auf RAM
select_model() {
    if [[ "$MEM_MB" -lt 3000 ]]; then
        # <= ~2.5GB RAM
        MODEL_NAME="Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        MODEL_SIZE_HINT="~1.2GB"
        MODEL_SHA256="3e1e74c8f1459f58b4a4e27d68dcfe038ccea0c53f5adb5ce744c3b5a8e55e58"
        INFO "RAM <= 2GB: Wähle Qwen2.5-Coder-1.5B-Q4_K_M (${MODEL_SIZE_HINT})"
    elif [[ "$MEM_MB" -lt 6000 ]]; then
        # ~4GB RAM
        MODEL_NAME="Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
        MODEL_SIZE_HINT="~2.5GB"
        MODEL_SHA256=""
        INFO "RAM ~4GB: Wähle Qwen2.5-Coder-3B-Q4_K_M (${MODEL_SIZE_HINT})"
    else
        # >= 8GB RAM (Pi 5 Empfehlung)
        MODEL_NAME="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        MODEL_SIZE_HINT="~4.1GB"
        MODEL_SHA256=""
        OK "RAM >= 8GB (Pi 5): Wähle Qwen2.5-Coder-7B-Q4_K_M (${MODEL_SIZE_HINT}) — beste Qualität"
    fi
}

select_model

MODEL_PATH="${AIRPI_MODELS_DIR}/${MODEL_NAME}"

download_model() {
    INFO "Lade Modell herunter: ${MODEL_NAME}"
    INFO "Quelle: ${MODEL_URL}"
    INFO "Ziel:   ${MODEL_PATH}"
    WARN "Download-Größe: ${MODEL_SIZE_HINT} — das kann einige Minuten dauern..."
    echo ""

    # Download mit wget oder curl (beide zeigen Progress)
    if command -v wget &>/dev/null; then
        wget --progress=bar:force:noscroll \
             --tries=3 \
             --continue \
             -O "${MODEL_PATH}.tmp" \
             "${MODEL_URL}" 2>&1 \
        || { rm -f "${MODEL_PATH}.tmp"; die "Modell-Download fehlgeschlagen." "Prüfe Internetverbindung oder lade manuell herunter: wget -O '${MODEL_PATH}' '${MODEL_URL}'"; }
    elif command -v curl &>/dev/null; then
        curl -L \
             --progress-bar \
             --retry 3 \
             --continue-at - \
             -o "${MODEL_PATH}.tmp" \
             "${MODEL_URL}" \
        || { rm -f "${MODEL_PATH}.tmp"; die "Modell-Download fehlgeschlagen." "Prüfe Internetverbindung oder lade manuell herunter."; }
    else
        die "Weder wget noch curl verfügbar." "sudo apt-get install wget"
    fi

    # Atomisches Verschieben nach erfolgreichem Download
    mv "${MODEL_PATH}.tmp" "${MODEL_PATH}"
    OK "Download abgeschlossen: ${MODEL_PATH}"
}

verify_model_checksum() {
    local path="$1"
    local expected_sha="$2"
    if [[ -z "$expected_sha" ]]; then
        INFO "Keine SHA256-Prüfsumme hinterlegt, überspringe Verifikation."
        return 0
    fi
    INFO "Prüfe SHA256-Checksumme..."
    local actual_sha
    actual_sha=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual_sha" == "$expected_sha" ]]; then
        OK "Checksumme OK: ${actual_sha:0:16}..."
    else
        ERR "Checksumme FALSCH!"
        ERR "  Erwartet: ${expected_sha}"
        ERR "  Erhalten: ${actual_sha}"
        rm -f "$path"
        die "Modell-Datei beschädigt oder manipuliert, wurde gelöscht." \
            "Starte den Installer erneut, um die Datei neu herunterzuladen."
    fi
}

if [[ -f "${MODEL_PATH}" ]]; then
    FILE_SIZE_MB=$(( $(stat -c%s "${MODEL_PATH}") / 1024 / 1024 ))
    if [[ "$FILE_SIZE_MB" -gt 100 ]]; then
        INFO "Modell bereits vorhanden (${FILE_SIZE_MB}MB), überspringe Download."
        if [[ -n "${MODEL_SHA256:-}" ]]; then
            verify_model_checksum "${MODEL_PATH}" "${MODEL_SHA256}"
        fi
    else
        WARN "Modell-Datei zu klein (${FILE_SIZE_MB}MB), scheint unvollständig — lade neu herunter."
        rm -f "${MODEL_PATH}"
        download_model
        verify_model_checksum "${MODEL_PATH}" "${MODEL_SHA256:-}"
    fi
else
    download_model
    verify_model_checksum "${MODEL_PATH}" "${MODEL_SHA256:-}"
fi

OK "Modell bereit: ${MODEL_NAME}"

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 7: SYSTEMD SERVICE INSTALLIEREN
# ─────────────────────────────────────────────────────────────────────────────
step_banner 7 "Systemd Service installieren"

SERVICE_FILE_SRC="${AIRPI_SRC_DIR}/systemd/airpi.service"
SERVICE_FILE_DST="/etc/systemd/system/${AIRPI_SERVICE_NAME}.service"
ENV_FILE="/etc/airpi/airpi.env"

if [[ ! -f "$SERVICE_FILE_SRC" ]]; then
    die "Service-Datei nicht gefunden: ${SERVICE_FILE_SRC}" \
        "Prüfe ob das Repo korrekt geklont wurde: ls ${AIRPI_SRC_DIR}/systemd/"
fi

# Service-Datei anpassen: User, Pfade und Modell eintragen
SERVICE_CONTENT=$(cat "$SERVICE_FILE_SRC")

# Aktuellen Benutzer bestimmen
SERVICE_USER="${SUDO_USER:-$(whoami)}"
[[ "$SERVICE_USER" == "root" ]] && SERVICE_USER="root"

# Service-Datei mit korrekten Pfaden generieren
cat > /tmp/airpi_service_tmp.service << EOF
[Unit]
Description=AirPI Inference Server
Documentation=https://github.com/HerrMaschinist/AirPI
After=network.target
StartLimitIntervalSec=60s
StartLimitBurst=3

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${AIRPI_SRC_DIR}
EnvironmentFile=-${ENV_FILE}

ExecStart=${AIRPI_VENV_DIR}/bin/uvicorn server:app \\
    --host 127.0.0.1 \\
    --port ${AIRPI_PORT} \\
    --workers 1 \\
    --log-level info \\
    --no-access-log

Environment=AIRPI_MODELS_DIR=${AIRPI_MODELS_DIR}
Environment=AIRPI_DEFAULT_MODEL=${MODEL_NAME}
Environment=AIRPI_FAST_MODEL=${MODEL_NAME}
Environment=AIRPI_N_THREADS=4
Environment=AIRPI_N_CTX_SMALL=2048
Environment=AIRPI_N_CTX_LARGE=2048
Environment=AIRPI_N_THREADS_BATCH=4
Environment=AIRPI_N_BATCH_SMALL=512
Environment=AIRPI_N_BATCH_LARGE=256
Environment=AIRPI_N_UBATCH_SMALL=256
Environment=AIRPI_N_UBATCH_LARGE=128
Environment=AIRPI_FLASH_ATTN=true
Environment=AIRPI_MMAP=true
Environment=AIRPI_MLOCK=false
Environment=AIRPI_MAX_QUEUE=10
Environment=AIRPI_KEEP_ALIVE_TIMEOUT=900

Restart=on-failure
RestartSec=5s
TimeoutStartSec=180
TimeoutStopSec=30
OOMPolicy=stop
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
EOF

run_as_root cp /tmp/airpi_service_tmp.service "${SERVICE_FILE_DST}"
rm /tmp/airpi_service_tmp.service

# Env-Datei anlegen (falls nicht vorhanden, zum manuellen Anpassen)
if [[ ! -f "$ENV_FILE" ]]; then
    run_as_root tee "$ENV_FILE" > /dev/null << ENVEOF
# AirPI Konfiguration — hier können Einstellungen überschrieben werden
# Änderungen nach: systemctl restart airpi

# Beispiel: Anderes Modell verwenden
# AIRPI_DEFAULT_MODEL=anderes-modell.gguf

# Mehr Threads (maximal = Anzahl CPU-Kerne)
# AIRPI_N_THREADS=4
ENVEOF
fi

# Systemd neu laden und Service aktivieren
run_as_root systemctl daemon-reload
run_as_root systemctl enable "${AIRPI_SERVICE_NAME}"

# Bereits laufenden Service neu starten, sonst starten
if run_as_root systemctl is-active --quiet "${AIRPI_SERVICE_NAME}" 2>/dev/null; then
    INFO "Service läuft bereits, starte neu..."
    run_as_root systemctl restart "${AIRPI_SERVICE_NAME}"
else
    run_as_root systemctl start "${AIRPI_SERVICE_NAME}"
fi

OK "Systemd Service: ${AIRPI_SERVICE_NAME} aktiviert und gestartet"

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 8: SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────
step_banner 8 "Smoke-Test: Warte auf AirPI..."

AIRPI_BASE_URL="http://localhost:${AIRPI_PORT}"
MAX_WAIT=60
WAITED=0
READY=false

INFO "Warte bis AirPI antwortet (max ${MAX_WAIT}s)..."
echo -n "  "

while [[ "$WAITED" -lt "$MAX_WAIT" ]]; do
    if curl -sf "${AIRPI_BASE_URL}/live" &>/dev/null; then
        READY=true
        echo ""
        break
    fi
    echo -n "."
    sleep 2
    WAITED=$((WAITED + 2))
done

if [[ "$READY" == "false" ]]; then
    echo ""
    ERR "AirPI hat nicht innerhalb von ${MAX_WAIT}s geantwortet."
    INFO "Service-Log anzeigen:"
    run_as_root journalctl -u "${AIRPI_SERVICE_NAME}" -n 30 --no-pager || true
    die "Smoke-Test fehlgeschlagen." \
        "Prüfe den Service mit: sudo journalctl -u airpi -f"
fi

OK "/live Endpoint antwortet"

# Test-Request ausführen
INFO "Führe Test-Request aus..."
TEST_PAYLOAD='{"model":"'${MODEL_NAME}'","prompt":"Was ist 2+2?","max_tokens":20,"stream":false}'
START_TS=$(date +%s%3N)

TEST_RESPONSE=$(curl -sf \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$TEST_PAYLOAD" \
    --max-time 60 \
    "${AIRPI_BASE_URL}/api/generate" 2>/dev/null || echo "")

END_TS=$(date +%s%3N)
ELAPSED_MS=$((END_TS - START_TS))

if [[ -n "$TEST_RESPONSE" ]]; then
    # Tokens/Sek aus Response extrahieren (falls vorhanden)
    TOKENS_PER_SEC=$(echo "$TEST_RESPONSE" | grep -o '"eval_rate":[0-9.]*' | cut -d: -f2 || echo "")
    OK "Test-Request erfolgreich (${ELAPSED_MS}ms)"
    if [[ -n "$TOKENS_PER_SEC" ]]; then
        OK "Inferenz-Geschwindigkeit: ${TOKENS_PER_SEC} Token/Sekunde"
    fi
else
    WARN "Test-Request Antwort war leer oder fehlerhaft, aber Service läuft."
    INFO "Manuell testen: curl -X POST ${AIRPI_BASE_URL}/api/generate -H 'Content-Type: application/json' -d '{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Hallo\",\"max_tokens\":20}'"
fi

# Health-Check
HEALTH=$(curl -sf "${AIRPI_BASE_URL}/health" 2>/dev/null || echo "{}")
INFO "Health-Status: ${HEALTH}"

# ─────────────────────────────────────────────────────────────────────────────
# SCHRITT 9: ABSCHLUSS
# ─────────────────────────────────────────────────────────────────────────────
step_banner 9 "Installation abgeschlossen"

# airpi CLI-Befehl anlegen (falls nicht vorhanden)
AIRPI_CLI="/usr/local/bin/airpi"
if [[ ! -f "$AIRPI_CLI" ]]; then
    run_as_root tee "$AIRPI_CLI" > /dev/null << 'CLIPEOF'
#!/usr/bin/env bash
# AirPI CLI Shortcut
AIRPI_BASE="http://localhost:11435"

case "${1:-help}" in
    chat)
        echo "AirPI Chat (Strg+C zum Beenden)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        MODEL=$(curl -sf "${AIRPI_BASE}/api/tags" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['models'][0]['name'] if d.get('models') else '')" 2>/dev/null || echo "")
        while true; do
            read -r -p "Du: " INPUT
            [[ -z "$INPUT" ]] && continue
            [[ "$INPUT" == "exit" || "$INPUT" == "quit" ]] && break
            echo -n "AirPI: "
            curl -sf -X POST "${AIRPI_BASE}/api/generate" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL}\",\"prompt\":\"${INPUT}\",\"stream\":true}" \
                | while IFS= read -r line; do
                    echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('response',''), end='', flush=True)" 2>/dev/null
                done
            echo ""
        done
        ;;
    bench)
        echo "AirPI Benchmark..."
        MODEL=$(curl -sf "${AIRPI_BASE}/api/tags" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['models'][0]['name'] if d.get('models') else '')" 2>/dev/null || echo "")
        START=$(date +%s%3N)
        RESP=$(curl -sf -X POST "${AIRPI_BASE}/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"Erkläre Quicksort in 3 Sätzen.\",\"max_tokens\":100,\"stream\":false}")
        END=$(date +%s%3N)
        MS=$((END - START))
        echo "Antwort in ${MS}ms"
        echo "$RESP" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); r=d.get('eval_rate'); print(f'Token/Sek: {r}' if r else 'Token/Sek: N/A')" 2>/dev/null
        ;;
    status)
        curl -sf "${AIRPI_BASE}/health" | python3 -m json.tool 2>/dev/null || echo "Nicht erreichbar"
        ;;
    stop)   sudo systemctl stop airpi ;;
    start)  sudo systemctl start airpi ;;
    restart) sudo systemctl restart airpi ;;
    logs)   sudo journalctl -u airpi -f ;;
    *)
        echo "Verwendung: airpi <Befehl>"
        echo ""
        echo "Befehle:"
        echo "  chat      Interaktiver Chat"
        echo "  bench     Inferenz-Benchmark"
        echo "  status    Server-Status anzeigen"
        echo "  start     Service starten"
        echo "  stop      Service stoppen"
        echo "  restart   Service neu starten"
        echo "  logs      Live-Log anzeigen"
        ;;
esac
CLIPEOF
    run_as_root chmod +x "$AIRPI_CLI"
    OK "CLI-Befehl installiert: airpi"
fi

# Finale Zusammenfassung
echo ""
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}${GREEN}  ✅ AirPI erfolgreich installiert!${RESET}"
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  ${GREEN}✅ AirPI läuft auf   ${BOLD}http://localhost:${AIRPI_PORT}${RESET}"
echo -e "  ${BLUE}📊 Chat UI:          ${BOLD}http://localhost:${AIRPI_PORT}/ui${RESET}"
echo -e "  ${BLUE}📋 Health-Check:     ${BOLD}http://localhost:${AIRPI_PORT}/health${RESET}"
echo -e "  ${BLUE}🔌 OpenAI-API:       ${BOLD}http://localhost:${AIRPI_PORT}/v1/chat/completions${RESET}"
echo ""
echo -e "  ${BOLD}Aktives Modell:${RESET}  ${MODEL_NAME}"
echo -e "  ${BOLD}Hardware:${RESET}        ${PI_MODEL_STRING:-ARM Device}"
echo -e "  ${BOLD}RAM:${RESET}             ${MEM_GB}GB"
echo ""
echo -e "${BOLD}  Nächste Schritte:${RESET}"
echo -e "    ${GREEN}airpi chat${RESET}       # Chat im Terminal"
echo -e "    ${GREEN}airpi bench${RESET}      # Inferenz-Benchmark"
echo -e "    ${GREEN}airpi status${RESET}     # Service-Status"
echo -e "    ${GREEN}airpi logs${RESET}       # Live-Log anzeigen"
echo ""
echo -e "  ${YELLOW}Konfiguration:${RESET} ${ENV_FILE}"
echo -e "  ${YELLOW}Modelle:${RESET}       ${AIRPI_MODELS_DIR}/"
echo -e "  ${YELLOW}Quellcode:${RESET}     ${AIRPI_SRC_DIR}/"
echo ""
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
