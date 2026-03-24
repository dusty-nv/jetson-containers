#!/usr/bin/env bash
set -e

ROOT="$(dirname "$(readlink -f "$0")")"
DEVPI_DIR="${ROOT}/packages/net/devpi"
COMPOSE_FILE="${DEVPI_DIR}/compose.yml"
COMPOSE_PROJECT="devpi-local"

if [ -f "${ROOT}/.env" ]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi

DEVPI_PORT="${DEVPI_PORT:-3141}"
DEVPI_PASSWORD="${DEVPI_PASSWORD:-123}"
DEVPI_USER_PASSWORD="${DEVPI_USER_PASSWORD:-123}"
DEVPI_URL="http://localhost:${DEVPI_PORT}"
DEVPI_CACHE="${DEVPI_CACHE:-${DEVPI_DIR}/cache/devpi}"

APT_PORT="${APT_PORT:-8034}"
APT_URL="http://localhost:${APT_PORT}"
APT_ROOT="${APT_ROOT:-${DEVPI_DIR}/cache/apt}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

compose() {
    docker compose -p "${COMPOSE_PROJECT}" -f "${COMPOSE_FILE}" "$@"
}

devpi_exec() {
    docker compose -p "${COMPOSE_PROJECT}" -f "${COMPOSE_FILE}" \
        exec devpi-server "$@"
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local max=60
    for i in $(seq 1 "${max}"); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            return 0
        fi
        printf "\r  waiting for %s... (%d/%d)" "${name}" "$i" "$max"
        sleep 2
    done
    echo ""
    echo "ERROR: ${name} did not become ready within ${max} attempts"
    compose logs --tail=30
    exit 1
}

echo "========================================="
echo " Launching local PyPI + APT servers"
echo "========================================="

mkdir -p "${DEVPI_CACHE}"
mkdir -p "${APT_ROOT}/jp6/cu126" \
         "${APT_ROOT}/jp6/cu129" \
         "${APT_ROOT}/jp6/cu132" \
         "${APT_ROOT}/jp7/cu132" \
         "${APT_ROOT}/sbsa/cu130/24.04" \
         "${APT_ROOT}/sbsa/cu132/24.04" \
         "${APT_ROOT}/sbsa/cu133/24.04" \
         "${APT_ROOT}/sbsa/cu133/26.04" \
         "${APT_ROOT}/amd64/cu132/24.04" \
         "${APT_ROOT}/amd64/cu133/26.04" \
         "${APT_ROOT}/assets" \
         "${APT_ROOT}/multiarch"

# Clean up old containers (from previous docker-run or compose runs)
docker rm -f devpi-local 2>/dev/null || true
compose down 2>/dev/null || true

fix_cache_permissions() {
    local dir="$1"
    if [ -d "${dir}" ]; then
        local owner
        owner="$(stat -c '%u' "${dir}" 2>/dev/null || echo "")"
        if [ -n "${owner}" ] && [ "${owner}" != "${HOST_UID}" ]; then
            echo "    fixing ownership: ${dir}"
            sudo chown -R "${HOST_UID}:${HOST_GID}" "${dir}"
        fi
    fi
}

echo ""
echo "==> Ensuring cache directory permissions..."
fix_cache_permissions "$(dirname "${APT_ROOT}")"
fix_cache_permissions "${APT_ROOT}"
fix_cache_permissions "${DEVPI_CACHE}"

echo ""
echo "==> Building and starting services..."
export DEVPI_PORT DEVPI_PASSWORD DEVPI_CACHE APT_PORT APT_ROOT HOST_UID HOST_GID
compose up -d --build

echo ""
echo "==> Waiting for devpi server..."
wait_for_server "${DEVPI_URL}/+api" "devpi"
echo ""
echo "    devpi server is ready at ${DEVPI_URL}"

echo ""
echo "==> Waiting for APT server..."
wait_for_server "${APT_URL}" "apt-server"
echo ""
echo "    APT server is ready at ${APT_URL}"

# ── Configure devpi (first run only) ────────────────────────────────
CONFIGURED_MARKER="${DEVPI_CACHE}/.configured"

if [ ! -f "${CONFIGURED_MARKER}" ]; then
    echo ""
    echo "==> Configuring devpi server (first run)..."

    devpi_exec devpi use http://localhost:3141
    devpi_exec devpi login root --password "${DEVPI_PASSWORD}"

    # Disable local file storage for packages fetched from upstream PyPI
    devpi_exec devpi index root/pypi mirror_use_external_urls=true

    # Centralized index that all user indexes inherit from
    echo "    Creating root/dev-pypi..."
    devpi_exec devpi index -c root/dev-pypi bases=root/pypi mirror_whitelist='*'

    # Create users
    echo "    Creating users: jp6, jp7, sbsa, amd64..."
    for user in jp6 jp7 sbsa amd64; do
        devpi_exec devpi user -c "${user}" "password=${DEVPI_USER_PASSWORD}"
    done

    # JP6 indexes
    echo "    Creating jp6/cu126..."
    devpi_exec devpi login jp6 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c jp6/cu126 bases=root/dev-pypi

    echo "    Creating jp6/cu129..."
    devpi_exec devpi login jp6 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c jp6/cu129 bases=root/dev-pypi

    echo "    Creating jp6/cu132..."
    devpi_exec devpi login jp6 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c jp6/cu132 bases=root/dev-pypi

    # JP7 indexes
    echo "    Creating jp7/cu132..."
    devpi_exec devpi login jp7 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c jp7/cu132 bases=root/dev-pypi

    echo "    Creating jp7/cu133..."
    devpi_exec devpi login jp7 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c jp7/cu133 bases=root/dev-pypi

    # SBSA indexes
    echo "    Creating sbsa/cu130, sbsa/cu132, sbsa/cu133..."
    devpi_exec devpi login sbsa --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c sbsa/cu130 bases=root/dev-pypi
    devpi_exec devpi index -c sbsa/cu132 bases=root/dev-pypi
    devpi_exec devpi index -c sbsa/cu133 bases=root/dev-pypi

    # AMD64 indexes
    echo "    Creating amd64/cu132...,"
    devpi_exec devpi login amd64 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c amd64/cu132 bases=root/dev-pypi

    echo "    Creating amd64/cu133..."
    devpi_exec devpi login amd64 --password "${DEVPI_USER_PASSWORD}"
    devpi_exec devpi index -c amd64/cu133 bases=root/dev-pypi

    touch "${CONFIGURED_MARKER}"
    echo "    Configuration complete!"
else
    echo ""
    echo "==> Server already configured (found ${CONFIGURED_MARKER})."
    echo "    Delete that file and re-run to force reconfiguration."
fi

# ── Update .env with local server settings ───────────────────────────
ENV_FILE="${ROOT}/.env"
MARKER_START="# --- LOCAL PYPI/APT --- auto-generated by launch_pypi.sh ---"
MARKER_END="# --- END LOCAL PYPI/APT ---"

if [ -f "${ENV_FILE}" ]; then
    awk -v start="${MARKER_START}" -v end="${MARKER_END}" '
        $0 == start { skip=1; next }
        $0 == end   { skip=0; next }
        !skip
    ' "${ENV_FILE}" > "${ENV_FILE}.tmp"
    mv "${ENV_FILE}.tmp" "${ENV_FILE}"
fi

cat >> "${ENV_FILE}" <<EOF

${MARKER_START}
DEVPI_URL=${DEVPI_URL}
DEVPI_PORT=${DEVPI_PORT}
DEVPI_PASSWORD=${DEVPI_PASSWORD}
DEVPI_USER_PASSWORD=${DEVPI_USER_PASSWORD}
DEVPI_CACHE=${DEVPI_CACHE}
PIP_UPLOAD_HOST=localhost:${DEVPI_PORT}
PIP_UPLOAD_PASS=${DEVPI_USER_PASSWORD}
APT_PORT=${APT_PORT}
APT_ROOT=${APT_ROOT}
LOCAL_TAR_INDEX_URL=${APT_URL}
${MARKER_END}
EOF

echo ""
echo "==> Updated ${ENV_FILE} with local server settings"

set -a
source "${ENV_FILE}"
set +a

echo ""
echo "========================================="
echo " Local servers are running!"
echo "========================================="
echo ""
echo " PyPI (devpi): ${DEVPI_URL}"
echo " APT (nginx):  ${APT_URL}"
echo ""
echo " PyPI indexes:"
echo "   jp6/cu126   -> ${DEVPI_URL}/jp6/cu126/+simple/"
echo "   jp6/cu129   -> ${DEVPI_URL}/jp6/cu129/+simple/"
echo "   jp7/cu132   -> ${DEVPI_URL}/jp7/cu132/+simple/"
echo "   jp7/cu133   -> ${DEVPI_URL}/jp7/cu133/+simple/"
echo "   sbsa/cu130  -> ${DEVPI_URL}/sbsa/cu130/+simple/"
echo "   sbsa/cu132  -> ${DEVPI_URL}/sbsa/cu132/+simple/"
echo "   sbsa/cu133  -> ${DEVPI_URL}/sbsa/cu133/+simple/"
echo "   amd64/cu132 -> ${DEVPI_URL}/amd64/cu132/+simple/"
echo "   amd64/cu133 -> ${DEVPI_URL}/amd64/cu133/+simple/"
echo ""
echo " APT indexes:"
echo "   jp6/cu126   -> ${APT_URL}/jp6/cu126/"
echo "   jp6/cu129   -> ${APT_URL}/jp6/cu129/"
echo "   jp7/cu132   -> ${APT_URL}/jp7/cu132/"
echo "   jp7/cu133   -> ${APT_URL}/jp7/cu133/"
echo "   sbsa/cu130  -> ${APT_URL}/sbsa/cu130/"
echo "   sbsa/cu132  -> ${APT_URL}/sbsa/cu132/"
echo "   sbsa/cu133  -> ${APT_URL}/sbsa/cu133/"
echo "   amd64/cu132 -> ${APT_URL}/amd64/cu132/"
echo "   amd64/cu133 -> ${APT_URL}/amd64/cu133/"
echo "   multiarch   -> ${APT_URL}/multiarch/"
echo "   assets      -> ${APT_URL}/assets/"
echo ""
echo " APT files served from: ${APT_ROOT}"
echo ""
echo " config.py will auto-select the correct index per architecture."
echo " Primary: localhost  |  Fallback: jetson-ai-lab.io"
echo ""
echo " To stop:  docker compose -p ${COMPOSE_PROJECT} -f ${COMPOSE_FILE} down"
echo ""
