#!/bin/sh
set -e

# Helper function used to make all logging messages look similar.
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S,000') $1 [devpi] $2"
}

# Check that we have some kind of password for the root user.
if [ -z "${DEVPI_PASSWORD}" ]; then
    log "ERROR" "Root password cannot be empty"
    exit 1
fi

# Perform an export and exit in case the /export folder exists.
if [ -d "/export" ]; then
    log "INFO" "Exporting current database"
    devpi-export $@ /export
    log "INFO" "Export finished"
    exit 0
fi

# Perform an import and exit in case the /import folder exists.
if [ -d "/import" ]; then
    log "INFO" "Beginning import of data"
    devpi-import --root-passwd "${DEVPI_PASSWORD}" $@ /import
    log "INFO" "Import complete"
    exit 0
fi

# Execute any potential shell scripts in the devpi/ folder, or source
# any file ending with ".envsh".
find "/devpi.d/" -follow -type f -print | sort -V | while read -r f; do
    case "${f}" in
        *.envsh)
            if [ -x "${f}" ]; then
                log "INFO" "Sourcing ${f}";
                . "${f}"
            else
                log "INFO" "Ignoring ${f}, not executable";
            fi
            ;;
        *.sh)
            if [ -x "${f}" ]; then
                log "INFO" "Launching ${f}";
                "${f}"
            else
                log "INFO" "Ignoring ${f}, not executable";
            fi
            ;;
        *)
            log "INFO" "Ignoring ${f}"
            ;;
    esac
done

# Initialize devpi if there is no indication of it being run before.
if [ ! -f "${DEVPISERVER_SERVERDIR}/.serverversion" ]; then
    devpi-init --root-passwd "${DEVPI_PASSWORD}"
fi

# change from default 3141 to 80 to push it public

devpi-server \
    --host 0.0.0.0 \
    --port 3141 \
    --max-request-body-size 4294967296 \
    --serverdir /devpi \
    --indexer-backend null
