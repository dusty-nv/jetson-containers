#!/usr/bin/env bash
# install python packages required for running build.sh/autotag
# and link these scripts under /usr/local so they're in the path
set -ex

ROOT="$(dirname "$(readlink -f "$0")")"
INSTALL_PREFIX="/usr/local/bin"
LSB_RELEASE="$(lsb_release -rs)"

# use virtualenv if 24.04
if [ "$LSB_RELEASE" = "24.04" ]; then
  # ensure python3-venv is available
  if ! dpkg -s python3-venv >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv openssl
  fi

  VENV="$ROOT/venv"
  mkdir -p "$VENV" || echo "warning:  $VENV either previously existed, or failed to be created"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
fi

# install pip if needed
pip3 --version || sudo apt-get install -y python3-pip

# install package requirements
pip3 install -r "$ROOT/requirements.txt"

# link scripts to path
sudo ln -sf "$ROOT/autotag" "$INSTALL_PREFIX/autotag"
sudo ln -sf "$ROOT/jetson-containers" "$INSTALL_PREFIX/jetson-containers"

# create a default env
if [ ! -f "${ROOT}/.env" ]; then
  cp "${ROOT}/.env.default" "${ROOT}/.env"
fi

# Load environment variables from .env and check if we need to create local dist dirs or users
if [ -f ".env" ]; then
    set -a
    source .env
    set +a

    if [ -n "$SCP_UPLOAD_HOST" ]; then
        for var in SCP_UPLOAD_USER SCP_UPLOAD_PASS LOCAL_TAR_INDEX_URL; do
            if [ -z "${!var:-}" ]; then
                echo "Error: Required APT Server environment variable \$${var} is not set" >&2
                exit 1
            fi
        done

        # Check if SCP_UPLOAD_HOST contains: /home/${SCP_UPLOAD_USER}/dist
        if [ "${SCP_UPLOAD_HOST#*/home/${SCP_UPLOAD_USER}/dist}" = "$SCP_UPLOAD_HOST" ]; then
            echo "ERROR: SCP_UPLOAD_HOST must contain /home/${SCP_UPLOAD_USER}/dist path after the ':' char."
            echo "  Current: '${SCP_UPLOAD_HOST}'"
            echo "  Expected: 'localhost:/home/${SCP_UPLOAD_USER}/dist/apt' (or similar)"
            exit 1
        fi

        # Check if SCP_UPLOAD_HOST starts with "localhost:" OR "127.0.0.1:"
        if [ "${SCP_UPLOAD_HOST#localhost:}" != "$SCP_UPLOAD_HOST" ] || \
           [ "${SCP_UPLOAD_HOST#127.0.0.1:}" != "$SCP_UPLOAD_HOST" ]; then
            # Create shared group for consistent permissions
            GROUP_NAME="jetson-containers"
            if ! getent group "$GROUP_NAME" >/dev/null 2>&1; then
                sudo groupadd --system "$GROUP_NAME" || { echo "ERROR: Failed to create group $GROUP_NAME"; exit 1; }
                echo "Created group: $GROUP_NAME"
            fi

            # make sure that the SCP_UPLOAD_USER with SCP_UPLOAD_PASS exists on localhost
            if ! id "$SCP_UPLOAD_USER" >/dev/null 2>&1; then
                echo "Creating user: $SCP_UPLOAD_USER"
                # Create user with home directory and bash shell
                sudo useradd -m -s /bin/bash -g "$GROUP_NAME" "$SCP_UPLOAD_USER" 2>/dev/null || echo "Failed to created $SCP_UPLOAD_USER user!" && echo "Created $SCP_UPLOAD_USER user!"

                # Set password securely (using chpasswd)
                sudo usermod -p "$(openssl passwd -6 $SCP_UPLOAD_PASS)" "$SCP_UPLOAD_USER"
                if [ $? -ne 0 ]; then
                    echo "ERROR: Failed to set password for $SCP_UPLOAD_USER"
                    exit 1
                fi
                echo "User $SCP_UPLOAD_USER created successfully"
            else
                echo "User $SCP_UPLOAD_USER already exists. Skipping creation."
            fi

            # create the dist directory tree (as root)
            sudo mkdir -p "/home/${SCP_UPLOAD_USER}/dist/apt/jp6/cu126" \
                         "/home/${SCP_UPLOAD_USER}/dist/apt/jp6/cu129/24.04" \
                         "/home/${SCP_UPLOAD_USER}/dist/apt/sbsa/cu130" \
                         "/home/${SCP_UPLOAD_USER}/dist/apt/assets" \
                         "/home/${SCP_UPLOAD_USER}/dist/apt/multiarch" \
                         \
                         "/home/${SCP_UPLOAD_USER}/dist/pypi/jp6/cu126" \
                         "/home/${SCP_UPLOAD_USER}/dist/pypi/jp6/cu129" \
                         "/home/${SCP_UPLOAD_USER}/dist/pypi/sbsa/cu130"

            # set owner and group for the tree (owner: jetson, group: jetson-containers)
            sudo chown -R ${SCP_UPLOAD_USER}:${GROUP_NAME} "/home/${SCP_UPLOAD_USER}"
            sudo chgrp "${GROUP_NAME}" "/home/${SCP_UPLOAD_USER}"
            sudo chmod 770 "/home/${SCP_UPLOAD_USER}"
            sudo chmod -R g+rwX "/home/${SCP_UPLOAD_USER}"
            sudo find "/home/${SCP_UPLOAD_USER}" -type d -exec chmod g+s {} \;

            # Add current user to the jetson-containers group
            if ! id -Gn | grep -qw "$GROUP_NAME"; then
                sudo usermod -aG "$GROUP_NAME" "$USER" || { echo "ERROR: Failed to add current user to $GROUP_NAME"; exit 1; }
                echo "Added current user ($USER) to $GROUP_NAME"
            fi

            # apply the permissions in new terminal sessions
            exec sg "$GROUP_NAME" newgrp `id -gn`
        fi
    fi
fi
