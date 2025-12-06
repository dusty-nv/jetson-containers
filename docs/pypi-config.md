# Local PyPI & APT Servers Configuration

This document explains how to run local **PyPI** and **APT** servers using the provided `docker-compose.yaml` stack so you can build containers with the `jetson-containers` CLI and fetch local wheels / tarballs during builds. The stack exposes multiple PyPI servers (one per CUDA/architecture variant) and a single static APT file server. Each service will automatically fallback to the Jetson Community registries when a package is not available locally.

> [!NOTE]  
> The stack serves files from a `SCP_UPLOAD_USER` home directory (`/home/${SCP_UPLOAD_USER}/dist/apt` by default). The APT server is a **simple** static HTTP server (no apt metadata generation). It is intended to host tarballs / `.deb` artifacts used by `jetson-containers`, not to act as a full-featured apt repository manager.

## Quick Start

### Start build with the local servers

1. Uncomment `LOCAL_PIP_INDEX_URL` and `LOCAL_TAR_INDEX_URL` in `.env` file.
2. Start local servers:

    ```bash
    docker compose up -d
    ```
3. Start container build with `jetson-containers build ...`

### Stop the local servers
```bash
docker compose down
```

## Custom Configuration

### 1. Local PyPI configuration

The `jetson-containers` PyPI & APT servers can be pointed to any custom location using environment variables (you can use a `.env` file for that purpose).

#### Ports & PyPI mapping

The compose file provides three PyPI endpoints (one per CUDA/arch). Use the appropriate port for the CUDA + host architecture you need:

| port | CUDA   | Description |
|----------|--------|---------|
| `8126` | `12.6` | Jetson Orin (Tegra) |
| `8129` | `12.9` | Jetson Orin (Tegra) |
| `8130` | `13.0` | Jetson Thor (SBSA) |
| `8130` | `13.1` | Jetson Thor (SBSA) |

Example environment variable to point `jetson-containers` at a local PyPI server pinned to **CUDA 12.6** for **Jetson Orin**:

```bash
export LOCAL_PIP_INDEX_URL="http://localhost:8126"
```

### 2. APT / tarball server configuration

The APT server (static HTTP) is exposed on host port `8034` by default:

```bash
export LOCAL_TAR_INDEX_URL="http://localhost:8034"
```

### 3. (Optional) Configure PyPI & APT local uploads

If you want to publish your own wheels to your local PyPI and tarballs to local APT server, you also should set following variables:

```bash
# for PyPI wheels uploads
export PIP_UPLOAD_REPO="http://localhost:8126"
export PIP_UPLOAD_HOST="localhost:8126"

# for APT tarballs uploads
export SCP_UPLOAD_HOST="localhost:/home/jetson/dist/apt"
export SCP_UPLOAD_USER="jetson"
export SCP_UPLOAD_PASS="jetson"
```

> [!NOTE]  
> For APT tarballs upload the `SCP_UPLOAD_USER` and `SCP_UPLOAD_PASS` variables should point to the existing `SCP_UPLOAD_HOST` user.

## How it works?

When local servers are configured, `jetson-containers` will try registries in this order:

### When pulling a wheel or tarball

1. Local PyPI/APT servers (the docker-compose services)
2. Jetson Community registries:
    - `https://pypi.jetson-ai-lab.io` for PyPI fallback (and per-variant paths)
    - `https://apt.jetson-ai-lab.io` for APT/tarballs
3. (PyPI only) Official PyPI `https://pypi.org/simple` as a last resort

### When publishing a wheel or tarball

If you publish to your local instances, artifacts are placed in the `/home/${SCP_UPLOAD_USER}/dist/apt` home directory of  `SCP_UPLOAD_USER` and served from there. The stack does **not** automatically mirror upstream into the local directories.

## Environment Variables

| Variable | Description | Example |
|----------|---------|---------|
| `LOCAL_PIP_INDEX_URL` | Primary PyPI index URL used by builds/tooling | `http://localhost:8126` |
| `LOCAL_TAR_INDEX_URL` | Primary APT index URL used by builds/tooling | `http://localhost:8034` |
| `PIP_UPLOAD_REPO` | PyPI upload URL (and port) used for uploading wheels | `http://localhost:8126` |
| `PIP_UPLOAD_HOST` | PyPI upload host (and port) used for PIP trusted hosts | `localhost:8126` |
| `SCP_UPLOAD_HOST` | SCP target host (with path) for APT tarballs | `localhost:/dist/apt` |
| `SCP_UPLOAD_USER` | User for SCP uploads of tarballs (if applicable) | `jetson` |
| `SCP_UPLOAD_PASS` | Password for SCP uploads of tarballs (if applicable) | `jetson` |

## Troubleshooting

### Verify Configuration
```bash
# Check what pip will use in container
docker run --rm <your-container> pip config list
```

### Common problems & checks

- **Authentication failures**: if you enabled authentication on `pypiserver`, confirm the auth files and credentials are correct and mounted into the container.
- **APT artifacts not behaving like a repo**: the APT server here is a static file server — it does **not** create apt Packages indices and it is created to work only with the `jetson-containers` `tarpack` utility.

### Notes & security

- These default services are intended for local / private networks and CI environments. Do not expose them directly to the public internet without adding authentication, TLS, and network protections.
- `pypiserver` supports basic auth via `htpasswd` files — mount auth files into the container if you require authentication.
- If you need a full APT repository with indices, consider using reprepro or aptly and serve the generated repository directory instead of the simple static server.
