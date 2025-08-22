# Custom PyPI Configuration

This document explains how to configure custom PyPI sources when building jetson-containers.

## Quick Start

To use a custom PyPI server, set these environment variables before building:

```bash
# Set your custom PyPI server
export LOCAL_PIP_INDEX_URL="https://your-pypi-server.com/simple"
export PIP_UPLOAD_HOST="your-pypi-server.com"

# Build with custom PyPI server
./build.sh package-name
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LOCAL_PIP_INDEX_URL` | Primary PyPI index URL | `https://pypi.example.com/simple` |
| `PIP_UPLOAD_HOST` | Hostname for SSL trust | `pypi.example.com` |
| `PIP_TRUSTED_HOSTS` | Additional trusted hosts | `pypi.example.com,pypi.org` |

## Common Use Cases

### Corporate PyPI Mirror
```bash
export LOCAL_PIP_INDEX_URL="https://pypi.corporate.com/simple"
export PIP_UPLOAD_HOST="pypi.corporate.com"
```

### Regional Mirrors
```bash
# China - Tsinghua University
export LOCAL_PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_UPLOAD_HOST="pypi.tuna.tsinghua.edu.cn"

# China - Aliyun
export LOCAL_PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
export PIP_UPLOAD_HOST="mirrors.aliyun.com"
```

### Private Repository
```bash
export LOCAL_PIP_INDEX_URL="https://pypi.internal.company.com/simple"
export PIP_UPLOAD_HOST="pypi.internal.company.com"
export PIP_TRUSTED_HOSTS="pypi.internal.company.com"
```

## How It Works

By default, jetson-containers uses:
- **Primary**: `https://pypi.jetson-ai-lab.io` (NVIDIA's cache)
- **Fallback**: `https://pypi.org/simple` (Standard PyPI)

Your custom settings override the primary cache while keeping the fallback for reliability.

## Troubleshooting

### SSL Certificate Issues
```bash
# Add to trusted hosts
export PIP_TRUSTED_HOSTS="your-host.com,pypi.org"
```

### Authentication Required
```bash
export PIP_UPLOAD_USER="your-username"
export PIP_UPLOAD_PASS="your-password"
```

### Verify Configuration
```bash
# Check what pip will use in container
docker run --rm your-container pip config list
```

The system automatically falls back to standard PyPI if your custom server is unavailable.