# GitHub Token Integration

This document describes the enhanced GitHub token integration feature that automatically handles GitHub API rate limiting during Docker builds.

## Overview

The GitHub token integration automatically detects GitHub API calls in Dockerfiles and pre-fetches the required data using authenticated API requests. This provides several benefits:

- **Higher rate limits**: Authenticated requests get 5000 requests/hour vs 60 for unauthenticated
- **More reliable builds**: No more build failures due to rate limiting
- **Better caching**: Uses `COPY` instead of `ADD` for improved Docker layer caching
- **Automatic fallback**: Falls back to the `--no-github-api` workaround if needed

## How It Works

1. **Detection**: The build system scans Dockerfiles for `ADD https://api.github.com/...` lines
2. **Authentication**: Uses GitHub token from environment variables if available
3. **Pre-fetching**: Fetches commit hashes and other data using authenticated API calls
4. **File creation**: Creates temporary JSON files with the fetched data
5. **Dockerfile modification**: Replaces `ADD` with `COPY` instructions
6. **Build integration**: Passes commit hashes as build arguments
7. **Cleanup**: Automatically removes temporary files after build completion

## Environment Variables

The system supports multiple environment variable names for GitHub tokens:

- `GITHUB_TOKEN` (primary)
- `GITHUB_PAT` (Personal Access Token)
- `GH_TOKEN` (GitHub CLI style)

Set any of these variables before running builds:

```bash
export GITHUB_TOKEN=ghp_your_token_here
./build.sh sudonim
```

## Usage Examples

### Basic Usage (with token)

```bash
# Set your GitHub token
export GITHUB_TOKEN=ghp_your_token_here

# Build normally - GitHub API calls will be automatically handled
./build.sh sudonim
./build.sh mlc vllm
```

### Fallback Usage (without token)

```bash
# If no token is set, the system will warn you but continue
./build.sh sudonim

# Or explicitly disable GitHub API usage
./build.sh --no-github-api sudonim
```

### Multiple Packages

```bash
# Build multiple packages with automatic GitHub API handling
./build.sh --multiple sudonim mlc vllm

# Or chain them together
./build.sh sudonim mlc vllm
```

## Build Arguments

When GitHub API calls are pre-processed, the commit hashes are automatically added as build arguments:

- `GITHUB_DUSTY_NV_SUDONIM_COMMIT`: Latest commit SHA for dusty-nv/sudonim
- `GITHUB_DUSTY_NV_MLC_COMMIT`: Latest commit SHA for dusty-nv/mlc
- etc.

These can be used in Dockerfiles or build scripts for version tracking.

## Technical Details

### File Structure

During preprocessing, the system creates:

```
package_directory/
├── Dockerfile                    # Original Dockerfile
├── Dockerfile.with-github-data  # Modified Dockerfile (temporary)
└── .github-api-temp/            # Temporary data directory
    ├── dusty_nv_sudonim_main.json
    └── dusty_nv_mlc_main.json
```

### Dockerfile Transformation

**Before (original):**
```dockerfile
ADD https://api.github.com/repos/dusty-nv/sudonim/git/refs/heads/main /tmp/sudonim_version.json
```

**After (processed):**
```dockerfile
COPY .github-api-temp/dusty_nv_sudonim_main.json /tmp/sudonim_version.json
```

### Error Handling

- **Token missing**: System continues with unauthenticated requests (may hit rate limits)
- **API failures**: Falls back to original Dockerfile behavior
- **Preprocessing errors**: Logs warnings and continues with original approach

## Troubleshooting

### Rate Limit Errors

If you still see rate limit errors:

1. **Check token**: Verify your GitHub token is set correctly
2. **Token permissions**: Ensure token has `repo` access for private repositories
3. **Token expiration**: GitHub tokens can expire; generate a new one if needed

### Build Failures

If builds fail with GitHub API issues:

1. **Use fallback**: Add `--no-github-api` flag
2. **Check logs**: Look for GitHub API preprocessing messages
3. **Verify connectivity**: Ensure your system can reach GitHub's API

### Debug Mode

Enable verbose logging to see detailed GitHub API processing:

```bash
./build.sh --verbose --log-level=debug sudonim
```

## Migration from --no-github-api

The `--no-github-api` flag is still supported and works as before. The new integration:

- **Enhances** the existing system rather than replacing it
- **Automatically** handles GitHub API calls when possible
- **Falls back** to the original workaround when needed
- **Maintains** backward compatibility

## Future Enhancements

Potential improvements for future versions:

- **Caching**: Cache GitHub API responses to reduce API calls
- **Batch processing**: Process multiple repositories in single API calls
- **Webhook integration**: Trigger rebuilds on repository updates
- **Rate limit monitoring**: Track and report API usage

## Contributing

To contribute to this feature:

1. **Test thoroughly**: Ensure your changes work with various GitHub API scenarios
2. **Handle errors gracefully**: Always provide fallback behavior
3. **Update documentation**: Keep this document current with any changes
4. **Follow patterns**: Use the existing logging and error handling patterns
