# Webhook on Build Finish Implementation

This document outlines the implemented webhook notification system that triggers upon the completion of a build process in `jetson-containers`.

## 1. Objective

To notify an external service or user when a build process finishes, providing the final status (success or failure) and other relevant build metadata through various webhook formats.

## 2. Implementation

The solution uses a webhook that sends an HTTP POST request to a user-defined URL at the end of each build. The system supports multiple webhook formats with automatic detection.

### Key Components:

- **Configuration**: The webhook URL is supplied via a `WEBHOOK_URL` environment variable. The script silently skips the webhook process if this variable is not defined.
- **Format Detection**: The system automatically detects webhook format based on URL:
  - Discord webhooks (discord.com/api/webhooks)
  - Slack webhooks (hooks.slack.com)
  - Microsoft Teams (outlook.office.com/office365.com)
  - Generic JSON format (all others)
- **Manual Override**: Use `WEBHOOK_FORMAT` environment variable to force a specific format (`discord`, `slack`, `teams`, `generic`)
- **Trigger**: The webhook is triggered from the main build script, `jetson_containers/build.py`, after the build process concludes
- **Error Handling**: Webhook failures do not cause build failures
- **Message Content**: For successful builds, sends status message. For failures, includes last 10 lines of error logs

### Example Payloads:

#### Generic Format:
```json
{
  "status": "success",
  "end_time": "2025-09-06T18:30:00Z",
  "packages": ["ros:humble-desktop", "pytorch:2.1"],
  "message": "Build completed successfully"
}
```

For failures:
```json
{
  "status": "failure", 
  "end_time": "2025-09-06T18:30:00Z",
  "packages": ["ros:humble-desktop"],
  "message": "ERROR: Package build failed\nDocker build command failed\n..."
}
```

#### Discord Format:
```json
{
  "content": "✅ **Build SUCCESS** - ros:humble-desktop, pytorch:2.1\n⏰ Completed at: 2025-09-06T18:30:00Z"
}
```

For failures:
```json
{
  "content": "❌ **Build FAILURE** - ros:humble-desktop\n```\nERROR: Package build failed\nDocker build command failed\n```\n⏰ Completed at: 2025-09-06T18:30:00Z"
}
```

#### Slack Format:
```json
{
  "attachments": [
    {
      "color": "good",
      "title": ":white_check_mark: Build SUCCESS",
      "fields": [
        {
          "title": "Packages",
          "value": "ros:humble-desktop, pytorch:2.1",
          "short": false
        },
        {
          "title": "Completed", 
          "value": "2025-09-06T18:30:00Z",
          "short": true
        }
      ]
    }
  ]
}
```

## 3. Supported Webhook Services

### Discord
- **Auto-detection**: URLs containing `discord.com/api/webhooks`
- **Manual override**: `WEBHOOK_FORMAT=discord`
- **Features**: Rich text formatting with emojis and code blocks
- **Message limit**: 1500 characters for error details

### Slack
- **Auto-detection**: URLs containing `hooks.slack.com`
- **Manual override**: `WEBHOOK_FORMAT=slack`
- **Features**: Colored attachments and structured fields
- **Colors**: Green for success, red for failure

### Microsoft Teams
- **Auto-detection**: URLs containing `outlook.office.com` or `office365.com`
- **Manual override**: `WEBHOOK_FORMAT=teams`
- **Features**: MessageCard format with themed colors
- **Theme**: Green (#00FF00) for success, red (#FF0000) for failure

### Generic/Custom
- **Default**: Any other URL
- **Manual override**: `WEBHOOK_FORMAT=generic`
- **Features**: Simple JSON format, suitable for custom integrations
- **Content**: Full error messages without truncation

## 4. Usage Examples

### Basic Usage (Auto-detection)
```bash
# Discord webhook
export WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"
./build.sh ros:humble-desktop pytorch:2.1

# Slack webhook  
export WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
./build.sh ros:humble-desktop pytorch:2.1

# Teams webhook
export WEBHOOK_URL="https://outlook.office.com/webhook/YOUR_TEAMS_WEBHOOK"
./build.sh ros:humble-desktop pytorch:2.1

# Custom/generic webhook
export WEBHOOK_URL="https://your-service.com/webhook"
./build.sh ros:humble-desktop pytorch:2.1
```

### Force Specific Format
```bash
export WEBHOOK_URL="https://your-custom-service.com/webhook"
export WEBHOOK_FORMAT="generic"
./build.sh ros:humble-desktop pytorch:2.1

# Or force Discord format even for non-Discord URLs
export WEBHOOK_URL="https://your-service.com/webhook"
export WEBHOOK_FORMAT="discord"
./build.sh ros:humble-desktop pytorch:2.1
```

### Testing Webhook
```bash
# Test with a simple build that should succeed
export WEBHOOK_URL="YOUR_WEBHOOK_URL"
./build.sh --skip-tests=all --name test-webhook cudnn

# Test with a build that will fail (non-existent package)
export WEBHOOK_URL="YOUR_WEBHOOK_URL"
./build.sh non-existent-package
```

## 5. Implementation Details

### Files Modified:
- `requirements.txt`: Added `requests` dependency
- `jetson_containers/network.py`: Added webhook functions
- `jetson_containers/build.py`: Added webhook integration
- `docs/plan/webhook-support.md`: Original plan documentation

### Error Handling:
- Webhook failures are logged but don't affect build success/failure
- Network timeouts are handled gracefully
- Malformed webhook URLs are caught and logged
- Missing log files are handled without errors

### Security Considerations:
- Webhook URLs should be kept secure (contain sensitive tokens)
- Error messages in webhooks may contain build details
- No sensitive system information is included in webhook payloads
