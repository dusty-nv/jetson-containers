# Webhook Event Integration for Jetson Containers

Jetson Containers supports hooking to build-finish events via webhooks.  

See .env file for example on how to set up 

## Supported Environment Variables 

- `JC_BUILD_SUCCESS_WEBHOOK_URL`: On build success
- `JC_BUILD_FAILURE_WEBHOOK_URL`: On build failure, including `--multiple` flag where one package fails

## Supported Webhook Formats

### Discord Webhooks

Jetson Containers events have special treatment for Discord webhooks.
When a Discord channel webhook URL is provided, a formatted structure is sent:

<img width="689" height="508" alt="image" src="https://github.com/user-attachments/assets/8ffd4083-2d2b-4e18-9390-0b4b2bcbe38a" /><img width="665" height="518" alt="image" src="https://github.com/user-attachments/assets/2ea9f14c-f3dd-48b0-863d-fc38fcd0971e" />



Payload:
```json
{
  "content": "✅ **Build Successful**\n📦 **Packages:** pytorch, torchvision\n⏰ **Time:** 2025-09-07T12:34:56Z\n💻 **Command:** `./build.sh pytorch torchvision`\n🔧 **Environment:** CUDA_VERSION=12.2, LSB_RELEASE=20.04, PYTHON_VERSION=3.8",
  "embeds": [
    {
      "title": "Jetson Containers Build Success",
      "description": "Packages: pytorch, torchvision",
      "color": 65280,
      "timestamp": "2025-09-07T12:34:56Z",
      "fields": [
        {
          "name": "Status",
          "value": "Build Successful",
          "inline": true
        },
        {
          "name": "Package Count",
          "value": "2",
          "inline": true
        },
        {
          "name": "Build Command",
          "value": "```bash\n./build.sh pytorch torchvision\n```",
          "inline": false
        },
        {
          "name": "Environment",
          "value": "CUDA_VERSION: 12.2\nLSB_RELEASE: 20.04\nPYTHON_VERSION: 3.8",
          "inline": true
        },
        {
          "name": "Details",
          "value": "```\nBuild completed successfully\n```",
          "inline": false
        }
      ]
    }
  ]
}
```

### Generic Webhooks

For non-Discord webhooks, a simpler JSON structure is used:
```json
{
  "status": "success",
  "end_time": "2025-09-07T12:34:56Z",
  "packages": ["pytorch", "torchvision"],
  "build_command": "./build.sh pytorch torchvision",
  "environment": {
    "CUDA_VERSION": "12.2",
    "LSB_RELEASE": "20.04",
    "PYTHON_VERSION": "3.8"
  },
  "message": "Build completed successfully"
}
```

## Payload Fields

- **status**: Either `"success"` or `"failure"`
- **end_time**: ISO 8601 timestamp in UTC
- **packages**: Array of package names that were built
- **build_command**: The command used to trigger the build
- **environment** (optional): Key environment variables (CUDA_VERSION, LSB_RELEASE, PYTHON_VERSION)
- **details** (optional): Additional details or error messages

## Discord-Specific Features

- Color-coded embeds (green for success, red for failure)
- Emoji indicators (✅ for success, ❌ for failure)
- Rich formatting with inline fields
- Automatic truncation for Discord's message limits
- Command highlighting with syntax highlighting
