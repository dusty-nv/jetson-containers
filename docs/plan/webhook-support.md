# Webhook on Build Finish Plan

This document outlines a plan to implement a webhook notification system that triggers upon the completion of a build process in `jetson-containers`.

## 1. Objective

To notify an external service or user when a build process finishes, providing the final status (success or failure) and other relevant build metadata.

## 2. Proposed Solution

The solution involves using a webhook, which will send an HTTP POST request to a user-defined URL at the end of each build.

### Key Components:

- **Configuration**: The webhook URL will be supplied via a `WEBHOOK_URL` environment variable. The script will silently skip the webhook process if this variable is not defined.
- **Trigger**: The webhook will be triggered from the main build script, `jetson_containers/build.py`, after the build process for all specified packages has concluded. It will execute in all cases, whether the build succeeds or fails.
- **Dependency**: The `requests` Python library will be used to send the HTTP POST request. It will be added to `requirements.txt`.
- **Payload**: The request will contain a JSON body with essential information about the build.

### Example JSON Payload:

```json
{
  "status": "success",
  "end_time": "2025-09-04T18:30:00Z",
  "packages": [
    "ros:humble-desktop",
    "pytorch:2.1"
  ],
  "log_file": "/path/to/logs/build_20250904.log"
}
```

## 3. Implementation Plan

1.  **Add Dependency**:
    - Add `requests` to the `requirements.txt` file.

2.  **Create Webhook Utility Function**:
    - In `jetson_containers/utils.py`, create a new function `send_webhook(status, packages, log_file)`.
    - This function will:
        a. Read the `WEBHOOK_URL` from the environment. If it's not set, return immediately.
        b. Get the current UTC timestamp for the `end_time`.
        c. Construct the JSON payload dictionary.
        d. Use `requests.post()` to send the data to the `WEBHOOK_URL`.
        e. Wrap the request in a `try...except` block to catch and log any network errors, ensuring that a webhook failure does not cause the main script to fail.

3.  **Integrate into Build Script**:
    - In `jetson_containers/build.py`, locate the main function that orchestrates the build process.
    - Wrap the core build logic in a `try...finally` block.
    - Inside the `try` block, track the build status. Initialize `status = 'success'`. If any exception occurs, set `status = 'failure'`.
    - In the `finally` block, call the `utils.send_webhook()` function, passing the determined `status`, the list of packages, and the path to the build log.

## 4. Example Usage

Once implemented, a user can enable the webhook by setting the environment variable before running the build:

```bash
export WEBHOOK_URL="https://my-service.com/build-notifications"
./build.sh ros:humble-desktop pytorch:2.1
```
