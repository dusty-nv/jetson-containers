#!/usr/bin/env python3
import os
import requests
from datetime import datetime, timezone
from typing import Dict, List

from .logging import log_error, log_warning, log_verbose


def _format_env_vars(env_vars: Dict[str, str]) -> List[str]:
    """
    Format environment variables into a list of key=value strings.
    
    Args:
        env_vars (Dict[str, str]): Environment variables dictionary
        
    Returns:
        List[str]: Formatted environment variable strings
    """
    env_parts = []
    for key in ['CUDA_VERSION', 'LSB_RELEASE', 'PYTHON_VERSION']:
        if key in env_vars and env_vars[key]:
            env_parts.append(f"{key}={env_vars[key]}")
    return env_parts


def _format_env_vars_for_embed(env_vars: Dict[str, str]) -> List[str]:
    """
    Format environment variables for Discord embed fields.
    
    Args:
        env_vars (Dict[str, str]): Environment variables dictionary
        
    Returns:
        List[str]: Formatted environment variable strings for embed
    """
    env_parts = []
    for key in ['CUDA_VERSION', 'LSB_RELEASE', 'PYTHON_VERSION']:
        if key in env_vars and env_vars[key]:
            env_parts.append(f"{key}: {env_vars[key]}")
    return env_parts


def _get_status_info(status: str) -> tuple:
    """
    Get status emoji, color, and text based on build status.
    
    Args:
        status (str): Build status ('success' or 'failure')
        
    Returns:
        tuple: (emoji, color, status_text)
    """
    if status == 'success':
        return '✅', 0x00ff00, 'Build Successful'  # Green
    else:
        return '❌', 0xff0000, 'Build Failed'  # Red


def _format_build_command(build_command: str, max_length: int = 100) -> str:
    """
    Format build command with truncation if needed.
    
    Args:
        build_command (str): The build command to format
        max_length (int): Maximum length before truncation
        
    Returns:
        str: Formatted command string
    """
    if len(build_command) > max_length:
        return f"\n💻 **Command:** `{build_command[:max_length-3]}...`"
    else:
        return f"\n💻 **Command:** `{build_command}`"


def _create_discord_payload(status: str, packages: List[str], end_time: str, 
                           build_command: str = None, env_vars: Dict[str, str] = None, 
                           message: str = None) -> Dict:
    """
    Create Discord webhook payload.
    
    Args:
        status (str): Build status
        packages (List[str]): List of packages
        end_time (str): End time timestamp
        build_command (str, optional): Build command
        env_vars (Dict[str, str], optional): Environment variables
        message (str, optional): Additional message
        
    Returns:
        Dict: Discord webhook payload
    """
    emoji, color, status_text = _get_status_info(status)
    packages_str = ', '.join(packages) if packages else 'No packages'
    
    # Format environment variables
    env_info = ""
    if env_vars:
        env_parts = _format_env_vars(env_vars)
        if env_parts:
            env_info = f"\n🔧 **Environment:** {', '.join(env_parts)}"
    
    # Format build command
    cmd_info = _format_build_command(build_command) if build_command else ""
    
    # Create content
    content = f"{emoji} **{status_text}**\n📦 **Packages:** {packages_str}\n⏰ **Time:** {end_time}{cmd_info}{env_info}"
    
    if message:
        # Limit message length for Discord (max 2000 characters total)
        max_msg_len = 1500 - len(content)
        if len(message) > max_msg_len:
            message = message[:max_msg_len] + "..."
        content += f"\n📝 **Details:**\n```\n{message}\n```"
    
    # Create base payload
    payload = {
        'content': content,
        'embeds': [{
            'title': f'Jetson Containers Build {status.title()}',
            'description': f'Packages: {packages_str}',
            'color': color,
            'timestamp': end_time,
            'fields': [
                {
                    'name': 'Status',
                    'value': status_text,
                    'inline': True
                },
                {
                    'name': 'Package Count',
                    'value': str(len(packages)),
                    'inline': True
                }
            ]
        }]
    }
    
    # Add build command field if available
    if build_command:
        cmd_display = build_command if len(build_command) <= 1000 else f"{build_command[:997]}..."
        payload['embeds'][0]['fields'].append({
            'name': 'Build Command',
            'value': f"```bash\n{cmd_display}\n```",
            'inline': False
        })
    
    # Add environment variables field if available
    if env_vars:
        env_parts = _format_env_vars_for_embed(env_vars)
        if env_parts:
            payload['embeds'][0]['fields'].append({
                'name': 'Environment',
                'value': '\n'.join(env_parts),
                'inline': True
            })
    
    # Add message field if available
    if message:
        payload['embeds'][0]['fields'].append({
            'name': 'Details',
            'value': f"```\n{message[:1000]}{'...' if len(message) > 1000 else ''}\n```",
            'inline': False
        })
    
    return payload


def _create_generic_payload(status: str, packages: List[str], end_time: str,
                           build_command: str = None, env_vars: Dict[str, str] = None,
                           message: str = None) -> Dict:
    """
    Create generic webhook payload.
    
    Args:
        status (str): Build status
        packages (List[str]): List of packages
        end_time (str): End time timestamp
        build_command (str, optional): Build command
        env_vars (Dict[str, str], optional): Environment variables
        message (str, optional): Additional message
        
    Returns:
        Dict: Generic webhook payload
    """
    payload = {
        'status': status,
        'end_time': end_time,
        'packages': packages,
    }
    
    if build_command:
        payload['build_command'] = build_command
        
    if env_vars:
        payload['environment'] = env_vars
    
    if message:
        payload['message'] = message
        
    return payload


def send_webhook(status: str, packages: List[str], message: str = None, 
                 build_command: str = None, env_vars: Dict[str, str] = None,
                 webhook_url: str = None):
    """
    Sends a webhook notification with build completion status.

    Args:
        status (str): Build status, either 'success' or 'failure'
        packages (List[str]): List of packages that were built
        message (str, optional): Status message or error details
        build_command (str, optional): The command used for the build
        env_vars (Dict[str, str], optional): Environment variables (CUDA_VERSION, LSB_RELEASE, PYTHON_VERSION)
        webhook_url (str, optional): The webhook URL to send the notification to

    Returns:
        None
    """
    if not webhook_url:
        log_verbose("No webhook URL provided, skipping webhook notification")
        return
    
    try:
        # Get current UTC timestamp
        end_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Create payload based on webhook type
        if 'discord.com' in webhook_url.lower():
            payload = _create_discord_payload(status, packages, end_time, build_command, env_vars, message)
        else:
            payload = _create_generic_payload(status, packages, end_time, build_command, env_vars, message)
        
        log_verbose(f"Sending webhook notification to {webhook_url}")
        
        # Send POST request
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        log_verbose(f"Webhook notification sent successfully (status: {response.status_code})")
        
    except Exception as e:
        log_warning(f"Failed to send webhook notification: {e}")
        # Don't raise the exception - webhook failure should not cause build failure
