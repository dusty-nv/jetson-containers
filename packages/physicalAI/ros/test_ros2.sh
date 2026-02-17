#!/usr/bin/env bash
set -euo pipefail

echo "Testing ROS2 installation"
echo "Getting ROS version -"
echo "ROS_DISTRO   $ROS_DISTRO"
echo "ROS_ROOT     $ROS_ROOT"
echo "AMENT_PREFIX_PATH $AMENT_PREFIX_PATH"

ros2 pkg list >/dev/null || { echo "❌ ros2 pkg list failed"; exit 1; }

which python; python -V
which pip
which ros2
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"

# --- Demo pub/sub ---
echo "Launching talker in background..."
ros2 run demo_nodes_cpp talker &
TALKER_PID=$!

cleanup() {
  kill $TALKER_PID 2>/dev/null || true
  wait $TALKER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2

echo "Listening for a single message on /chatter..."
if ros2 topic echo /chatter --once; then
  echo "✅ Listener received a message, test OK"
else
  echo "❌ Listener did not receive any message"
  exit 1
fi

echo "✅ ROS2 installation test complete"
