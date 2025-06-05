#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${KAI_SCHEDULER_VERSION} --depth=1 --recursive https://github.com/NVIDIA/KAI-Scheduler /opt/kai_scheduler || \
git clone --depth=1 --recursive https://github.com/NVIDIA/KAI-Scheduler /opt/kai_scheduler

export MAX_JOBS=$(nproc)
cd /opt/kai_scheduler
make
helm package ./deployments/kai-scheduler -d ./charts
helm upgrade -i kai-scheduler -n kai-scheduler --create-namespace ./charts/kai-scheduler-0.0.0.tgz
