#!/bin/bash

# Check if ninja command is available
if ! command -v ninja &> /dev/null
then
    echo "Ninja is not installed or not in the PATH."
    exit 1
fi

# Create a simple build.ninja file to test Ninja
echo -e "rule echo\n  command = echo Hello, Ninja!\nbuild test: echo" > build.ninja

# Run Ninja to test the build
if ninja
then
    echo "Ninja is installed and working correctly."
    # Clean up
    rm -f build.ninja
    exit 0
else
    echo "Ninja failed to run correctly."
    # Clean up
    rm -f build.ninja
    exit 1
fi
