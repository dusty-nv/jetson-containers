#!/usr/bin/env bash

echo "getting protobuf API implementation..."
echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = $PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
echo ""

echo "getting protobuf Python package info..."
pip3 show protobuf
echo ""

echo "getting protobuf compiler version..."
protoc --version
echo ""
