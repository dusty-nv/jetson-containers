#!/usr/bin/env bash

set -e

echo "testing go-lang..."

go version

go env

go run /${GOPATH}/test/test.go

echo "go-lang OK"
