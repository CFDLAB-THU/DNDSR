#!/bin/bash

DIR_NAME=$(dirname ${BASH_SOURCE[0]})
export PATH="$(realpath ${DIR_NAME}):${PATH}"
export PATH="$(realpath ${DIR_NAME}/../build/app):${PATH}"