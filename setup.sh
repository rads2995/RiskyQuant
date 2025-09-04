#!/usr/bin/env bash

export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HCC_AMDGPU_TARGET=gfx1010

source /opt/rocm/share/rocprofiler-register/setup-env.sh
