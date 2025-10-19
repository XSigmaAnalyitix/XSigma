#!/bin/bash
# XSigma CI - Ubuntu/Linux Dependency Installation Script
# This script installs all required dependencies for building XSigma on Ubuntu/Linux
# Usage: ./install-deps-ubuntu.sh [--with-cuda] [--with-tbb]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
WITH_CUDA=false
WITH_TBB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-cuda)
            WITH_CUDA=true
            shift
            ;;
        --with-tbb)
            WITH_TBB=true
            shift
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

log_info "Starting Ubuntu/Linux dependency installation..."
log_info "CUDA support: $WITH_CUDA"
log_info "TBB support: $WITH_TBB"

# Update package manager
log_info "Updating package manager..."
sudo apt-get update || {
    log_error "Failed to update package manager"
    exit 1
}

# Core build tools
log_info "Installing core build tools..."
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    || {
        log_error "Failed to install core build tools"
        exit 1
    }

# Clang compiler (default version only)
# Note: Specific Clang versions are installed on-demand via install-clang-version.sh
# to avoid package conflicts when multiple versions are needed
log_info "Installing default Clang compiler..."
sudo apt-get install -y \
    clang \
    clang++ \
    || {
        log_error "Failed to install default Clang"
        exit 1
    }

# GCC compiler (for compiler version testing)
log_info "Installing GCC compiler..."
sudo apt-get install -y \
    gcc \
    g++ \
    || {
        log_error "Failed to install GCC"
        exit 1
    }

# Common libraries
log_info "Installing common libraries..."
sudo apt-get install -y \
    libssl-dev \
    zlib1g-dev \
    libnuma-dev \
    || {
        log_error "Failed to install common libraries"
        exit 1
    }

# TBB (Threading Building Blocks)
if [ "$WITH_TBB" = true ]; then
    log_info "Installing Intel TBB..."
    sudo apt-get install -y \
        libtbb-dev \
        libtbb2 \
        || {
            log_warning "Failed to install TBB from package manager"
        }
fi

# CUDA Toolkit (optional)
if [ "$WITH_CUDA" = true ]; then
    log_info "Installing CUDA Toolkit..."
    # CUDA installation is complex and platform-specific
    # For CI, we typically use pre-installed CUDA or skip it
    log_warning "CUDA installation skipped - typically pre-installed in CI environment"
fi

# Python dependencies
log_info "Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel || {
    log_warning "Failed to upgrade pip"
}

pip3 install colorama psutil || {
    log_warning "Failed to install Python dependencies"
}

log_success "Ubuntu/Linux dependency installation completed successfully!"
log_info "You can now build XSigma using: python Scripts/setup.py ninja clang config build test"

