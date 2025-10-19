#!/bin/bash
# XSigma CI - On-Demand Clang Version Installation Script
# This script installs a specific version of Clang without conflicts
# Usage: ./install-clang-version.sh <version> [--with-llvm-tools]
# Example: ./install-clang-version.sh 16 --with-llvm-tools

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

# Check arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <clang_version> [--with-llvm-tools]"
    log_info "Example: $0 16 --with-llvm-tools"
    exit 1
fi

CLANG_VERSION=$1
WITH_LLVM_TOOLS=false

# Parse optional arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-llvm-tools)
            WITH_LLVM_TOOLS=true
            shift
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

log_info "Installing Clang version $CLANG_VERSION..."
log_info "LLVM tools: $WITH_LLVM_TOOLS"

# Validate Clang version
if ! [[ "$CLANG_VERSION" =~ ^[0-9]+$ ]]; then
    log_error "Invalid Clang version: $CLANG_VERSION (must be a number)"
    exit 1
fi

# Update package manager
log_info "Updating package manager..."
sudo apt-get update || {
    log_error "Failed to update package manager"
    exit 1
}

# Add LLVM repository for newer Clang versions if needed
if [ "$CLANG_VERSION" -ge 15 ]; then
    log_info "Adding LLVM repository for Clang $CLANG_VERSION..."
    
    # Install dependencies for adding repository
    sudo apt-get install -y lsb-release wget software-properties-common gnupg || {
        log_warning "Failed to install repository management tools"
    }
    
    # Add LLVM repository
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - || {
        log_warning "Failed to add LLVM GPG key (may already exist)"
    }
    
    UBUNTU_VERSION=$(lsb_release -cs)
    echo "deb http://apt.llvm.org/$UBUNTU_VERSION/ llvm-toolchain-$UBUNTU_VERSION-$CLANG_VERSION main" | \
        sudo tee /etc/apt/sources.list.d/llvm-$CLANG_VERSION.list > /dev/null || {
        log_warning "Failed to add LLVM repository (may already exist)"
    }
    
    # Update package manager again after adding repository
    sudo apt-get update || {
        log_warning "Failed to update package manager after adding LLVM repository"
    }
fi

# Install Clang and related tools
log_info "Installing Clang $CLANG_VERSION and related tools..."
PACKAGES="clang-$CLANG_VERSION clang++-$CLANG_VERSION"

if [ "$WITH_LLVM_TOOLS" = true ]; then
    PACKAGES="$PACKAGES llvm-$CLANG_VERSION llvm-$CLANG_VERSION-dev"
fi

sudo apt-get install -y $PACKAGES || {
    log_error "Failed to install Clang $CLANG_VERSION"
    exit 1
}

# Create symbolic links for easy access (optional)
log_info "Creating symbolic links for Clang $CLANG_VERSION..."
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-$CLANG_VERSION 100 || {
    log_warning "Failed to create symbolic link for clang"
}
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$CLANG_VERSION 100 || {
    log_warning "Failed to create symbolic link for clang++"
}

if [ "$WITH_LLVM_TOOLS" = true ]; then
    sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-$CLANG_VERSION 100 || {
        log_warning "Failed to create symbolic link for llvm-config"
    }
fi

# Verify installation
log_info "Verifying Clang $CLANG_VERSION installation..."
if command -v clang-$CLANG_VERSION &> /dev/null; then
    log_success "Clang $CLANG_VERSION installed successfully!"
    clang-$CLANG_VERSION --version
else
    log_error "Clang $CLANG_VERSION installation verification failed"
    exit 1
fi

log_success "Clang $CLANG_VERSION installation completed successfully!"

