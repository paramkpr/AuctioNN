#!/bin/bash
# RunPod Setup Script for AuctioNN using pyenv
# Usage: ./setup_runpod.sh <pod_ip> <pod_port>

set -e  # Exit on error

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: ./setup_runpod.sh <pod_ip> <pod_port>"
    exit 1
fi

POD_IP=$1
POD_PORT=$2
KEY_PATH="./keys/runpod/id_rsa"
REPO_URL="https://github.com/paramkpr/AuctioNN"
DATA_PATH="./data/clean_data.parquet"
REMOTE_PATH="/root/AuctioNN/data/clean_data.parquet"
PYTHON_VERSION="3.13.0"

# Validate key exists
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: SSH key not found at $KEY_PATH"
    exit 1
fi

# Ensure key has correct permissions
chmod 600 "$KEY_PATH"

echo "==== Setting up RunPod instance at $POD_IP:$POD_PORT ===="

# Check if repo already exists
echo "Checking if repository already exists..."
REPO_EXISTS=$(ssh -i "$KEY_PATH" -p "$POD_PORT" -o StrictHostKeyChecking=no root@"$POD_IP" "[ -d /root/AuctioNN ] && echo 'exists' || echo 'not_exists'")
if [ "$REPO_EXISTS" = "exists" ]; then
    echo "Repository already exists, pulling latest changes..."
    ssh -i "$KEY_PATH" -p "$POD_PORT" root@"$POD_IP" "cd /root/AuctioNN && git pull"
else
    # Clone repository to the pod
    echo "Cloning repository to pod..."
    ssh -i "$KEY_PATH" -p "$POD_PORT" -o StrictHostKeyChecking=no root@"$POD_IP" "git clone $REPO_URL"
fi

# Check if data file already exists
echo "Checking if data file already exists..."
FILE_EXISTS=$(ssh -i "$KEY_PATH" -p "$POD_PORT" root@"$POD_IP" "[ -f $REMOTE_PATH ] && echo 'exists' || echo 'not_exists'")
if [ "$FILE_EXISTS" = "exists" ]; then
    echo "Data file already exists, skipping copy..."
else
    # Copy data file to the pod
    echo "Copying data file to pod..."
    # Create directory if it doesn't exist
    ssh -i "$KEY_PATH" -p "$POD_PORT" root@"$POD_IP" "mkdir -p /root/AuctioNN/data"
    scp -i "$KEY_PATH" -P "$POD_PORT" "$DATA_PATH" root@"$POD_IP":"$REMOTE_PATH"
fi

# Install Python 3.13 using pyenv, create virtual environment and install dependencies
echo "Setting up Python environment using pyenv and installing dependencies..."
ssh -i "$KEY_PATH" -p "$POD_PORT" root@"$POD_IP" << EOF
# Set noninteractive mode for apt
export DEBIAN_FRONTEND=noninteractive

# Set timezone automatically to avoid prompts
ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
dpkg-reconfigure --frontend noninteractive tzdata

# Install dependencies for pyenv and Python installation
apt-get update
apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev git

# Check if pyenv is already installed
if [ ! -d "/root/.pyenv" ]; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
fi

# Add pyenv to PATH and initialize
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv init --path)"
eval "\$(pyenv init -)"

# Check if Python $PYTHON_VERSION is already installed with pyenv
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION with pyenv..."
    pyenv install $PYTHON_VERSION
fi

# Set Python $PYTHON_VERSION as the local version for the AuctioNN directory
cd /root/AuctioNN
pyenv local $PYTHON_VERSION
python --version

# Check if virtual environment already exists
if [ ! -d "/root/AuctioNN/.venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment and install dependencies
source .venv/bin/activate
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Add pyenv initialization to .bashrc if not already there
if ! grep -q "pyenv init" /root/.bashrc; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc
fi

# Set up automatic activation of Python environment when entering the directory
if [ ! -f "/root/AuctioNN/.envrc" ]; then
    echo "Setting up automatic environment activation..."
    echo 'source .venv/bin/activate' > /root/AuctioNN/.envrc
    echo 'cd /root/AuctioNN && source .venv/bin/activate' >> /root/.bashrc
fi

echo "Python setup complete! Using Python version: \$(python --version)"
EOF

echo "==== RunPod setup completed successfully! ===="
echo "To connect to your pod, use: ssh -i $KEY_PATH -p $POD_PORT root@$POD_IP"
echo "The virtual environment is at /root/AuctioNN/.venv"
echo "It will be activated automatically when you log in"