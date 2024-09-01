#!/bin/bash

# Enable error handling
set -e

requirements_file="$1"

# Check if requirements.txt exists
if [[ ! -f "$requirements_file" ]]; then
    echo "Requirements file not found!"
    exit 1
fi

# Check if pip3 is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Installing pip3..."
    sudo apt update && sudo apt install -y python3-pip
    if [[ $? -ne 0 ]]; then
        echo "Failed to install pip3. Please install it manually."
        exit 1
    fi
fi

# Install development tools for building Python packages
echo "Installing development tools..."
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev || { echo "Failed to install development tools. Exiting..."; exit 1; }

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools || { echo "Failed to upgrade pip and setuptools. Exiting..."; exit 1; }

# Read dependencies from requirements.txt
dependencies=$(grep -E '^[^#]' "$requirements_file" | awk '{print $1}' | grep -v '^--')

for dependency in $dependencies; do
    # Extract the package name (remove any version specifiers)
    package_name=$(echo "$dependency" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1)

    # Check if the package is installed using pip
    if python3 -m pip show "$package_name" &> /dev/null; then
        echo "$package_name is already installed in the system. Continuing..."
    else
        # If not installed, install it using pip
        echo "Installing $package_name..."
        sudo -H python3 -m pip install --quiet --no-input --no-cache-dir "$package_name" || { echo "Failed to install $package_name. Exiting..."; exit 1; }
        echo "$package_name has been installed successfully. Continuing..."
    fi
done
