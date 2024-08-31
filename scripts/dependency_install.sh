#!/bin/bash

requirements_file="$1"

# Check if requirements.txt exists
if [[ ! -f "$requirements_file" ]]; then
    echo "Requirements file not found!"
    exit 1
fi

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
        sudo -H python3 -m pip install --quiet --no-input "$package_name"
        if [[ $? -eq 0 ]]; then
            echo "$package_name has been installed successfully. Continuing..."
        else
            echo "Failed to install $package_name. Please check the error messages above."
        fi
    fi
done





# python_file="$1"

# # Extract dependencies from the Python file
# dependencies=$(grep -E '^\s*import\s+|^from\s+\w+\s+import' "$python_file" | awk '{print $NF}')

# # Check for `from dotenv import load_dotenv`
# if grep -q 'from dotenv import load_dotenv' "$python_file"; then
#     dependencies+=" python-dotenv"
# fi

# for dependency in $dependencies; do
#     # Check if the dependency is a system package
#     if python3 -c "import $dependency" &> /dev/null; then
#         echo "$dependency is already inside the system. Continuing..."
#     else
#         # If not a system package, install it using pip
#         sudo -H python3 -m pip install --quiet --no-input "$dependency"
#         echo "$dependency has been installed successfully. Continuing..."
#     fi
# done
