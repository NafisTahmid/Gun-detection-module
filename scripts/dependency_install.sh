#!/bin/bash

python_file="$1"
dependencies=$(grep -E '^\s*import\s+|^from\s+\w+\s+import' "$python_file" | awk '{print $NF}')

for dependency in $dependencies; do
    # Check if the dependency is a system package
    if python3 -c "import $dependency" &> /dev/null; then
        echo "$dependency is already inside the system. Continuning..."
    else
        # If not a system package, install it using pip
        sudo -H python3 -m pip install --quiet --no-input "$dependency"
        echo "$dependency has been installed succesfully. Continuing..."
    fi
done
