#!/bin/bash

# Define the project directory
PROJECT_DIR="$HOME/MIAI"

# Determine the shell configuration file (.bashrc or .zshrc)
SHELL_RC="$HOME/.bashrc"
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
fi

# Ensure the project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: MIAI project directory not found at $PROJECT_DIR"
    exit 1
fi

# Append PYTHONPATH to .bashrc if it's not already added
if ! grep -q "$PROJECT_DIR" "$SHELL_RC"; then
    echo "Appending MIAI to PYTHONPATH in $SHELL_RC..."
    echo "" >> "$SHELL_RC"
    echo "# Add MIAI project to PYTHONPATH" >> "$SHELL_RC"
    echo "export PYTHONPATH=\$PYTHONPATH:$PROJECT_DIR" >> "$SHELL_RC"
    echo "✅ Successfully added MIAI to PYTHONPATH in $SHELL_RC"
fi

# Source .bashrc or .zshrc to apply changes immediately
echo "Applying changes..."
source "$SHELL_RC"

# Verify PYTHONPATH
echo "🚀 Current PYTHONPATH: $PYTHONPATH"
