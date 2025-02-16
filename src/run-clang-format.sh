#!/bin/bash

if command -v clang-format-19 >/dev/null 2>&1; then
    echo "clang-format-19 is installed."
else
    echo "error: clang-format-19 is not installed."
    return 1
fi

CURRENT_DIR="$(pwd)"

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# cd $SCRIPT_DIR

# Directories to search for C++ files relative to the script position
DIRS=(\
"DNDS" \
"Solver" \
"Geom" \
"CFV" \
"Euler" \
)

# Loop through each directory in the array
for DIR in "${DIRS[@]}"; do
  # Check if the directory exists
  if [ -d "$SCRIPT_DIR/$DIR" ]; then
    # Find all C++ source files (.cpp, .hpp) in the directory
    FILES=$(find "$SCRIPT_DIR/$DIR" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.hxx" \))
    
    # Run clang-format on each found file
    for FILE in $FILES; do
      echo "Formatting: $FILE"
      clang-format-19 -i "$FILE" 
    done
  else
    echo "Directory $SCRIPT_DIR/$DIR does not exist."
  fi
done

echo "Formatting complete."

# cd "$CURRENT_DIR" || exit

