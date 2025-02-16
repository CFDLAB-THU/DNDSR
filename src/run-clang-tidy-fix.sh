#!/bin/bash

# if command -v clang-tidy-19 >/dev/null 2>&1; then
#     echo "clang-tidy-19 is installed."
# else
#     echo "error: clang-tidy-19 is not installed."
#     return 1
# fi

CURRENT_DIR="$(pwd)"

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# cd $SCRIPT_DIR

# Directories to search for C++ files relative to the script position
DIRS=(\
"DNDS" \
"Solver" \
"Geom" \
"CFV"  \
"Euler" \
 )

if [[ $# -gt 0 ]]; then
  IFS=',' read -r -a DIRS <<< "$1"
fi
echo "DIRS=${DIRS[@]}"

# CHECKS="*"

# CHECKS+=,-clang-diagnostic-unused-command-line-argument
# CHECKS+=,-google-readability-namespace-comments
# CHECKS+=,-modernize-use-trailing-return-type

# Loop through each directory in the array
for DIR in "${DIRS[@]}"; do
  # Check if the directory exists
  if [ -d "$SCRIPT_DIR/$DIR" ]; then
    # Find all C++ source files (.cpp, .hpp) in the directory
    FILES=$(find "$SCRIPT_DIR/$DIR" -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.hxx" )
    
    # Run clang-format on each found file
    for FILE in $FILES; do
      echo "Clang-tidy: $FILE"
      clang-tidy -p=$SCRIPT_DIR/../build  --config-file=$SCRIPT_DIR/.clang-tidy-fix --fix-errors "$FILE"
      # clang-tidy -p=$SCRIPT_DIR/../build "$FILE"
    done
    # echo "Clang-tidy: $FILES"
    # echo "$FILES" | xargs clang-tidy -p=$SCRIPT_DIR/../build --checks=$CHECKS
  else
    echo "Directory $SCRIPT_DIR/$DIR does not exist."
  fi
done


echo "Tidying complete."

# cd "$CURRENT_DIR" || exit

