zip -r7 \
    DNDSR_pack.zip \
    src app test tools\
    python cases CMakeLists.txt external cmake docs \
    cmakeCommonUtils.cmake \
    README.md \
    -x "*/*.git/*" \
    -x "node_modules/*" \
    -x "*.log" \
    -x ".DS_Store" \
    -x "__pycache__/*" \
    -x "*/CMakeFiles/*" \
    -x "external/*/bin/*" \
    