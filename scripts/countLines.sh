#!/bin/bash



FILES=" "
FILES=${FILES}" "$(tree -fi --noreport -P "*.cpp")
FILES=${FILES}" "$(tree -fi --noreport -P "*.hpp")
FILES=${FILES}" "$(tree -fi --noreport -P "*.c")
FILES=${FILES}" "$(tree -fi --noreport -P "*.h")



echo ${FILES}
wc -l ${FILES}