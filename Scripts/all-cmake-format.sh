
cd ..

FMT="cmake-format"

files_added=`git diff --cached --name-only | grep -E '\.cmake$' `
files_existent=`git diff --name-only | grep -E '\.cmake$' `
"${FMT}" -i $files_existent $files_added