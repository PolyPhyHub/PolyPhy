import sys
import re


def check_python_header(file_path):
    # Define the expected header format
    expected_header = r'''# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: .*
'''

    with open(file_path, 'r') as file:
        content = file.read()
        if not re.match(expected_header, content, re.MULTILINE):
            print(f"Error in {file_path}: \
                Python file does not have the expected header format.")
            sys.exit(1)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        if filename != ".github/scripts/pre_commit_hook.py":
            check_python_header(filename)
