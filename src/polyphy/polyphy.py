# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

from utils.cli_helper import CliHelper


def main():
    CliHelper.parse_args()
    ppConfig = None
    CliHelper.parse_values(ppConfig)


if __name__ == "__main__":
    main()
