import logging
import sys

from polyphy.lib.get_opts import parser


def main():
    """This function launches the polyphy."""
    if len(sys.argv) == 1:
        # if no command line args are passed, show the help options
        parser.parse_args(['-h'])

    else:

        # parse them
        args = parser.parse_args()


if __name__ == "__main__":
    main()
