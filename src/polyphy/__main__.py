import sys

from .lib.get_opts import parser
from .lib.runner import run2d, run3d


def main():
    """This function launches the polyphy."""
    if len(sys.argv) == 1:
        # if no command line args are passed, show the help options
        parser.parse_args(['-h'])

    else:

        # parse them
        args = parser.parse_args()
        if args.command == "run2d":
            run2d(args.command)
        elif args.command == "run3d":
            run3d(args.command)


if __name__ == "__main__":
    main()
