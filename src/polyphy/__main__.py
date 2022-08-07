import sys

from .lib.get_opts import parser
from .lib.PolyPhy2D import run_2D


def main():
    """This function launches the polyphy."""
    if len(sys.argv) == 1:
        # if no command line args are passed, show the help options
        parser.parse_args(['-h'])

    else:

        # parse them
        args = parser.parse_args()
        if args.command == "run2D":
            run_2D()
        if args.command == "run3D":
            # run_3D()
            print("polyphy3D is under development")


if __name__ == "__main__":
    main()
