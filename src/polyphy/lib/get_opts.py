import argparse

from .defaults import VERSION

parser = argparse.ArgumentParser(prog="polyphy")
parser.add_argument('-v', '--version', action='version', version=VERSION)
parser.add_argument('-q', '--quiet', help='suppress output', action='store_true')

# subparsers

# polyphy help
subparsers = parser.add_subparsers(help='sub command help', dest='command')

PolyPhy2D_parser = subparsers.add_parser('run2d', help='run 2D PolyPhy')
PolyPhy3D_parser = subparsers.add_parser('run3d', help='run 2D PolyPhy')
