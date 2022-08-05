import argparse

from polyphy.lib.defaults import VERSION

parser = argparse.ArgumentParser(prog="polyphy")
parser.add_argument('-v', '--version', action='version', version=VERSION)
parser.add_argument('-q', '--quiet', help='suppress output', action='store_true')

# subparsers

# polyphy help
subparsers = parser.add_subparsers(help='sub command help', dest='command')
