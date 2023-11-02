from utils.cli_helper import CliHelper


def main():
    CliHelper.parse_args()
    ppConfig = None
    CliHelper.parse_values(ppConfig)


if __name__ == "__main__":
    main()
