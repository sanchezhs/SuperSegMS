import sys
from cli.cli_utils import handle_generate_config, parse_cli_args
from cli.runner import run_pipeline


def main():
    """
    Main entry point for the CLI application.
    Parses command line arguments and runs the pipeline.
    """
    args = parse_cli_args()

    if args.generate_config is not None:
        handle_generate_config(args.generate_config)
        sys.exit(0)

    run_pipeline(args)


if __name__ == "__main__":
    main()
