#!/usr/bin/env python3
import argparse
import logging
import time


def setup_logger(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Show logging in action.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")

    for i in range(3):
        logging.info(f"Step {i+1}/3")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
