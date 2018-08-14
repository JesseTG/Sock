#!/usr/bin/env python3

import argparse
import fileinput
import json
import sys


def main():
    with open(sys.argv[1], "r") as users:
        # TODO: Give this via a command line arg
        userset = frozenset(u.strip() for u in users.readlines())
        for line in fileinput.input("-"):
            # TODO: Generalize to multiple input streams
            tweet = json.loads(line)

            if "user" in tweet:
                user = tweet["user"]
                if "screen_name" in user:
                    screen_name = user["screen_name"]
                    if screen_name in userset:
                    print(line, end="")

if __name__ == "__main__":
    main()
