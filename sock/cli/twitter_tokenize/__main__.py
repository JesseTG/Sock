#!/usr/bin/env python

import fileinput

from sock.model.data import twitter_tokenize


def main():
    # TODO: Move this function to a module in sock.cli
    for line in fileinput.input():
        print(' '.join(twitter_tokenize.tokenize(line)))

if __name__ == "__main__":
    main()
