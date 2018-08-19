#!/usr/bin/env python3

from typing import List
import re
import fileinput

eyes = "[8:=;]"
nose = "['`-]?"
# Different regex parts for smiley faces

URLS = re.compile(r"https?://\S+\b|www\.(\w +\.)+\S*")
MENTIONS = re.compile(r"@\w+")
SMILE = re.compile(rf"{eyes}{nose}[)D>\]]+|[(<\[]+{nose}{eyes}")
LOLFACE = re.compile(rf"\B{eyes}{nose}[pP]+")
SADFACE = re.compile(rf"{eyes}{nose}[(\[<]+| [D)\]>]+{nose}{eyes}")
NEUTRALFACE = re.compile(rf"\B{eyes}{nose}[\\\\/|lL*]")
HEARTS = re.compile("(<3)|[❤♥🖤💓💖💕💚💞💗💝💙💜💘💟🧡💛]")
SLASHES = re.compile("/")
NUMBERS = re.compile(r"\.?\b[-+:,.\d]*\d\b")
HASHTAGS = re.compile(r"\B#(\S+)")
REPEATED_PUNCTUATION = re.compile(r"([!?.])\1+")
OTHER_PUNCTUATION = re.compile(r"([\",:;=+.!?()])")
ELONGATED_WORDS = re.compile(r"\b(\S*?)(.)\2{2,}\b")
RETWEET = re.compile(r"^RT\b")
ALL_CAPS_WORDS = re.compile(r"\b([^a-z0-9()<>'`\s-]{2,})\b")


def tokenize(input: str) -> List[str]:
    # TODO: Rewrite with a custom iterator of some kind and benchmark it
    # As it is, this generates a lot of garbage

    if len(input) == 0:
        return ["<empty>"]

    input = RETWEET.sub("<retweet> ", input)

    input = URLS.sub("<url>", input)
    # URLS

    input = MENTIONS.sub("<user>", input)
    # User mentions

    input = SMILE.sub("<smile> ", input)
    input = LOLFACE.sub("<lolface> ", input)
    input = SADFACE.sub("<sadface> ", input)
    input = NEUTRALFACE.sub("<neutralface> ", input)
    # Common old-style emoticons

    input = HEARTS.sub(" <heart> ", input)
    # Hearts (<3 and emoji-style)

    input = SLASHES.sub(" / ", input)
    # Force splitting words appended with slashes (once we tokenized the URLs, of course)

    input = NUMBERS.sub("<number>", input)
    # Numbers

    input = HASHTAGS.sub(r"<hashtag> \1 ", input)
    # Hashtags
    # TODO: Tokenize HashTagsWithCamelCase as multiple words

    input = REPEATED_PUNCTUATION.sub(r"\1 <repeat> ", input)
    # Mark punctuation repetitions (eg. "!!!" => "! <repeat>")

    input = OTHER_PUNCTUATION.sub(r" \1 ", input)
    # Other common punctuation that might or might not be surrounded by whitespace

    input = ELONGATED_WORDS.sub(r"\1\2 <elong> ", input)
    # Mark elongated words (eg. "wayyyy" => "way <elong>")

    input = ALL_CAPS_WORDS.sub(r"\1 <allcaps> ", input)
    # Mark all-caps words

    return input.lower().split()


def main():
    for line in fileinput.input():
        print(' '.join(tokenize(line)))

if __name__ == "__main__":
    main()
