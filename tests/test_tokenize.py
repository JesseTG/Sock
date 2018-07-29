import pytest
from sockpuppet.model.dataset.twitter_tokenize import tokenize


@pytest.mark.parametrize("input,expected", [
    ("<3", "<heart>"),
    ("💓", "<heart>"),
    ("<3 ♥", "<heart> <heart>"),
    ("♥♥", "<heart> <heart>"),
    ("@iconography", "<user>"),
    ("@NASA", "<user>"),
    ("@", "@"),
    ("#EXCELLENT", "<hashtag> excellent <allcaps>"),
    ("NASA", "nasa <allcaps>"),
    ("NASAcool", "nasacool"),
    ("wayyyy", "way <elong>"),
    ("!!!", "! <repeat>"),
    ("holy shit!!", "holy shit ! <repeat>"),
    ("What are you doing?", "what are you doing ?"),
    ("Are you ok!?", "are you ok ! ?"),
    ("be careful what you wish for.....", "be careful what you wish for . <repeat>"),
    ("you are wayyyy out of line, buddy", "you are way <elong> out of line , buddy"),
    ("Here's an idea: be nice to them. :)", "here's an idea : be nice to them . <smile>"),
    ("Let's be sure to #getoutthevote", "let's be sure to <hashtag> getoutthevote"),
    ("We must #GetOutTheVote this November", "we must <hashtag> getoutthevote this november"),
    ("I met Samuel L. Jackson #MOTHERFUCKER", "i met samuel l . jackson <hashtag> motherfucker <allcaps>"),
    ("#", "#"),
    ("alpha#beta", "alpha#beta"),
    (".", "."),
    ("RT @idaho: Look at me!", "rt <allcaps> <user> : look at me !"),
    (":( ): :< >: :[ ]:", "<sadface> <sadface> <sadface> <sadface> <sadface> <sadface>"),
    ("Not sad):", "not sad) :"),
    (":L :l :| :/ :\\ :*", "<neutralface> <neutralface> <neutralface> <neutralface> <neutralface> <neutralface>"),
    ("Download to C://Users", "download to c : / / users"),
    (":P :p index:P :p", "<lolface> <lolface> index : p <lolface>")
])
def test_tokenize(input, expected):
    assert tokenize(input) == expected.split()
