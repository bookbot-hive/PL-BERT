# IPA Phonemizer: https://github.com/bootphon/phonemizer

import os
import string

_pad = "[PAD]"
_punctuation = ";:,.!? "
_letters_ipa = [
    "a",
    "b",
    "tʃ",
    "d",
    "e",
    "f",
    "ɡ",
    "h",
    "i",
    "dʒ",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "j",
    "z",
    "ŋ",
    "ə",
    "ɲ",
    "ʃ",
    "x",
    "ʔ",
]

# Export all symbols:
symbols = [_pad] + list(_punctuation) + _letters_ipa

letters = _letters_ipa

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self):
        self.word_index_dictionary = dicts
        # print(len(dicts))

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                indexes.append(self.word_index_dictionary["[PAD]"])  # unknown token
        #                 print(char)
        return indexes