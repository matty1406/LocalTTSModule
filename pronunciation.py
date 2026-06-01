import os
import re
from pathlib import Path

STUTTER_RE = re.compile(
    r"\b(?P<prefixes>(?:[A-Za-z]{1,3}-)+)(?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)\b"
)

STUTTER_PIECE_TO_ARPA = {
    # Single consonant sounds
    "b": "B",
    "d": "D",
    "f": "F",
    "g": "G",
    "h": "HH",
    "j": "JH",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "p": "P",
    "r": "R",
    "s": "S",
    "t": "T",
    "v": "V",
    "w": "W",
    "y": "Y",
    "z": "Z",

    # Common consonant clusters / digraphs
    "ch": "CH",
    "sh": "SH",
    "th": "TH",
    "dh": "DH",
    "ph": "F",
    "wh": "W",
    "ng": "NG",

    # Optional clusters
    "bl": "B L",
    "br": "B R",
    "cl": "K L",
    "cr": "K R",
    "dr": "D R",
    "fl": "F L",
    "fr": "F R",
    "gl": "G L",
    "gr": "G R",
    "pl": "P L",
    "pr": "P R",
    "sk": "S K",
    "sl": "S L",
    "sm": "S M",
    "sn": "S N",
    "sp": "S P",
    "st": "S T",
    "sw": "S W",
    "tr": "T R",
}

VOWELS = {"a", "e", "i", "o", "u"}

class PronunciationProcessor:
    def __init__(self, cmu_dict_dir: str):
        self.pronunciation_dict = self._load_pronunciation_dictionary(cmu_dict_dir)

    def _load_pronunciation_dictionary(self, cmu_dict_dir: str) -> dict[str, str]:
        dictionary_path = Path(cmu_dict_dir) / "merged.dict.txt"
        pronunciations = {}

        with open(dictionary_path, "r", encoding="utf-8") as file:
            for line in reversed(file.read().splitlines()):
                if not line.strip():
                    continue

                word, arpa = line.split(" ", 1)
                pronunciations[word.upper()] = arpa.strip()

        return pronunciations

    def apply(self, text: str, punctuation=r"!?,.;:'\"", eos_token: bool = True) -> str:
        output = []

        for raw_token in text.split(" "):
            if not raw_token:
                continue

            start_chars, token, end_chars = self._split_edge_punctuation(
                raw_token,
                punctuation,
            )

            converted = self._token_to_arpa(token)
            output.append(f"{start_chars}{converted}{end_chars}")

        result = " ".join(output)

        if eos_token and result and result[-1] != ";":
            result += ";"

        return result

    def _split_edge_punctuation(
        self,
        token: str,
        punctuation: str,
    ) -> tuple[str, str, str]:
        start_chars = ""
        end_chars = ""

        while len(token) > 1 and token[-1] in punctuation:
            end_chars = token[-1] + end_chars
            token = token[:-1]

        while len(token) > 1 and token[0] in punctuation:
            start_chars += token[0]
            token = token[1:]

        return start_chars, token, end_chars

    def _token_to_arpa(self, token: str) -> str:
        stutter_arpa = self._stutter_to_arpa(token)
        if stutter_arpa:
            return stutter_arpa

        word_arpa = self._word_to_arpa(token)
        if word_arpa:
            return self._wrap_arpa(word_arpa)

        return token

    def _word_to_arpa(self, word: str) -> str | None:
        return self.pronunciation_dict.get(word.upper())

    def _stutter_to_arpa(self, token: str) -> str | None:
        match = STUTTER_RE.fullmatch(token)
        if not match:
            return None

        word = match.group("word")
        word_arpa = self._word_to_arpa(word)

        if not word_arpa:
            return None

        prefix_parts = match.group("prefixes").rstrip("-").split("-")
        arpa_parts = []

        for part in prefix_parts:
            part_arpa = self._stutter_piece_to_arpa(part, word_arpa)

            if not part_arpa:
                return None

            # Makes "s-s-stutter" sound like "ss-ss-stutter",
            # not "ess ess stutter".
            if len(part) == 1:
                part_arpa = f"{part_arpa} {part_arpa}"

            arpa_parts.append(self._wrap_arpa(part_arpa))

        arpa_parts.append(self._wrap_arpa(word_arpa))
        return " ".join(arpa_parts)

    def _stutter_piece_to_arpa(self, piece: str, word_arpa: str) -> str | None:
        piece = piece.lower()

        if piece in STUTTER_PIECE_TO_ARPA:
            return STUTTER_PIECE_TO_ARPA[piece]
        
        if piece in VOWELS:
            return self._first_arpa_phone(word_arpa)
        
    def _first_arpa_phone(self, word_arpa: str) -> str | None:
        phones = word_arpa.split()

        if not phones:
            return None

        return phones[0]

    def _wrap_arpa(self, arpa: str) -> str:
        return "{" + arpa + "}"