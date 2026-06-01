import traceback
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from localTTS import LocalTTS

TEXT_RULE_LIBRARY = {
    "remove_tags":          (re.compile(r"<.*?>"), ""),
    "split_percent":        (re.compile(r"(\d+)(%)"), r"\1 \2"),
    "split_percent_words":  (re.compile(r"([a-zA-Z]+)(%)"), r"\1 \2"),
    "split_hashtag":        (re.compile(r"(#)(\d+)"), r"\1 \2"),
    "split_hashtag_words":  (re.compile(r"(#)([a-zA-Z]+)"), r"\1 \2"),
    "split_g_suffix":       (re.compile(r"(\d+)(G)"), r"\1 \2"),
    "fix_ellipsis":         (re.compile(r"\.\.\.(\S)"), r"... \1"),
}

HYPHENATED_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+(?:'[A-Za-z]+)?)+")

# CONFIGURATION
@dataclass
class TextRule:
    """A single regex replacement rule."""
    pattern: str
    repl: str

@dataclass
class TTSConfig:
    device: str = 'cpu'

    # Text normalization rules
    text_rule_settings: Dict[str, bool] = field(default_factory=lambda: {
        "remove_tags": True,
        "split_percent": True,
        "split_percent_words": True,
        "split_hashtag": True,
        "split_hashtag_words": True,
        "split_g_suffix": True,
        "fix_ellipsis": True,
    })

    convert_hyphens: bool = True
    enable_pronunciation: bool = True
    enable_stroke_prevention: bool = True
    
    # Directories for models
    tacotron_dir: str = '1_TACOTRON_MODELS'
    hifigan_dir: str = '0_HIFIGAN_MODELS'

# Main TTS class

class TTS:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config if config is not None else TTSConfig()
        self.__post_init__()
        self.local_tts = LocalTTS(deviceType=self.config.device, tacotron_dir=self.config.tacotron_dir, hifigan_dir=self.config.hifigan_dir)

    def __post_init__(self):
        # Ensure valid device
        if self.config.device not in ('cpu', 'cuda'):
            print(f"Invalid device '{self.config.device}'. Falling back to 'cpu'.")
            self.config.device = 'cpu'

    def _normalize_text(self, text: str) -> str:
        result = text.strip()

        # Apply regex rules
        for name, (pattern, repl) in TEXT_RULE_LIBRARY.items():
            if self.config.text_rule_settings.get(name, True):
                result = pattern.sub(repl, result)
        
        if self.config.convert_hyphens:
            result = self._convert_hyphens(result)

        return result

    def _convert_hyphens(self, text: str) -> str:
        preserved_hyphen = "\0"

        def convert_match(match: re.Match) -> str:
            token = match.group(0)
            parts = token.split("-")
            final_word = parts[-1].lstrip("'").lower()

            if final_word and all(final_word.startswith(part.lower()) for part in parts[:-1]):
                return token.replace("-", preserved_hyphen)

            return token.replace("-", " ")

        return HYPHENATED_WORD_RE.sub(convert_match, text).replace("-", " ").replace(preserved_hyphen, "-")

    def speak(self, dialogue: str, character: str, output_path: str):
        """
        Synthesize speech for the given dialogue and character.
        Args:
            dialogue (str): The text to be synthesized.
            character (str): The character model name to use.
            output_path (str): The path to save the output audio file.
        Returns:
            str: The path to the generated audio file, or None if failed.
        """

        # Normalize text
        normalized_text = self._normalize_text(dialogue)
        
        # Synthesize speech
        output_path = f"{output_path}.wav" if not output_path.endswith('.wav') else output_path

        try:
            self.local_tts.infer(
                text=normalized_text,
                model_name=character,
                hifigan_model_name=character,
                output_file=output_path,
                pronounciation_dictionary=self.config.enable_pronunciation,
                EOS_Token=self.config.enable_stroke_prevention
            )

            print(f"{output_path} - Written successfully.")
            return output_path
        
        except Exception:
            traceback.print_exc()
            return None