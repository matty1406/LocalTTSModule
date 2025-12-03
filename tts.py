import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional

from localTTS import LocalTTS

TEXT_RULE_LIBRARY = {
    "remove_tags":          (r"<.*?>", ""),
    "split_percent":        (r"(\d+)(%)", r"\1 \2"),
    "split_percent_words":  (r"([a-zA-Z]+)(%)", r"\1 \2"),
    "split_hashtag":        (r"(#)(\d+)", r"\1 \2"),
    "split_hashtag_words":  (r"(#)([a-zA-Z]+)", r"\1 \2"),
    "split_g_suffix":       (r"(\d+)(G)", r"\1 \2"),
    "fix_ellipsis":         (r"\.\.\.(\S)", r"... \1"),
}

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
                result = re.sub(pattern, repl, result)
        
        if self.config.convert_hyphens:
            result = result.replace("-", " ")

        return result

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

        # Find model
        tacotron_model = os.path.join(self.config.tacotron_dir, character)

        if not os.path.exists(tacotron_model):
            print(f"Tacotron model for {character} not found. Aborting.")
            return None
        
        hifigan_name = character
        hifigan_model = os.path.join(self.config.hifigan_dir, hifigan_name)

        if not os.path.exists(hifigan_model):
            print(f"Hifi-GAN model for {character} not found, using universal model.")
            hifigan_model = os.path.join(self.config.hifigan_dir, "universal")
            hifigan_name = "universal"
            
        if not os.path.exists(hifigan_model):
            print("Universal Hifi-GAN model not found. Aborting.")
            return None
        
        # Synthesize speech
        output_path = f"{output_path}.wav" if not output_path.endswith('.wav') else output_path

        try:
            self.local_tts.infer(
                text=normalized_text,
                model_name=character,
                hifigan_model_name=hifigan_name,
                output_file=output_path,
                pronounciation_dictionary=self.config.enable_pronunciation,
                EOS_Token=self.config.enable_stroke_prevention
            )

            print(f"{output_path} - Written successfully.")
            return output_path
        
        except Exception as e:
            print(e)
            return None