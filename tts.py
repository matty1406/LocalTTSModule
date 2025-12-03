import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional

from localTTS import LocalTTS

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
    text_rules: List[TextRule] = field(default_factory=lambda: [
        TextRule(r"<.*?>", ""),                         # Remove TMP tags
        TextRule(r"(\d+)(%)", r"\1 \2"),                # 50% → 50 %
        TextRule(r"([a-zA-Z]+)(%)", r"\1 \2"),          # test% → test %
        TextRule(r"(#)(\d+)", r"\1 \2"),                # #10 → # 10
        TextRule(r"(#)([a-zA-Z]+)", r"\1 \2"),          # #test → # test
        TextRule(r"(\d+)(G)", r"\1 \2"),                # 10G → 10 G
    ])

    convert_hyphens: bool = True

    # Pronunciation toggle
    enable_pronunciation: bool = True

    # EOS toggle
    enable_stroke_prevention: bool = True

    # Directories for models
    tacotron_dir: str = '1_TACOTRON_MODELS'
    hifigan_dir: str = '0_HIFIGAN_MODELS'

    # Custom text pass
    custom_processor: Optional[Callable[[str], str]] = None

# Main TTS class

class TTS:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config if config is not None else TTSConfig()
        self.local_tts = LocalTTS(deviceType=self.config.device, tacotron_dir=self.config.tacotron_dir, hifigan_dir=self.config.hifigan_dir)

    def __post_init__(self):
        # Ensure valid device
        if self.config.device not in ('cpu', 'cuda'):
            print(f"Invalid device '{self.config.device}'. Falling back to 'cpu'.")
            self.config.device = 'cpu'

    def _normalize_text(self, text: str) -> str:
        result = text.strip()

        # Apply regex rules
        for rule in self.config.text_rules:
            result = re.sub(rule.pattern, rule.repl, result)
        
        if self.config.convert_hyphens:
            result = result.replace("-", " ")
        
        # Custom processing
        if self.config.custom_processor:
            result = self.config.custom_processor(result)

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
        print(f"Normalized Text: {normalized_text}")

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