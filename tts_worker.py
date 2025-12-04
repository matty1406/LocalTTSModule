import sys
import json
from tts import TTS, TTSConfig

def log(msg):
    """Send log messages to stdout with a special prefix."""
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()

def send_json(data):
    """Send JSON messages to stdout with a special prefix."""
    sys.stdout.write(f"{json.dumps(data)}\n")
    sys.stdout.flush()

# Keep TTS instance alive across multiple requests
current_config = None
tts_instance = None

for line in sys.stdin:
    if not line.strip():
        continue

    try:
        data = json.loads(line)
    except Exception as e:
        log(f"JSON parse error: {e}")
        continue
    try:
        # Rebuild TTS instance if config has changed
        if data["config"] != current_config:
            current_config = data["config"]
            config = TTSConfig(**current_config)
            tts_instance = TTS(config)
    except Exception as e:
        log(f"Config creation error: {e}")
        continue
    
    try:
        result = tts_instance.speak(
            dialogue=data["dialogue"],
            character=data["character"],
            output_path=data["outputPath"]
        )
    except Exception as e:
        log(f"TTS speak error: {e}")
        continue

    try:
        send_json({"output": result})
    except Exception as e:
        log(f"Output write error: {e}")
        continue