# Local TTS Module (designed for AI Show developers)

LocalTTSModule is an open source Tacotron 2 Python module that can be easily implemented to any use case. It was created and made open source to assist AI Show developers use local TTS and/or transition from online services such as FakeYou, originally made for my own show, [The Sans & Papyrus Show](https://www.youtube.com/@SansAndPapyrusShow).

It *should* be easy to implement into any Python project and has very little to learn to get started.

The project uses Tacotron 2 and Hifi-GAN dependencies from forked versions of [justinjohn0306's Tacotron 2](https://github.com/justinjohn0306/TTS-TT2) and [justinjohn0306's Hifi-GAN](https://github.com/justinjohn0306/hifi-gan) repositories, which have been modified to work together in a simplified manner.

## How To Install

1. #### Clone this repository to your local machine.
    - `git clone https://github.com/matty1406/LocalTTSModule.git`

2. #### Navigate to the cloned directory.
    - `cd LocalTTSModule`

3. #### Clone the submodules for Tacotron 2 and Hifi-GAN.
    - `git submodule update --init --recursive`

4. #### (Optional) Create and activate a virtual environment.
    - `python -m venv venv`
    - On Windows: `venv\Scripts\activate`
    - On Linux: `source venv/bin/activate`

5. #### Install the required Python packages.
    - `pip install -r requirements.txt`

6. #### Install PyTorch correctly.
    - Follow the instructions at [PyTorch's official website](https://pytorch.org/get-started/locally/) to install the appropriate versions for your system.

    - For CPU only, you can use:
      - `pip3 install torch torchvision`

    - (**RECOMMENDED**) For systems with CUDA support, install the appropriate version as per your CUDA toolkit.
        - *Don't have CUDA?*
            - Follow [NVIDIA's CUDA installation guide](https://developer.nvidia.com/cuda-downloads) to install CUDA for your system.
            - **Ensure you install CUDA 12.6, CUDA 12.8, or CUDA 13.0 to use PyTorch.**

7. #### Create Tacotron 2 and Hifi-GAN models.
    
    In order to use the TTS module, you need to create Tacotron 2 and Hifi-GAN models. You can either train your own models or download pre-trained models.
    
    - To train your own Tacotron 2 models, use this training notebook provided by FakeYou: [Tacotron 2 Training Notebook](https://colab.research.google.com/github/justinjohn0306/FakeYou-Tacotron2-Notebook/blob/dev/FakeYou_Tacotron_2_Training.ipynb)

    - To train your own Hifi-GAN models, use this training notebook provided by FakeYou: [Hifi-GAN Training Notebook](https://colab.research.google.com/github/justinjohn0306/FakeYou-Tacotron2-Notebook/blob/main/FakeYou_HiFi_GAN_Fine_Tuning.ipynb)

## How To Use

You use the module by importing both `TTSConfig` and `TTS` classes from `tts.py`. You can then create a configuration object and a TTS object, and use the `speak` method to generate speech. Here is a simple example:

```python
# Example usage of the Local TTS Module

# You import the TTS module and use it to synthesize speech
from tts import TTS, TTSConfig

# Create a TTS configuration. You can customize parameters as needed.
# Device is set to 'cpu' here but can be set to 'cuda' if a GPU is available.
config = TTSConfig(device='cpu')

# Initialize the TTS system
tts = TTS(config)

# Give some text to synthesize
text = "Hello, this is a text-to-speech synthesis example."

# Add the speech, specifying the character model and output file path.
# It will return the path to the generated audio file.
audio = tts.speak(text, "Character", "output.wav")
print(f"Audio generated at: {audio}")

# Output: Audio generated at: output.wav
```

### TTSConfig Parameters

- `device`: The device to run the TTS model on. Options are `'cpu'` or `'cuda'`. Default is `'cpu'`.
- `text_rule_settings`: A dictionary for text normalization settings. Defaults to all rules set to `True`.
    - `remove_tags`: Removes <> tags from the text. Default is `True`.
    - `split_percent`: Splits numbers like '50%' into '50 %'. Default is `True`.
    - `split_percent_words`: Splits words with percent signs, like 'fifty%' to 'fifty %'. Default is `True`.
    - `split_hashtag`: Splits numbers from hashtags, like '#1' to '# 1'. Default is `True`.
    - `split_hashtag_words`: Splits words from hashtags, like '#hello' to '# hello'. Default is `True`.
    - `split_g_suffix`: Splits 'G' from numbers, like '5G' to '5 G'. Default is `True`.
    - `fix_ellipsis`: Fixes spaces around ellipses, replacing 'hello...world' with 'hello... world'. Default is `True`.
- `convert_hyphens`: Converts hyphens to spaces in the text. Default is `True`.
- `enable_pronunciation`: Enables pronunciation dictionary using ARPAbet for better pronunciation. Default is `True`.
- `enable_stroke_prevention`: Helps fix intentional "strokes" (a stroke in the AI Show context is when a voice fails to speak correctly and never finishes the sentence) by using an EOS token. Default is `True`.
- `tacotron_dir`: The directory path to the Tacotron 2 models. Default is `1_TACOTRON_MODELS`.
- `hifigan_dir`: The directory path to the Hifi-GAN models. Default is `0_HIFIGAN_MODELS`.

For best results, keep the default settings unless you have specific needs.

### TTS Methods

The `TTS` class has the following methods:

`.speak(text, character, output_path)`
- `text` (**str**): The text to be synthesized.
- `character` (**str**): The character/model name to use for synthesis.
- `output_path` (**str**): The file path where the synthesized audio will be saved.
- **Returns**: The path to the generated audio file.

This is the main method to generate speech from text. The text can be any string, and the character should correspond to a pre-trained model available in the Tacotron 2 models directory. Only specify the character name, not the full path.

The output path is where the resulting audio file will be saved. The method returns the path to the generated audio file. You can then use this audio file as needed in your application.

### Node.js Wrapper

A Node.js wrapper has been created to allow easy integration into JavaScript/Node.js projects. It uses child processes to run the Python TTS module in the background through standard input/output. You can use it by importing the `TTS` and `TTSConfig` classes from `tts.js`. Since it relies on the Python module, make sure you have Python and the required dependencies installed as per the instructions above.

If your Python environment is in a virtual environment, you can specify the path to the virtual environment when creating the `TTS` object. If the virtual environment is in the same directory as the `tts.js` file, simply pass the name of the folder (e.g., `venv`) and it will automatically find the executable based on your operating system.

Here's a simple example of how to use the Node.js wrapper:

```javascript
// Example usage of the Local TTS Module in Node.js

const { TTS, TTSConfig } = require('./tts'); // Import the TTS module (adjust the path as necessary)

// Create a TTS configuration
const config = new TTSConfig({
    device: 'cpu' // or 'cuda' if you have a GPU
});

// Initialize the TTS system
const tts = new TTS(config);

// Synthesize speech
(async () => {
    const audioPath = await tts.speak("Hello, this is a text-to-speech synthesis example.", "Character", "output.wav");
    console.log(`Audio generated at: ${audioPath}`);
})();
```

If you use ESM modules, you can import it like this:

```javascript
import pkg from './tts.js';
const { TTS, TTSConfig } = pkg;
```

## Specifications

- **Programming Language**: Python 3.8+
- **Dependencies**: PyTorch, NumPy, SciPy, and other libraries as specified in `requirements.txt`.
- **Compatibility**: Designed to work on Windows and Linux systems. MacOS does not work due to PyTorch limitations.

### GPU Specifications

Tacotron 2 came out in 2017 and was designed to run on GPUs available at that time. As long as the GPU uses CUDA, it should work fine. However, newer GPUs will provide better performance.

I have tested the module with a ***NVIDIA RTX 2080*** and a ***NVIDIA RTX 3070***, and both work well with roughly 3-5 seconds inference time, depending on length of text.

**Keep in mind that the more models you have, the more VRAM is required to load them all. Each Tacotron 2 model is 300 MB and each Hifi-GAN model is 50 MB. So if you have 3 Tacotron 2 models and 3 Hifi-GAN models loaded, that is roughly 1.05 GB of VRAM used just for the models.**

## License

*The project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.*

#### tl;dr: You can use, modify, and distribute the code as long as you give credit, don't use the author's name for promotion without permission, and understand that there's no warranty.

## Final Notes

I created this module mainly because I noticed a lack of people using local TTS for their AI Show projects, and I wanted to provide a simple solution for developers who want to use local TTS without dealing with complex setups. With the unfortunate slowness and unreliability of online TTS services running Tacotron 2, having a local solution is beneficial.

The module is based on my my own show's implementation. Check out [The Sans & Papyrus Show](https://www.youtube.com/@SansAndPapyrusShow) on YouTube to see it in action!

[![Watch on YouTube](https://img.shields.io/badge/Watch-YouTube-FF0000?logo=youtube&style=for-the-badge)](https://www.youtube.com/@SansAndPapyrusShow)

You can join my show's Discord server for support and discussions:

[![Join our Discord](https://img.shields.io/badge/Join-Discord-7289DA?logo=discord&style=for-the-badge)](https://discord.gg/qMzFAXhGsS)

<img src="https://yt3.googleusercontent.com/mR8bnvrsRKHOBE9S3R2ooJEmgVEfCnBA8ch6ABrFiiboUPNgwKsNcsZDFHFdrlfk2EbLA8mXdZc=s160-c-k-c0x00ffffff-no-rj" alt="The Sans & Papyrus Show Profile" width="150"/>

## To Do

- ✅ Create basic module structure
- ✅ Implement Tacotron 2 and Hifi-GAN integration
- ✅ Create TTSConfig class for easy configuration
- ✅ Add wrapper for JavaScript/Node.js projects

*If you find any issues, inform me via Issues or Discord! I will continue to fix bugs but nothing else is planned for now.*