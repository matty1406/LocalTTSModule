import torch
import numpy as np
import scipy.signal
from scipy.signal import firwin, lfilter
import resampy
import json
import os
import soundfile as sf
import sys

# Ensure TTS-TT2 and HiFi-GAN are in your Python path
tacotron_lib_path = os.path.join(os.path.dirname(__file__), 'libs', 'Local-Tacotron-2')
hifigan_lib_path = os.path.join(os.path.dirname(__file__), 'libs', 'Local-Hifi-Gan')

sys.path.append(tacotron_lib_path)
sys.path.append(hifigan_lib_path)

from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser

"""LocalTTS: A class for local text-to-speech synthesis using Tacotron2 and HiFi-GAN models."""
class LocalTTS:
    def __init__(self, deviceType: str, tacotron_dir: str = '1_TACOTRON_MODELS', hifigan_dir: str = '0_HIFIGAN_MODELS'):
        """
        Initializes the LocalTTS class by loading Tacotron2 and HiFi-GAN models.
        Args:
            deviceType (str): The device type to use ('cpu' or 'cuda').
            tacotron_dir (str): Directory containing Tacotron2 model files.
            hifigan_dir (str): Directory containing HiFi-GAN model files.
        """

        self.deviceType = deviceType
        self.device = torch.device(self.deviceType)
        self.super_res = 3

        self.pronounciation_dict = self.__load_pronounciation_dictionary()

        self.tacotron2_models = {}
        self.hifigan_models = {}

        self.__load_all_tacotron2(tacotron_dir)
        self.__load_all_hifigan(hifigan_dir)
        
        hifigan2, h2, denoiser2 = self.__load_hifigan(os.path.join('SR_hifigan', 'Superres_Twilight_33000'), 'config_32k')
        self.hifigan_models['Superres_Twilight_33000'] = (hifigan2, h2, denoiser2)

    def __load_pronounciation_dictionary(self) -> dict:
        """Loads the pronunciation dictionary from a file."""
        thisdict = {}
        for line in reversed((open(os.path.join('CMU_DICTIONARY', 'merged.dict.txt'), "r").read()).splitlines()):
            thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()
        return thisdict
    
    def _apply_pronounciation_dictionary(self, text : str, punctuation=r"!?,.;:'\"", EOS_Token=True) -> str:
        """
        Applies the pronunciation dictionary to the input text.
        Args:
            text (str): The input text to process.
            punctuation (str): Punctuation characters to consider.
            EOS_Token (bool): Whether to append an end-of-sentence token.
            Returns:
            str: The processed text with pronunciation dictionary applied.
        """
        out = ''
        for word_ in text.split(" "):
            word=word_; end_chars = ''; start_chars = ''
            while any(elem in word for elem in punctuation) and len(word) > 1:
                if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
                else: break
            while any(elem in word for elem in punctuation) and len(word) > 1:
                if word[0] in punctuation: start_chars += word[0]; word = word[1:]
                else: break
            try:
                word_arpa = self.pronounciation_dict[word.upper()]
                word = "{" + str(word_arpa) + "}"
            except KeyError: pass
            out = (out + " " + start_chars + word + end_chars).strip()
        
        if EOS_Token and out[-1] != ";": out += ";"
        return out
    
    def __load_all_tacotron2(self, tacotron2_dir : str):
        """
        Loads all Tacotron2 models from the specified directory.
        Args:
            tacotron2_dir (str): Directory containing Tacotron2 model files.
        """
        for file in os.listdir(tacotron2_dir):
            if file.startswith('.'): continue
            model_name = file.split('.')[0]
            model_path = os.path.join(tacotron2_dir, file)
            model, hparams = self.__load_tacotron2(model_path)
            self.tacotron2_models[model_name] = (model, hparams)
            print(f"Loaded Tacotron2 model: {model_name}")

    def __load_tacotron2(self, model_path : str) -> tuple:
        """
        Loads a Tacotron2 model from the specified path.
        Args:
            model_path (str): Path to the Tacotron2 model file.
        Returns:
            tuple: A tuple containing the Tacotron2 model and its hyperparameters.
        """
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = 3000
        hparams.gate_threshold = 0.25
        model = Tacotron2(hparams)
        if self.deviceType == 'cuda':
            state_dict = torch.load(model_path, weights_only=True)['state_dict']
            model.load_state_dict(state_dict)
            _ = model.cuda().eval().half()
        else:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)['state_dict']
            model.load_state_dict(state_dict)
            _ = model.to(self.device).eval()
        return model, hparams
    
    def __load_all_hifigan(self, hifigan_dir : str):
        """
        Loads all HiFi-GAN models from the specified directory.
        Args:
            hifigan_dir (str): Directory containing HiFi-GAN model files.
        """
        for file in os.listdir(hifigan_dir):
            model_name = file.split('.')[0]
            model_path = os.path.join(hifigan_dir, file)
            hifigan, h, denoiser = self.__load_hifigan(model_path, 'config_v1')
            self.hifigan_models[model_name] = (hifigan, h, denoiser)
            print(f"Loaded HiFi-GAN model: {model_name}")

    def __load_hifigan(self, model_path : str, conf_name : str) -> tuple:
        """
        Loads a HiFi-GAN model from the specified path and configuration name.
        Args:
            model_path (str): Path to the HiFi-GAN model file.
            conf_name (str): Configuration name for the HiFi-GAN model.
        Returns:
            tuple: A tuple containing the HiFi-GAN model, its hyperparameters, and the denoiser.
        """
        with open(os.path.join(hifigan_lib_path, f'{conf_name}.json')) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        hifigan = Generator(h).to(self.device)
        state_dict_g = torch.load(model_path, map_location=self.device, weights_only=True)
        hifigan.load_state_dict(state_dict_g['generator'])
        hifigan.eval()
        hifigan.remove_weight_norm()
        denoiser = Denoiser(hifigan, mode='normal')
        return hifigan, h, denoiser
    
    def infer(self, text : str, model_name : str, hifigan_model_name : str, output_file : str, pronounciation_dictionary : bool = True, EOS_Token : bool = True):
        """
        Performs inference to generate speech audio from the given text using specified models.
        Args:
            text (str): Input text to be synthesized.
            model_name (str): Name of the Tacotron2 model to use.
            hifigan_model_name (str): Name of the HiFi-GAN model to use.
            output_file (str): Path to save the generated audio file.
            pronounciation_dictionary (bool, optional): Whether to apply a pronunciation dictionary. Defaults to True.
            EOS_Token (bool, optional): Whether to append an end-of-sentence token. Defaults to True.
        """
        if pronounciation_dictionary:
            # Apply pronunciation dictionary and ensure text ends with a period
            text = self._apply_pronounciation_dictionary(text, EOS_Token=EOS_Token)
            if EOS_Token: text=text+"."
        else:
            # Ensure text ends with a semicolon and period
            if text[-1] != ";" and EOS_Token: text=text+";" + "."
        # ---
        # Load models - Get the Tacotron2 and HiFi-GAN models based on model names, plus the super-resolution model
        # ---
        model, _ = self.tacotron2_models[model_name]
        hifigan, h, denoiser = self.hifigan_models[hifigan_model_name]
        hifigan_sr, h2, _ = self.hifigan_models.get("Superres_Twilight_33000", (None, None, None))
        # ---

        # ---
        # Inference - With no_grad for efficiency
        # ---
        with torch.no_grad():
            # Text to sequence conversion
            sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]

            # Move to device
            if self.deviceType == 'cuda':
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            else:
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(self.device).long()
            _, mel_outputs_postnet, *_ = model.inference(sequence)

            # HiFi-GAN Vocoder - Convert mel spectrogram to audio waveform
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

            # Resampling and Super-Resolution - Upsample and apply super-resolution
            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            wave = resampy.resample(
                audio_denoised,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # Super-Resolution HiFi-GAN
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(self.device)
            new_mel = mel_spectrogram(wave.unsqueeze(0), h2.n_fft, h2.num_mels,
                                    h2.sampling_rate, h2.hop_size, h2.win_size,
                                    h2.fmin, h2.fmax)
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)

            # High-Frequency Filtering and Mixing
            b = firwin(101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False)
            y = lfilter(b, [1.0], audio2_denoised)
            y *= self.super_res  # superres_strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize

            # Save output audio file
            sf.write(output_file, sr_mix.astype(np.int16), h2.sampling_rate)
            print(f"Audio saved to {output_file}")