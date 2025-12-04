/**
 * @typedef {Object} TextRuleSettings
 * @property {boolean} remove_tags - Whether to remove <> tags from the text.
 * @property {boolean} split_percent - Whether to split percentage values (e.g., 50%) into 50 %.
 * @property {boolean} split_percent_words - Whether to split percentage words (e.g., fifty%) into fifty %.
 * @property {boolean} split_hashtag - Whether to split hashtags (e.g., #1) into # 1.
 * @property {boolean} split_hashtag_words - Whether to split hashtag words (e.g., #hello) into # hello.
 * @property {boolean} split_g_suffix - Whether to split 'G' suffixes (e.g., 5G) into 5 G.
 * @property {boolean} fix_ellipsis - Whether to fix ellipsis formatting (e.g., hello...world to hello... world).
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * Get the Python executable path from a virtual environment.
 * @param {string} venvPath - Path to the virtual environment
 * @returns {string|null} Path to the Python executable, or null if not found
 */
function getVenvPython(venvPath) {
    const unixPython = path.join(venvPath, 'bin', 'python');
    const windowsPython = path.join(venvPath, 'Scripts', 'python.exe');

    if (fs.existsSync(unixPython)) return unixPython;
    if (fs.existsSync(windowsPython)) return windowsPython;

    return null;
}

/**
 * Configuration class for Text-to-Speech (TTS) settings.
 * 
 * Mirrored in tts.py
 */
class TTSConfig {
    /**
     * @param {Object} [cfg={}] User provided configuration
     * @param {"cpu"|"cuda"} [cfg.device="cpu"] Device to use for the TTS engine
     * @param {TextRuleSettings} [cfg.text_rule_settings] Toggle specific text processing rules
     * @param {boolean} [cfg.convert_hyphens=true] Whether to convert hyphens to spaces in the input text
     * @param {boolean} [cfg.enable_pronunciation=true] Whether to enable pronunciation dictionary
     * @param {boolean} [cfg.enable_stroke_prevention=true] Whether to enable EOS token / stroke prevention
     * @param {string} [cfg.tacotron_dir="1_TACOTRON_MODELS"] Directory for Tacotron models
     * @param {string} [cfg.hifigan_dir="0_HIFIGAN_MODELS"] Directory for HiFi-GAN models
     */
    constructor(cfg = {}) {
        this.device = cfg.device ?? 'cpu';

        /** @type {TextRuleSettings} */
        this.text_rule_settings = {
            remove_tags: true,
            split_percent: true,
            split_percent_words: true,
            split_hashtag: true,
            split_hashtag_words: true,
            split_g_suffix: true,
            fix_ellipsis: true,
            ...cfg.text_rule_settings
        };

        this.convert_hyphens = cfg.convert_hyphens ?? true;
        this.enable_pronunciation = cfg.enable_pronunciation ?? true;
        this.enable_stroke_prevention = cfg.enable_stroke_prevention ?? true;

        this.tacotron_dir = cfg.tacotron_dir ?? '1_TACOTRON_MODELS';
        this.hifigan_dir = cfg.hifigan_dir ?? '0_HIFIGAN_MODELS';

        if (!['cpu', 'cuda'].includes(this.device)) {
            console.warn(`Invalid device '${this.device}'. Falling back to 'cpu'.`);
            this.device = 'cpu';
        }
    }

    /**
     * Convert the configuration to a JSON object.
     * @returns {Object} JSON representation of the configuration
     */
    toJSON() {
        return {
            device: this.device,
            text_rule_settings: this.text_rule_settings,
            convert_hyphens: this.convert_hyphens,
            enable_pronunciation: this.enable_pronunciation,
            enable_stroke_prevention: this.enable_stroke_prevention,
            tacotron_dir: this.tacotron_dir,
            hifigan_dir: this.hifigan_dir
        };
    }
}

/**
 * LocalTTSModule wrapper class for Node.js
 * 
 * Usage:
 * ```javascript
 *     const cfg = new TTSConfig({ device: 'cpu' });
 *     const tts = new TTS(cfg);
 *     await tts.speak("Hello world!", "Character", "output.wav");
 * ```
 */
class TTS {
    /**
     * @param {TTSConfig} [config] Configuration for the TTS engine
     * @param {string|null} [venvPath] Optional path to a Python virtual environment root
     */
    constructor(config = new TTSConfig(), venvPath = null) {
        /** @type {TTSConfig} */
        this.config = config;

        // Determine Python executable
        this.pythonExec = venvPath ? getVenvPython(venvPath) : 'python';

        // Spawn the Python TTS process
        this.proc = spawn(this.pythonExec, ["tts_worker.py"]);
        this.queue = [];

        let buffer = '';
        this.proc.stdout.on('data', data => {
            buffer += data.toString();
            const messages = buffer.split('\n');
            buffer = messages.pop(); // Keep incomplete message in buffer

            for (const msg of messages) {
                if (!msg.trim()) continue;

                try {
                    const json = JSON.parse(msg);
                    const resolve = this.queue.shift();
                    if (resolve) resolve(json.output);
                } catch (e) {
                    // Not a JSON message
                    console.log("[JS TTS]", msg);
                }
            }
        });

        this.proc.stderr.on('data', err => {
            console.error(err.toString());
        });

        this.proc.on('exit', () => {
            console.warn("Exiting TTS process.");

            while (this.queue.length > 0) {
                const resolve = this.queue.shift();
                if (resolve) resolve(null);
            }
        });
    }

    /**
     * Generate speech audio from dialogue text.
     * 
     * @param {string} dialogue The dialogue text to convert to speech
     * @param {string} character The character model name to use
     * @param {string} outputPath The file path to save the generated .wav audio
     * @returns {Promise<string|null>} Resolves to the output path on success, or null on failure
     */
    speak(dialogue, character, outputPath) {
        return new Promise(resolve => {
            this.queue.push(resolve);
            const payload = {
                dialogue,
                character,
                outputPath,
                config: this.config.toJSON()
            };
            this.proc.stdin.write(JSON.stringify(payload) + '\n');
        });
    }

    /**
     * Shuts down the TTS process.
     */
    close() {
        this.proc.kill();
    }
}

module.exports = { TTSConfig, TTS };