# GPU Transcriptor

GPU-accelerated audio and video transcription tool using OpenAI's Whisper model. Optimized for AMD GPUs (ROCm) with automatic CPU fallback.

## My experience using and deving it
* **I'm an Odoo Developer and functional Consultant**, nowadays, I usually have several meetings and i hate to take notes every time, even if u write really fast you can misunderstand some things. 

* Sooooo i made my own R+D process to create this project.
* With this, you can translate all the meetings or audio ideas to take notes and create specific files.
* The main thing, im doing is to create PRD files to describe all customer requirements
* Also, this is implemented for using with AMD Graphics Card, as Nvidia Graphics are a moooore Expensive in Mexico, it should work much better with Nvidia.
* **Note that a lot of the code was written with AI.**, to create a useful tool in a few days. now i process all my meetings and ideas, instead of writing them. 
* The main use i make: create a video of some topics and explain them the best i can, process it with transcriptor, pass the transcription to a very well contextualized tool to order main ideas and make improvements on the manuals, add index, side topics, and things like that.

## Features

- **GPU Acceleration**: Optimized for AMD RX 6600 with ROCm support
- **Automatic Fallback**: Uses CPU if GPU is unavailable
- **Multi-format Support**: Transcribes both audio and video files (MP3, MP4, WAV, etc.)
- **Organized Output**: Creates structured folders with timestamps
- **Duplicate Handling**: Automatic numbering for multiple transcriptions of the same file
- **Timestamp Support**: Optional timestamps in markdown format (30-second blocks)
- **Activity Logging**: Complete operation logs for tracking
- **Filename Sanitization**: Automatic cleanup of special characters

## Requirements

- Python 3.8+
- CUDA-compatible GPU (NVIDIA) or ROCm-compatible GPU (AMD)
- For AMD GPUs: ROCm 5.6+ installed
- 4GB+ RAM (8GB+ recommended for larger models)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/GPU_transcriptor_public.git
cd GPU_transcriptor_public
```

### 2. Install dependencies

#### For NVIDIA GPUs (CUDA):
```bash
pip install -r requirements.txt
```

#### For AMD GPUs (ROCm):
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
pip install openai-whisper
```

### 3. Verify GPU installation

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is detected
```

## Usage

### Basic usage

```bash
python3 transcriptor-gpu.py --full_file_path /path/to/your/video.mp4
```

### Advanced usage with all options

```bash
python3 transcriptor-gpu.py \
  --full_file_path /path/to/your/video.mp4 \
  --whisper-size medium \
  --lang es \
  --with_timestamps True \
  --verbose True
```

### Command-line arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--full_file_path` | string | Required | Full path to audio/video file |
| `--whisper-size` | string | `medium` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--lang` | string | `es` | Language code (e.g., `es`, `en`, `fr`) |
| `--with_timestamps` | boolean | `True` | Enable word-level timestamps |
| `--verbose` | boolean | `True` | Show detailed processing output |

### Whisper model sizes

| Model | Parameters | VRAM Required | Speed | Accuracy |
|-------|-----------|---------------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Lowest |
| base | 74M | ~1GB | Fast | Low |
| small | 244M | ~2GB | Moderate | Good |
| medium | 769M | ~5GB | Slow | High |
| large | 1550M | ~10GB | Slowest | Highest |

## GPU Configuration

### AMD RX 6600 Configuration

The script includes optimized settings for AMD RX 6600:

```python
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['GPU_MAX_ALLOC_PERCENT'] = '90'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
```

### For other AMD GPUs

Modify the `HSA_OVERRIDE_GFX_VERSION` according to your GPU architecture:
- RX 6600/6700: `10.3.0`
- RX 6800/6900: `10.3.0`
- RX 5700: `10.1.0`
- Vega 56/64: `9.0.0`

### For NVIDIA GPUs

Comment out or remove the AMD-specific environment variables. CUDA will be detected automatically.

## Output Structure

The script creates organized folders in `~/Videos/transcription_tools/`:

```
transcription_tools/
├── transcripciones_log.txt
├── transcript-filename-08_11_2025/
│   ├── filename_transcripcion.txt
│   └── filename_transcripcion_timestamps.md
└── transcript-filename-08_11_2025_1/  (if duplicate)
    ├── filename_transcripcion.txt
    └── filename_transcripcion_timestamps.md
```

### Output files

1. **Plain text transcription** (`*_transcripcion.txt`): Complete transcription without timestamps
2. **Markdown with timestamps** (`*_transcripcion_timestamps.md`): Formatted transcription with 30-second time blocks
3. **Activity log** (`transcripciones_log.txt`): System log with all operations

### Example markdown output format

```markdown
# Transcripción: video_example

**Date:** 08/11/2025 14:30
**Total time:** 01:23:45

---

## [00:00:00 - 00:00:30]

First block of transcribed text covering the first 30 seconds...

## [00:00:30 - 00:01:00]

Second block of transcribed text covering the next 30 seconds...
```

## Supported File Formats

Whisper supports various audio and video formats:

- **Audio**: MP3, WAV, M4A, FLAC, OGG
- **Video**: MP4, MKV, AVI, MOV, WebM

## Performance Tips

1. **Use appropriate model size**: Start with `medium` for balance of speed/accuracy
2. **Monitor VRAM usage**: Use smaller models if you encounter out-of-memory errors
3. **GPU allocation**: Adjust `GPU_MAX_ALLOC_PERCENT` if needed (default 90%)
4. **Long files**: For files >1 hour, consider using `small` or `medium` models

## Troubleshooting

### GPU not detected

**AMD GPUs:**
```bash
# Check ROCm installation
rocm-smi

# Verify PyTorch sees GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

**NVIDIA GPUs:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory errors

- Use a smaller Whisper model (`small` or `base`)
- Reduce `GPU_MAX_ALLOC_PERCENT` to 80%
- Close other GPU-intensive applications

### File not found errors

- Use absolute paths: `/home/user/videos/file.mp4`
- Check file permissions
- Verify file format is supported

### Slow transcription

- Ensure GPU is being used (check console output)
- Try a smaller model
- Check GPU utilization with `rocm-smi` or `nvidia-smi`

## Logging

All operations are logged to `~/Videos/transcription_tools/transcripciones_log.txt`:

- GPU detection status
- Model loading progress
- Transcription completion
- File operations
- Error messages

## Examples

### Transcribe English podcast

```bash
python3 transcriptor-gpu.py \
  --full_file_path ~/Downloads/podcast.mp3 \
  --whisper-size small \
  --lang en
```

### Transcribe Spanish video without timestamps

```bash
python3 transcriptor-gpu.py \
  --full_file_path ~/Videos/conference.mp4 \
  --whisper-size medium \
  --lang es \
  --with_timestamps False
```

### Fast transcription with tiny model

```bash
python3 transcriptor-gpu.py \
  --full_file_path ~/audio.wav \
  --whisper-size tiny \
  --lang en \
  --verbose False
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm) - AMD GPU computing platform

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

## Changelog

### Version 1.0.0
- Initial release
- GPU acceleration support (AMD/NVIDIA)
- Multi-format transcription
- Timestamp support
- Organized output structure
- Automatic logging