# Text-to-Image-to-Video Pipeline

A comprehensive CLI tool that converts text prompts into high-quality videos by chaining **GLM-Image** (Text-to-Image) and **LTX-2** (Image-to-Video) models.

## Overview
This pipeline leverages state-of-the-art diffusion models to generate realistic animations. It supports both direct Text-to-Video (T2V) and a two-stage Text-to-Image-to-Video (T2I2V) workflow, ensuring high fidelity and temporal consistency.

## GPU & VRAM Requirements
- **Recommended Hardware**: NVIDIA A100 (80GB VRAM) or equivalent.
- **Minimum VRAM**: 
  - GLM-Image: ~20-24GB
  - LTX-2 (Inference): ~32GB+ depending on resolution and frame count.
- **Total Pipeline**: Orchestrated to run sequentially to optimize VRAM usage on a single 80GB card.

## Usage Instructions

### 1. Installation
Ensure you have a Python 3.10+ environment.
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Pipeline
The CLI tool uses `main.py` as the entry point.

#### Text-to-Image-to-Video (Complete Pipeline)
```bash
python main.py --mode t2i2v --prompt "A cinematic shot of a futuristic city with flying cars" --output ./output.mp4
```

#### Text-to-Video (Direct LTX-2)
```bash
python main.py --mode t2v --prompt "A serene waterfall in a lush forest" --output ./waterfall.mp4
```

### 3. Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | The text description of the scene | Required |
| `--mode` | `t2i2v` (Full) or `t2v` (Direct Video) | `t2i2v` |
| `--output` | Path to save the final video | `./data/outputs/video.mp4` |
| `--height` | Video height (must be divisible by 32) | `480` |
| `--width` | Video width (must be divisible by 32) | `704` |
| `--num_frames` | Total frames (recommended $8n+1$) | `81` |

## Best Practices & Constraints

### Prompting (LTX-2)
- **Descriptive Quality**: Use keywords like "cinematic", "highly detailed", "4k", and "consistent lighting".
- **Action-Oriented**: Describe the motion explicitly (e.g., "camera slowly zooms in", "gentle swaying of trees").
- **Guide**: Reference the [LTX-2 Prompting Guide](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) for advanced techniques.

### Resolution and Frames
- **Symmetry**: Always use resolutions divisible by **32** (e.g., $704 \times 480$, $1280 \times 704$).
- **Frame Count**: For optimal results with the LTX-2 VAE, use the formula $8n + 1$ (e.g., 41, 65, 81, 121 frames). 
- **Aspect Ratio**: GLM-Image produces high-quality squares or rectangles; the pipeline automatically resizes to meet LTX-2 requirements.

## Performance
- **Image Gen Time**: ~5-10 seconds (A100)
- **Video Gen Time**: ~60-120 seconds for 81 frames (A100)
- **Reproducibility**: Set seeds via code if deterministic results are required across runs.