import argparse
import os
import sys
import torch
from model.src.pipeline import TextToVideoPipeline

def main():
    parser = argparse.ArgumentParser(description="Text-to-Video CLI Tool (GLM-Image + LTX-2)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save the output video")
    parser.add_argument("--mode", type=str, choices=["t2i2v", "t2v"], default="t2i2v", help="Generation mode: t2i2v (text-to-image-to-video) or t2v (direct text-to-video)")
    parser.add_argument("--enhance_prompt", action="store_true", help="Use Qwen LLM to expand the user prompt into a descriptive LTX-2 prompt")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (default: cuda)")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames (8n+1 formula recommended, e.g., 81, 121)")
    parser.add_argument("--fps", type=float, default=24.0, help="Frames per second for output video")
    parser.add_argument("--resolution", type=str, default="768x512", help="Resolution in WxH format (e.g., 768x512, 512x768)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale for LTX-2 (default: 7.5)")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print("Invalid resolution format. Use WxH (e.g., 768x512).")
        sys.exit(1)
        
    print(f"Initializing pipeline on {args.device}...")
    # Conditionally load models to save VRAM if only one mode is used
    pipeline = TextToVideoPipeline(
        device=args.device,
        load_t2i=(args.mode == "t2i2v"),
        load_t2v=(args.mode == "t2v")
    )
    
    print(f"Starting pipeline (Mode: {args.mode}, Enhance: {args.enhance_prompt}) for prompt: {args.prompt}")
    results = pipeline.run(
        prompt=args.prompt,
        mode=args.mode,
        enhance_prompt=args.enhance_prompt,
        output_path=args.output,
        num_frames=args.frames,
        height=height,
        width=width,
        fps=args.fps,
        ltx_guidance=args.guidance
    )
    
    print("\n" + "="*30)
    print("Pipeline Execution Complete!")
    print(f"Video saved to: {results['video_path']}")
    print(f"Image generation time: {results['t_img']:.2f}s")
    print(f"Video generation time: {results['t_vid']:.2f}s")
    print(f"Total time: {results['t_img'] + results['t_vid']:.2f}s")
    print(f"Peak VRAM usage: {results['vram_peak']:.2f} GB")
    print("="*30)

if __name__ == "__main__":
    main()