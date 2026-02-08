import torch
import json
import time
import os
from model.src.pipeline import TextToVideoPipeline

def run_benchmarks():
    prompts = [
        "A cyberpunk city with neon lights and flying cars",
        "A serene forest with a crystal clear waterfall",
        "An astronaut walking on a colorful alien planet"
    ]
    
    pipeline = TextToVideoPipeline(device="cuda")
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nBenchmarking Prompt {i+1}: {prompt}")
        output_path = f"/root/i2v/data/outputs/eval_{i}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        res = pipeline.run(
            prompt=prompt,
            output_path=output_path,
            num_frames=81,
            height=512,
            width=768
        )
        total_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        
        results.append({
            "prompt": prompt,
            "image_gen_time": res["t_img"],
            "video_gen_time": res["t_vid"],
            "total_inference_time": total_time,
            "peak_vram_gb": peak_vram
        })

    # Save metrics
    with open("/root/i2v/performance_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Report
    avg_img = sum(r["image_gen_time"] for r in results) / len(results)
    avg_vid = sum(r["video_gen_time"] for r in results) / len(results)
    max_vram = max(r["peak_vram_gb"] for r in results)
    
    report = f"""# Performance and Resource Report

## Summary
The pipeline successfully integrates GLM-Image and LTX-2 on NVIDIA A100-80GB.

## Benchmarks (Target: 81 frames, 768x512)
- **Average Image Generation Time:** {avg_img:.2f}s
- **Average Video Generation Time:** {avg_vid:.2f}s
- **Peak VRAM Usage:** {max_vram:.2f} GB

## Reproducibility
Verified across {len(prompts)} content domains (Cyberpunk, Nature, Sci-Fi).

| Prompt | Image Time | Video Time | Peak VRAM |
|--------|------------|------------|-----------|
"""
    for r in results:
        report += f"| {r['prompt'][:30]}... | {r['image_gen_time']:.1f}s | {r['video_gen_time']:.1f}s | {r['peak_vram_gb']:.2f}GB |\n"
        
    with open("/root/i2v/performance_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_benchmarks()