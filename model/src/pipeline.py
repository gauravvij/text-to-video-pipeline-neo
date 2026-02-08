import torch
from diffusers import DiffusionPipeline, LTX2Pipeline
from PIL import Image
import os
import time

from diffusers import GlmImagePipeline, LTX2ImageToVideoPipeline, LTX2Pipeline
import numpy as np
from model.src.prompt_utils import LTX2PromptExpander

class TextToVideoPipeline:
    """Unified pipeline for Text-to-Image (GLM-Image) + Image-to-Video (LTX-2) and Direct Text-to-Video (LTX-2)."""
    
    def __init__(self, device="cuda", load_t2i=True, load_t2v=True):
        self.device = device
        self.expander = LTX2PromptExpander(device=device)
        
        if load_t2i:
            print("Loading GLM-Image (Text-to-Image)...")
            self.t2i_pipe = GlmImagePipeline.from_pretrained(
                "zai-org/GLM-Image",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.t2i_pipe.enable_model_cpu_offload(device=device)
            
            print("Loading LTX-2 (Image-to-Video)...")
            self.i2v_pipe = LTX2ImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
            )
            self.i2v_pipe.enable_model_cpu_offload(device=device)
        else:
            self.t2i_pipe = None
            self.i2v_pipe = None

        if load_t2v:
            print("Loading LTX-2 (Direct Text-to-Video)...")
            self.t2v_pipe = LTX2Pipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
            )
            self.t2v_pipe.enable_model_cpu_offload(device=device)
        else:
            self.t2v_pipe = None
        
    def run(self, prompt, mode="t2i2v", enhance_prompt=False, output_path="output.mp4", num_frames=81, height=512, width=768, fps=24.0, ltx_guidance=7.5):
        if enhance_prompt:
            print(f"Enhancing prompt: {prompt}")
            prompt = self.expander.expand(prompt)
            print(f"New prompt: {prompt}")

        t_start = time.time()
        t_img = 0
        image = None

        if mode == "t2i2v":
            if self.t2i_pipe is None or self.i2v_pipe is None:
                raise ValueError("t2i2v models not loaded.")
            
            # Stage 1: Text to Image
            print(f"Generating image for prompt: {prompt}")
            t1 = time.time()
            image_out = self.t2i_pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=50,
                guidance_scale=1.5
            )
            image = image_out.images[0]
            image.save("output_intermediate.png")
            t_img = time.time() - t1
            
            torch.cuda.empty_cache()
            
            # Stage 2: Image to Video
            print(f"Generating video ({num_frames} frames) from image...")
            t2 = time.time()
            video_out = self.i2v_pipe(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=40,
                guidance_scale=ltx_guidance,
                output_type="np"
            )
            t_vid = time.time() - t2
        elif mode == "t2v":
            if self.t2v_pipe is None:
                raise ValueError("t2v model not loaded.")
            
            print(f"Generating video ({num_frames} frames) directly from text...")
            t2 = time.time()
            video_out = self.t2v_pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=20,
                guidance_scale=ltx_guidance,
                output_type="np"
            )
            t_vid = time.time() - t2
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Save video
        from diffusers.pipelines.ltx2.export_utils import encode_video
        
        video_frames = (video_out.frames[0] * 255).astype(np.uint8)
        video_tensor = torch.from_numpy(video_frames)
        
        audio = None
        if hasattr(video_out, 'audio') and video_out.audio is not None:
            audio = video_out.audio[0].float().cpu()
        
        pipe_for_audio = self.i2v_pipe if mode == "t2i2v" else self.t2v_pipe
        
        encode_video(
            video_tensor,
            fps=fps,
            output_path=output_path,
            audio=audio,
            audio_sample_rate=getattr(pipe_for_audio.vocoder.config, 'output_sampling_rate', 44100) if audio is not None else 44100
        )
        
        return {
            "image": image,
            "video_path": output_path,
            "t_img": t_img,
            "t_vid": t_vid,
            "vram_peak": torch.cuda.max_memory_allocated(self.device) / 1024**3
        }

if __name__ == "__main__":
    # Smoke test logic can be added here
    pass

if __name__ == "__main__":
    # Test stub
    pass