import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys

class LTX2PromptExpander:
    """Expands simple prompts into detailed LTX-2 structured prompts."""
    
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B", device="cuda"):
        self.device = device
        print(f"Loading prompt expansion model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            self.model.to("cpu")

    def expand(self, prompt: str) -> str:
        # Instruction based on LTX-2 prompting guide: character, environment, lighting, camera
        system_prompt = (
            "You are a professional cinematic prompt engineer for LTX-2. "
            "Convert a user's simple idea into a detailed, descriptive, and atmospheric video prompt. "
            "Describe the characters, their movements, the environment, specific lighting conditions (e.g., volumetric, neon, golden hour), "
            "and camera work (e.g., tracking shot, low angle, slow zoom). "
            "Keep the output as one high-quality paragraph. No labels, no prefixes."
        )
        
        user_input = f"Simple idea: {prompt}\nDetailed LTX-2 prompt:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Test prompt expansion")
    args = parser.parse_args()
    
    if args.test:
        try:
            expander = LTX2PromptExpander()
            expanded = expander.expand(args.test)
            print("-" * 50)
            print(f"Original: {args.test}")
            print(f"Expanded: {expanded}")
            print("-" * 50)
        except Exception as e:
            print(f"Error during expansion: {e}")
            sys.exit(1)