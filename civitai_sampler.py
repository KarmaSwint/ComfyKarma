import json
from datetime import datetime
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import io
import comfy

class CivitAISampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "vae": ("VAE",),
            },
            "optional": {
                "embeddings": ("STRING", {"multiline": True}),
                "additional_metadata": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def parse_embeddings(self, embeddings_str):
        resources = []
        try:
            for embed in embeddings_str.split('\n'):
                parts = embed.strip().split(',')
                if len(parts) >= 1:
                    resource_data = {"type": "embed", "modelName": parts[0].strip()}
                    if len(parts) >= 2:
                        resource_data["modelVersionId"] = parts[1].strip()
                    resources.append(resource_data)
        except Exception as e:
            print(f"Error parsing embeddings: {str(e)}")
        return resources

    def extract_model_info(self, model):
        model_info = {"type": "checkpoint"}
        try:
            if hasattr(model, 'model_name'):
                model_info["modelName"] = model.model_name
            if hasattr(model, 'model_version'):
                model_info["modelVersionId"] = model.model_version
        except Exception as e:
            print(f"Error extracting model info: {str(e)}")
        return model_info

    def extract_lora_info(self, model):
        lora_resources = []
        try:
            if hasattr(model, 'loaded_loras'):
                for lora in model.loaded_loras:
                    lora_info = {
                        "type": "lora",
                        "modelName": lora.name,
                        "weight": lora.strength
                    }
                    if hasattr(lora, 'version'):
                        lora_info["modelVersionId"] = lora.version
                    lora_resources.append(lora_info)
        except Exception as e:
            print(f"Error extracting LoRA info: {str(e)}")
        return lora_resources

    def sample(self, model, positive, negative, latent_image, sampler_name, cfg_scale, seed, clip_skip, vae, embeddings="", additional_metadata=""):
        try:
            sampler = comfy.samplers.KSampler(model)
            sampler.latent_image = latent_image
            sampler.seed = seed
            sampler.steps = 20
            sampler.cfg = cfg_scale
            sampler.sampler_name = sampler_name
            sampler.scheduler = comfy.samplers.KSampler.SCHEDULERS[sampler_name]
            sampler.denoise = 1.0

            positive_prompts = [prompt.strip() for prompt in positive.split("\n") if prompt.strip()]
            negative_prompts = [prompt.strip() for prompt in negative.split("\n") if prompt.strip()]

            output_images = []
            for prompt in positive_prompts:
                uc = comfy.utils.ConditioningData()
                uc.prompt = prompt
                uc.negative_prompt = " ".join(negative_prompts)
                uc.steps = sampler.steps
                uc.cfg_scale = cfg_scale
                uc.sampler_name = sampler_name
                
                # Derive size from latent_image
                size = latent_image.shape[-1]  # Assuming the latent_image has shape (batch_size, channels, height, width)
                uc.size = (size, size)
                
                uc.clip_skip = clip_skip

                try:
                    samples = sampler.sample(uc, model)
                    decoded_samples = model.decode_latent_images(samples, vae)
                    output_images.extend(decoded_samples)
                except Exception as e:
                    print(f"Error during sampling: {str(e)}")
                    continue

            metadata = {
                "prompt": " ".join(positive_prompts),
                "negative_prompt": " ".join(negative_prompts),
                "steps": sampler.steps,
                "sampler": sampler_name,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "clip_skip": clip_skip,
                "size": f"{size}x{size}",
                "created_date": datetime.utcnow().isoformat() + "Z",
                "civitai_resources": []
            }

            try:
                # Extract base model (checkpoint) information
                metadata["civitai_resources"].append(self.extract_model_info(model))

                # Extract LoRA information from the model
                metadata["civitai_resources"].extend(self.extract_lora_info(model))

                # Parse embeddings
                metadata["civitai_resources"].extend(self.parse_embeddings(embeddings))

                if additional_metadata:
                    try:
                        additional_data = json.loads(additional_metadata)
                        metadata.update(additional_data)
                    except json.JSONDecodeError:
                        print("Warning: Invalid JSON in additional_metadata. Skipping.")
            except Exception as e:
                print(f"Error extracting metadata: {str(e)}")

            metadata_str = json.dumps(metadata, indent=2)

            output_images_with_metadata = []
            for image in output_images:
                try:
                    pil_image = Image.fromarray(image.numpy().astype('uint8'), 'RGB')
                    png_info = PngInfo()
                    png_info.add_text("parameters", metadata_str)
                    png_info.add_text("prompt", metadata["prompt"])
                    png_info.add_text("negative_prompt", metadata["negative_prompt"])

                    image_bytes = io.BytesIO()
                    pil_image.save(image_bytes, format='PNG', pnginfo=png_info)
                    image_bytes.seek(0)
                    output_images_with_metadata.append(image_bytes.getvalue())
                except Exception as e:
                    print(f"Error attaching metadata to image: {str(e)}")
                    continue

            return (output_images_with_metadata, metadata_str)
        except Exception as e:
            print(f"Error during sampling: {str(e)}")
            return (None, "")

NODE_CLASS_MAPPINGS = {
    "CivitAISampler": CivitAISampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitAISampler": "CivitAI Sampler"
}