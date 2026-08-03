"""
api_client.py
Handles asynchronous interactions with the Nano Banana API via Google Cloud AI Platform
(formerly Vertex AI), BytePlus ModelArk API for Seedream models, and Local Inference pipelines.
For the Nano Banana API, it also includes decoupled methods for standard Generation
and Google Cloud Batch API processing.
"""

import os
import json
import time
import asyncio
import base64
import requests
from typing import List, Optional
from io import BytesIO
from PIL import Image

# Google GenAI Imports
from google import genai
from google.genai import types
from google.genai.types import FinishReason
from google.cloud import storage
from google.cloud.storage.blob import Blob


class BytePlusClient:
    """
    Handles interactions with the BytePlus ModelArk API for Seedream models.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        # Standard BytePlus ModelArk V3 endpoint for Image Generation
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

    async def check_reachability(self) -> bool:
        """
        Lightweight verification logic
        (skipped for BytePlus to avoid arbitrary inference costs).
        """
        return True if self.api_key else False

    async def generate_images_batch(
        self,
        prompt: str,
        model_name: str,
        batch_size: int,
        resolution: str,
        aspect_ratio: str,
        input_image_paths: Optional[List[str]] = None,
    ) -> List[Image.Image]:
        """
        Executes a real-time batch image generation request using BytePlus.
        Runs standard synchronous requests cleanly in a background thread to prevent UI freezing.
        """
        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            model_name,
            batch_size,
            resolution,
            aspect_ratio,
            input_image_paths,
        )

    def _calculate_dimensions(self, resolution: str, aspect_ratio: str):
        import math

        # 1. Determine base target area (1K defaults to 1024x1024)
        base_dim = 1024
        if resolution == "2K":
            base_dim = 2048
        elif resolution == "4K":
            base_dim = 4096

        target_area = base_dim * base_dim

        # 2. Parse the string aspect ratio (e.g., "16:9")
        w_ratio, h_ratio = map(float, aspect_ratio.split(":"))
        ratio = w_ratio / h_ratio

        # 3. Compute dimensions maintaining total pixel count
        ideal_height = math.sqrt(target_area / ratio)
        ideal_width = ideal_height * ratio

        # 4. Stable Diffusion requires dimensions to be multiples of 8
        width = int(round(ideal_width / 8.0) * 8)
        height = int(round(ideal_height / 8.0) * 8)

        return width, height

    def _generate_sync(
        self,
        prompt: str,
        model_name: str,
        batch_size: int,
        resolution: str,
        aspect_ratio: str,
        input_image_paths: Optional[List[str]],
    ) -> List[Image.Image]:
        # 1. Translate string resolutions and aspect ratios to explicit pixel targets
        # Dynamically calculate the aspect-ratio aware dimensions
        width, height = self._calculate_dimensions(resolution, aspect_ratio)

        # 2. Construct OpenAI-compatible ModelArk JSON payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "n": batch_size,
            "response_format": "b64_json",
        }

        # 3. Handle base64 Encoding for multi-image references
        if input_image_paths:
            b64_images = []
            for path in input_image_paths[:14]:  # Most Seedream models cap around 10-14 references
                try:
                    with Image.open(path) as img:
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        b64_images.append(f"data:image/jpeg;base64,{b64_str}")
                except Exception as e:
                    raise ValueError(f"Failed to process input image {path}: {str(e)}")

            if b64_images:
                payload["image"] = b64_images if len(b64_images) > 1 else b64_images[0]

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # 4. Execute the network request
        try:
            # Seedream Pro models take deep-reasoning paths; use a lengthy timeout.
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            images = []
            for item in data.get("data", []):
                if "b64_json" in item:
                    img_bytes = base64.b64decode(item["b64_json"])
                    images.append(Image.open(BytesIO(img_bytes)))
                elif "url" in item:
                    res = requests.get(item["url"], timeout=30)
                    images.append(Image.open(BytesIO(res.content)))

            if not images:
                raise RuntimeError("No image data found in the BytePlus API response.")
            return images

        except requests.exceptions.RequestException as e:
            err_msg = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    err_msg += f" - Response JSON: {json.dumps(e.response.json())}"
                except (ValueError, TypeError):
                    err_msg += f" - Response Text: {e.response.text}"
            raise RuntimeError(f"BytePlus API Error: {err_msg}")


class NanoBananaClient:
    # 1. Add location as a parameter (defaulting to us-central1 for maximum model compatibility)
    def __init__(self, api_key: str = "", project_id: str = "", location: str = "global"):
        self.api_key = api_key.strip()
        self.project_id = project_id.strip()
        self.location = location.strip()
        self.client = self._initialize_client()

    def _initialize_client(self):
        """
        Initializes the Agent Platform client.
        Routes through Google Cloud if Project ID is provided.
        """
        try:
            if self.project_id:
                # Corrected: Use enterprise=True for Gemini Enterprise Agent Platform (Vertex AI)
                return genai.Client(
                    enterprise=True, project=self.project_id, location=self.location
                )
            elif self.api_key:
                return genai.Client(api_key=self.api_key)
            return None
        except Exception:
            return None

    async def check_reachability(self) -> bool:
        """Pings the API to verify credentials and connectivity using a standard text model."""
        if not self.client:
            return False
        try:
            # Use a lightweight text model for the connectivity handshake test
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash",
                contents="test",
            )
            return True
        except Exception:
            return False

    async def generate_images_batch(
        self,
        prompt: str,
        model_name: str,
        batch_size: int,
        resolution: str,
        aspect_ratio: str,
        gcs_bucket_name: str,
        use_gcs_for_refs: bool = False,
        input_image_paths: Optional[List[str]] = None,
    ) -> List[Image.Image]:
        """
        Executes a real-time (on-demand) batch image generation request,
        with optional GCS offloading for large reference images.
        """
        if not self.client:
            raise ValueError("API Client not initialized. Please configure Settings.")
        if input_image_paths and use_gcs_for_refs and not gcs_bucket_name.strip():
            raise ValueError(
                "A Google Cloud Storage Bucket Name is required to upload reference images."
            )

        # 1. Compile the multimodal contents list (Text + Optional Reference Images)
        contents = [prompt]
        uploaded_blobs = []

        try:
            if input_image_paths:
                # BRANCH 1: GCS Upload Enabled
                if use_gcs_for_refs:
                    storage_client = storage.Client(project=self.project_id)
                    bucket = storage_client.bucket(gcs_bucket_name)
                    timestamp = int(time.time())

                    for i, img_path in enumerate(input_image_paths[:16]):
                        try:
                            with Image.open(img_path) as img:
                                if img.mode in ("RGBA", "P"):
                                    img = img.convert("RGB")

                                buffered = BytesIO()
                                img.save(buffered, format="JPEG")

                                blob_name = f"realtime_inputs/ref_{timestamp}_{i}.jpg"
                                blob = bucket.blob(blob_name)
                                blob.upload_from_string(
                                    buffered.getvalue(), content_type="image/jpeg"
                                )
                                uploaded_blobs.append(blob)

                                gcs_uri = f"gs://{gcs_bucket_name}/{blob_name}"
                                contents.append(
                                    types.Part.from_uri(file_uri=gcs_uri, mime_type="image/jpeg")
                                )
                        except Exception as e:
                            raise ValueError(
                                f"Failed to process and upload input image {img_path}: {str(e)}"
                            )

                # BRANCH 2: Standard Inline Execution (SDK handles Base64 conversion automatically)
                else:
                    for img_path in input_image_paths[:16]:
                        try:
                            img = Image.open(img_path)
                            contents.append(img)
                        except Exception as e:
                            raise ValueError(f"Failed to process input image {img_path}: {str(e)}")

            # 2. Configure the SDK to force a native image output
            # and disable all safety filters (as per user request)
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=resolution),
            )

            async def _generate_single():
                """
                Helper function to execute a single generate_content call
                with safety and None checks.
                """
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=config,
                )

                # Guard against empty candidates or blocked responses where response.parts is None
                if not response.candidates:
                    raise RuntimeError("The model returned no response candidates.")

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    finish_reason = getattr(
                        candidate, "finish_reason", FinishReason.FINISH_REASON_UNSPECIFIED
                    )
                    raise RuntimeError(
                        f"Generation stopped or was blocked. Finish reason: {finish_reason.value}"
                    )

                for part in candidate.content.parts:
                    if part.inline_data:
                        return part.as_image()

                raise RuntimeError("No image data found in the response parts.")

            # 3. Execute requests concurrently to fulfill the batch size parameter
            tasks = [_generate_single() for _ in range(batch_size)]
            generated_images = await asyncio.gather(*tasks)
            return generated_images

        except Exception as e:
            raise RuntimeError(f"API Generation Error: {str(e)}")

        finally:
            # 4. Mandatory Cleanup (Only triggers if blobs were actually uploaded)
            for blob in uploaded_blobs:
                try:
                    blob.delete()
                except Exception:
                    pass

    # --- Asynchronous Google Cloud Batch API Helpers ---

    async def submit_batch_job(
        self,
        prompt: str,
        model_name: str,
        batch_size: int,
        resolution: str,
        aspect_ratio: str,
        gcs_bucket_name: str,
        input_image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Builds the JSONL payload, uploads it to GCS, and triggers the Vertex AI Batch job.
        Returns the API-generated Job ID.
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(gcs_bucket_name)
        timestamp = int(time.time())

        # Set up precise routing paths for the Batch input and output folders
        input_file_path = f"batch_inputs/req_{timestamp}.jsonl"
        output_prefix = f"batch_outputs/res_{timestamp}"

        # 1. Prepare multimodal parts (Text + Base64 Reference Images)
        parts = [{"text": prompt}]
        if input_image_paths:
            for img_path in input_image_paths[:16]:
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB to ensure smooth JPEG saving
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")

                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": img_b64}})
                except Exception as e:
                    raise ValueError(f"Failed to process input image {img_path}: {str(e)}")

        # Construct the exact inline REST payload the model expects
        lines = []
        for _ in range(batch_size):
            request_payload = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                        "imageConfig": {
                            "aspectRatio": aspect_ratio,
                            "imageSize": resolution,
                        },
                    },
                    # Explicitly bypass configurable safety filters in the JSON payload
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ],
                }
            }
            lines.append(json.dumps(request_payload))

        jsonl_content = "\n".join(lines)

        # Upload the JSONL configuration to GCS
        input_blob = bucket.blob(input_file_path)
        input_blob.upload_from_string(jsonl_content)
        gcs_input_uri = f"gs://{gcs_bucket_name}/{input_file_path}"
        gcs_output_uri = f"gs://{gcs_bucket_name}/{output_prefix}"

        config = types.CreateBatchJobConfig(
            dest=gcs_output_uri,
        )

        batch_job = await asyncio.to_thread(
            self.client.batches.create,
            model=model_name,
            src=gcs_input_uri,
            config=config,
        )

        # Return the unique job name directly to Gradio for dashboard tracking
        return batch_job.name

    async def get_batch_job_status(self, job_id: str) -> str:
        """
        Queries Vertex AI for the current lifecycle state of a specific Batch job.
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        job_status = await asyncio.to_thread(self.client.batches.get, name=job_id)
        return job_status.state

    async def download_batch_results(self, job_id: str, gcs_bucket_name: str) -> List[Image.Image]:
        """
        Queries GCS for the output JSONL file associated with a completed job,
        decodes the base64 output parts, and transforms them into standard PIL Images.
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        storage_client = storage.Client(project=self.project_id)

        # 1. Fetch the exact job details to extract the destination URI reliably
        job = await asyncio.to_thread(self.client.batches.get, name=job_id)

        # Extract the destination prefix path where Vertex AI wrote the output files
        dest_uri = job.dest.gcs_uri

        if dest_uri.startswith(f"gs://{gcs_bucket_name}/"):
            output_prefix = dest_uri.replace(f"gs://{gcs_bucket_name}/", "")
        else:
            # Fallback in case of an unusual SDK structure
            output_prefix = "batch_outputs/"

        # 2. Iterate through all valid files in the destination path and parse images[cite: 5]
        generated_images = []
        blobs = storage_client.list_blobs(gcs_bucket_name, prefix=output_prefix)

        blob: Blob
        for blob in blobs:
            if blob.name.endswith(".jsonl"):
                content = blob.download_as_text()
                for line in content.strip().split("\n"):
                    if not line:
                        continue

                    result = json.loads(line)

                    # 1. Catch and parse the nested 'status' JSON string from Vertex AI
                    if "status" in result and isinstance(result["status"], str):
                        try:
                            status_obj = json.loads(result["status"])
                            if "message" in status_obj:
                                # Raising this will cause app.py to trigger a gr.Error popup!
                                raise RuntimeError(f"Model Error: {status_obj['message']}")
                        except json.JSONDecodeError:
                            pass

                    # 2. Catch standard explicit API errors (as a fallback)
                    if "error" in result:
                        raise RuntimeError(f"API Error: {json.dumps(result['error'])}")

                    try:
                        # 3. Attempt to extract the image
                        inline_data_base64 = result["response"]["candidates"][0]["content"][
                            "parts"
                        ][0]["inlineData"]["data"]

                        image_bytes = base64.b64decode(inline_data_base64)
                        generated_images.append(Image.open(BytesIO(image_bytes)))
                    except KeyError as e:
                        # 4. Silent fallback for structural issues not caught by the status check
                        print(f"Warning: Unexpected JSON structure missing key {e}. Skipping line.")

        if not generated_images:
            raise RuntimeError(
                "No completed image results found in the designated Google Cloud Storage bucket."
            )

        return generated_images

    async def delete_batch_job_files(self, job_id: str, gcs_bucket_name: str):
        """
        Deletes the input and output files associated with a Batch job from GCS
        to ensure they cannot be re-downloaded.
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(gcs_bucket_name)

        job = await asyncio.to_thread(self.client.batches.get, name=job_id)
        dest_uri = job.dest.gcs_uri

        if dest_uri.startswith(f"gs://{gcs_bucket_name}/"):
            output_prefix = dest_uri.replace(f"gs://{gcs_bucket_name}/", "")
        else:
            output_prefix = "batch_outputs/"

        # 1. Delete all output blobs
        blobs = storage_client.list_blobs(gcs_bucket_name, prefix=output_prefix)
        for blob in blobs:
            blob.delete()

        # 2. Attempt to delete the corresponding input JSONL file based on the timestamp
        if "res_" in output_prefix:
            try:
                timestamp = output_prefix.split("res_")[-1].strip("/")
                input_blob_name = f"batch_inputs/req_{timestamp}.jsonl"
                input_blob = bucket.blob(input_blob_name)
                if input_blob.exists():
                    input_blob.delete()
            except Exception:
                pass


class LocalImageGenerator:
    """
    Handles local execution of .safetensors and .gguf image generation models.
    """

    def __init__(self, model_filename: str):
        import torch
        from config import LOCAL_MODELS_DIR

        self.model_path = os.path.join(LOCAL_MODELS_DIR, model_filename)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Local model not found: {self.model_path}")

        # 1. Device Selection: Target CUDA first, fallback to Vulkan, then CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            self.device = "vulkan"
        else:
            self.device = "cpu"

        # 2. Hardware Acceleration: Use float16 for CUDA and Vulkan to halve VRAM requirements
        self.torch_dtype = torch.float16 if self.device in ["cuda", "vulkan"] else torch.float32

    def _calculate_dimensions(self, resolution: str, aspect_ratio: str):
        import math

        # 1. Determine base target area (1K defaults to 1024x1024)
        base_dim = 1024
        if resolution == "2K":
            base_dim = 2048
        elif resolution == "4K":
            base_dim = 4096

        target_area = base_dim * base_dim

        # 2. Parse the string aspect ratio (e.g., "16:9")
        w_ratio, h_ratio = map(float, aspect_ratio.split(":"))
        ratio = w_ratio / h_ratio

        # 3. Compute dimensions maintaining total pixel count
        ideal_height = math.sqrt(target_area / ratio)
        ideal_width = ideal_height * ratio

        # 4. Stable Diffusion requires dimensions to be multiples of 8
        width = int(round(ideal_width / 8.0) * 8)
        height = int(round(ideal_height / 8.0) * 8)

        return width, height

    async def generate_images_batch(
        self, prompt: str, batch_size: int, resolution: str, aspect_ratio: str
    ) -> List[Image.Image]:
        """Runs the local model generation in a background thread to prevent UI blocking."""
        return await asyncio.to_thread(
            self._generate_sync, prompt, batch_size, resolution, aspect_ratio
        )

    def _generate_sync(self, prompt: str, batch_size: int, resolution: str, aspect_ratio: str):
        import torch

        # Dynamically calculate the aspect-ratio aware dimensions
        width, height = self._calculate_dimensions(resolution, aspect_ratio)

        images = []

        try:
            if self.model_path.endswith(".safetensors"):
                from diffusers import StableDiffusionXLPipeline

                # Load without immediately pushing to a device
                pipe = StableDiffusionXLPipeline.from_single_file(
                    self.model_path, torch_dtype=self.torch_dtype, use_safetensors=True
                )

                # Low VRAM Optimizations (< 6GB)
                if self.device == "cuda":
                    # Offloads sub-models to CPU RAM, only moving them to the GPU
                    # during their active pass.
                    pipe.enable_model_cpu_offload()
                    # Slices and tiles the VAE to prevent OOM crashes
                    # during the final decoding step (crucial for 2K/4K).
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()
                elif self.device == "vulkan":
                    pipe.to("vulkan")
                else:
                    pipe.to("cpu")

                result = pipe(
                    prompt=prompt, num_images_per_prompt=batch_size, width=width, height=height
                )
                images = result.images

                # Aggressive VRAM cleanup
                del pipe
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            elif self.model_path.endswith(".gguf"):
                from stable_diffusion_cpp import StableDiffusion

                # sd.cpp handles memory mapping and Vulkan offloading automatically
                # based on its compilation flags.
                pipe = StableDiffusion(model_path=self.model_path, n_threads=8)
                images = pipe.txt2img(
                    prompt=prompt,
                    sample_steps=20,
                    width=width,
                    height=height,
                    batch_count=batch_size,
                )
                del pipe

            return images
        except Exception as e:
            raise RuntimeError(f"Local Model Inference Error: {str(e)}")
