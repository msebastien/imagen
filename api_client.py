"""
api_client.py
Handles asynchronous interactions with the Nano Banana API via Vertex AI.
"""

import json
import time
import asyncio
from typing import List, Optional
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image
from google.cloud import storage
from google.genai import types


class NanoBananaClient:
    # 1. Add location as a parameter (defaulting to us-central1 for maximum model compatibility)
    def __init__(
        self, api_key: str = "", project_id: str = "", location: str = "global"
    ):
        self.api_key = api_key.strip()
        self.project_id = project_id.strip()
        self.location = location.strip()
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initializes the Agent Platform client. Routes through Google Cloud if Project ID is provided."""
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
        input_image_paths: Optional[List[str]] = None,
    ) -> List[Image.Image]:
        """
        Executes a batch image generation request using Gemini's native multimodal capabilities.
        """
        if not self.client:
            raise ValueError("API Client not initialized. Please configure Settings.")

        # 1. Compile the multimodal contents list (Text + Optional Reference Images)
        contents = [prompt]
        if input_image_paths:
            for img_path in input_image_paths[:16]:
                try:
                    img = Image.open(img_path)
                    contents.append(img)
                except Exception as e:
                    raise ValueError(
                        f"Failed to process input image {img_path}: {str(e)}"
                    )

        # 2. Configure the SDK to force a native image output from the Gemini model
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio, image_size=resolution
            ),
        )

        async def _generate_single():
            """Helper function to execute a single generate_content call with safety and None checks."""
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
                finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                raise RuntimeError(
                    f"Generation stopped or was blocked. Finish reason: {finish_reason}"
                )

            for part in candidate.content.parts:
                if part.inline_data:
                    return part.as_image()

            raise RuntimeError("No image data found in the response parts.")

        # 3. Execute requests concurrently to fulfill the batch size parameter
        try:
            tasks = [_generate_single() for _ in range(batch_size)]
            generated_images = await asyncio.gather(*tasks)
            return generated_images
        except Exception as e:
            raise RuntimeError(f"API Generation Error: {str(e)}")

    async def generate_images_via_batch(
        self,
        prompt: str,
        model_name: str,
        batch_size: int,
        resolution: str,
        aspect_ratio: str,
        gcs_bucket_name: str,
    ) -> List[Image.Image]:
        """
        Executes an image generation request via the Google Cloud Batch API.
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(gcs_bucket_name)
        timestamp = int(time.time())

        # 1. Prepare the JSONL Input Data
        # Batch inference expects input in JSON Lines format, where each line is a separate request.
        input_file_path = f"batch_inputs/req_{timestamp}.jsonl"
        output_prefix = f"batch_outputs/res_{timestamp}"

        lines = []
        for _ in range(batch_size):
            # Format the request exactly as the model expects it inline
            request_payload = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    # Pass your configuration matching the GenerateContentConfig structure
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                        "imageConfig": {
                            "aspectRatio": aspect_ratio,
                            "imageSize": resolution,
                        },
                    },
                }
            }
            lines.append(json.dumps(request_payload))

        jsonl_content = "\n".join(lines)

        # 2. Upload Input to Cloud Storage
        input_blob = bucket.blob(input_file_path)
        input_blob.upload_from_string(jsonl_content)
        gcs_input_uri = f"gs://{gcs_bucket_name}/{input_file_path}"
        gcs_output_uri = f"gs://{gcs_bucket_name}/{output_prefix}"

        # 3. Create the Batch Prediction Job
        # We use the client.batches.create method to trigger the job.
        config = types.CreateBatchJobConfig(
            dest=gcs_output_uri,
        )

        batch_job = await asyncio.to_thread(
            self.client.batches.create,
            model=model_name,
            src=gcs_input_uri,
            config=config,
        )

        # 4. Poll for Job Completion
        # We must periodically check the job state until it completes.
        while True:
            job_status = await asyncio.to_thread(
                self.client.batches.get, name=batch_job.name
            )
            if job_status.state in [
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ]:
                break
            await asyncio.sleep(30)  # Poll every 30 seconds

        if job_status.state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch job ended with status: {job_status.state}")

        # 5. Retrieve and Parse Batch Output
        generated_images = []
        blobs = storage_client.list_blobs(gcs_bucket_name, prefix=output_prefix)

        for blob in blobs:
            if blob.name.endswith(".jsonl"):
                content = blob.download_as_text()
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    result = json.loads(line)

                    # Extract the Base64 image data from the returned parts payload
                    try:
                        inline_data_base64 = result["response"]["candidates"][0][
                            "content"
                        ]["parts"][0]["inlineData"]["data"]
                        import base64

                        image_bytes = base64.b64decode(inline_data_base64)
                        generated_images.append(Image.open(BytesIO(image_bytes)))
                    except KeyError:
                        print("Warning: A batch line failed or returned no image.")

        return generated_images
