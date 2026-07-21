"""
api_client.py
Handles asynchronous interactions with the Nano Banana API
via Google Cloud AI Platform (formerly Vertex AI).
Includes decoupled methods for standard Generation and Google Cloud Batch API processing.
"""

import json
import time
import asyncio
import base64
from typing import List, Optional
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image
from google.cloud import storage
from google.cloud.storage.blob import Blob


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
        Executes a real-time (on-demand) batchimage generation request,
        using Gemini's native multimodal capabilities.
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

    # --- Asynchronous Google Cloud Batch API Helpers ---

    async def submit_batch_job(
        self,
        prompt: str,
        model_name: str,
        batch_size: int,
        resolution: str,
        aspect_ratio: str,
        gcs_bucket_name: str,
    ) -> str:
        """
        Builds the JSONL payload, uploads it to GCS, and triggers the Vertex AI Batch job.
        Returns the API-generated Job ID[cite: 4, 5].
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(gcs_bucket_name)
        timestamp = int(time.time())

        # Set up precise routing paths for the Batch input and output folders
        input_file_path = f"batch_inputs/req_{timestamp}.jsonl"
        output_prefix = f"batch_outputs/res_{timestamp}"

        # Construct the exact inline REST payload the model expects[cite: 5]
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
        Queries Vertex AI for the current lifecycle state of a specific Batch job[cite: 4].
        """
        if not self.client:
            raise ValueError("API Client not initialized.")

        job_status = await asyncio.to_thread(self.client.batches.get, name=job_id)
        return job_status.state

    async def download_batch_results(
        self, job_id: str, gcs_bucket_name: str
    ) -> List[Image.Image]:
        """
        Queries GCS for the output JSONL file associated with a completed job,
        decodes the base64 output parts, and transforms them into standard PIL Images[cite: 5].
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
                    try:
                        inline_data_base64 = result["response"]["candidates"][0][
                            "content"
                        ]["parts"][0]["inlineData"]["data"]

                        image_bytes = base64.b64decode(inline_data_base64)
                        generated_images.append(Image.open(BytesIO(image_bytes)))
                    except KeyError:
                        print(
                            "Warning: A specific generation line failed or was blocked by safety filters."
                        )

        if not generated_images:
            raise RuntimeError(
                "No completed image results found in the designated Google Cloud Storage bucket."
            )

        return generated_images
