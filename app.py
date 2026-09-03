"""
app.py
Backend controller and business logic for Imagen AI Studio.
Includes real-time generation, local history/caching, usage statistics,
Google Cloud Batch API processing, and BytePlus Seedream integration.
"""

import gradio as gr
import os
import json
from datetime import datetime

import database
import config
from api_client import NanoBananaClient, BytePlusClient
from ui_layout import UILayout

# Initialize local SQLite DB on startup
database.init_db()

# In-memory session job tracking for active Batch API jobs
job_cache = []


def load_settings():
    """Loads application settings from a local JSON file."""
    if os.path.exists(config.SETTINGS_FILE):
        try:
            with open(config.SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "google_api_key": "",
        "google_project_id": "",
        "gcs_bucket": "",
        "use_gcs_for_refs": False,
        "use_flex_paygo": False,
        "byteplus_api_key": "",
    }


def load_settings_ui():
    """Loads settings from settings.json to dynamically populate UI components on page reload."""
    s = load_settings()
    return (
        s.get("google_api_key", ""),
        s.get("google_project_id", ""),
        s.get("gcs_bucket", ""),
        s.get("use_gcs_for_refs", False),
        s.get("use_flex_paygo", False),
        s.get("byteplus_api_key", ""),
    )


def save_settings(
    google_api_key: str,
    google_project_id: str,
    gcs_bucket: str,
    use_gcs_for_refs: bool,
    use_flex_paygo: bool,
    byteplus_api_key: str,
):
    """Saves application settings to a local JSON file."""
    settings = {
        "google_api_key": google_api_key.strip(),
        "google_project_id": google_project_id.strip(),
        "gcs_bucket": gcs_bucket.strip(),
        "use_gcs_for_refs": use_gcs_for_refs,
        "use_flex_paygo": use_flex_paygo,
        "byteplus_api_key": byteplus_api_key.strip(),
    }
    try:
        with open(config.SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        gr.Info("Settings successfully saved to settings.json!")
    except Exception as e:
        raise gr.Error(f"Failed to save settings: {str(e)}")


# Load persistent settings on application startup
app_settings = load_settings()


def format_currency(cents: int) -> str:
    return f"${cents / 100:.2f}"


def estimate_cost_display(
    engine: str, model: str, resolution: str, batch_size: int, use_flex_paygo: bool
) -> str:
    if not batch_size or batch_size < 1:
        batch_size = 1

    cost_per_img = config.COST_TABLE_CENTS.get(model, {}).get(resolution, 0)

    # Apply Flex PayGo 50% discount if utilizing Google Cloud engine and enabled in settings
    if engine == "Google Cloud (Nano Banana)" and use_flex_paygo:
        cost_per_img = cost_per_img * config.FLEX_PAYGO_DISCOUNT

    total_cents = cost_per_img * batch_size
    return f"**Estimated Cost:** {format_currency(total_cents)}"


def estimate_batch_cost_display(model: str, resolution: str, batch_size: int) -> str:
    if not batch_size or batch_size < 1:
        batch_size = 1

    # Apply 50% Batch API discount
    cost_per_img = config.BATCH_COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
    total_cents = int(cost_per_img * batch_size)
    return f"**Estimated Batch Cost (~50% off):** {format_currency(total_cents)}"


async def check_connection_gemini(google_api_key: str, google_project_id: str) -> str:
    """
    Checks the connection status of the Gemini API.
    Args:
        google_api_key (str): The Google API key.
        google_project_id (str): The Google project ID.

    Returns:
        str: A status message indicating whether the Gemini API is reachable or not.
    """
    client = NanoBananaClient(google_api_key, google_project_id)
    is_reachable = await client.check_reachability()
    if is_reachable:
        return config.API_STATUS_MESSAGES["gemini"]["success"]
    return config.API_STATUS_MESSAGES["gemini"]["failure"]


async def check_connection_byteplus(byteplus_api_key: str) -> str:
    """
    Checks the connection status of the BytePlus API.
    Args:
        byteplus_api_key (str): The BytePlus API key.

    Returns:
        str: A status message indicating whether the BytePlus API is reachable or not.
    """
    client = BytePlusClient(byteplus_api_key)
    is_reachable = await client.check_reachability()
    if is_reachable:
        return config.API_STATUS_MESSAGES["byteplus"]["success"]
    return config.API_STATUS_MESSAGES["byteplus"]["failure"]


async def check_connection_all(
    google_api_key: str, google_project_id: str, byteplus_api_key: str
) -> tuple:
    """
    Checks the connection status of both Gemini and BytePlus APIs.
    Args:
        google_api_key (str): The Google API key.
        google_project_id (str): The Google project ID.
        byteplus_api_key (str): The BytePlus API key.

    Returns:
        tuple: A tuple containing the status messages for both APIs.
    """
    gemini_status = await check_connection_gemini(google_api_key, google_project_id)
    byteplus_status = await check_connection_byteplus(byteplus_api_key)
    return gemini_status, byteplus_status


async def process_generation(
    engine: str,
    google_api_key: str,
    google_project_id: str,
    byteplus_api_key: str,
    use_flex_paygo: bool,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    batch_size: int,
    gcs_bucket: str,
    use_gcs_for_refs: bool,
    input_images: list,
):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if batch_size < 1 or batch_size > 8:
        raise gr.Error("Batch size must be between 1 and 8.")

    try:
        # Local
        if engine == "Local Inference":
            # Local inference path using the LocalImageGenerator class
            from api_client import LocalImageGenerator

            # Bypass cloud authentication and GCS completely
            local_client = LocalImageGenerator(model)
            images = await local_client.generate_images_batch(
                prompt, int(batch_size), resolution, aspect_ratio
            )
            # Local models cost $0
            cost_per_img = 0
            stat_key = "LOCAL_COMPUTE"

        # BytePlus ModelArk
        elif engine == "BytePlus Cloud (Seedream)":
            if not byteplus_api_key.strip():
                raise gr.Error("Missing BytePlus API Key. Please provide it in Settings.")

            img_paths = [img.name for img in input_images] if input_images else None

            if img_paths and len(img_paths) > 14:
                raise gr.Error("Maximum 14 reference images allowed for Seedream models.")

            bp_client = BytePlusClient(byteplus_api_key)
            images = await bp_client.generate_images_batch(
                prompt, model, int(batch_size), resolution, aspect_ratio, img_paths
            )
            cost_per_img = config.COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
            stat_key = byteplus_api_key

        # Google AI Platform (Vertex AI) / AI Studio
        else:
            # Existing Cloud API Logic
            if not google_api_key.strip() and not google_project_id.strip():
                raise gr.Error(
                    "Missing Credentials. Please provide an API Key or Project ID in Settings."
                )

            # Catch bucket requirement early if the user checked the GCS toggle
            img_paths = [img.name for img in input_images] if input_images else None
            if img_paths and use_gcs_for_refs and not gcs_bucket.strip():
                raise gr.Error(
                    "A GCS Bucket Name is required when GCS reference uploading is enabled."
                )

            if img_paths and len(img_paths) > 16:
                raise gr.Error("Maximum 16 reference images allowed.")

            client = NanoBananaClient(
                api_key=google_api_key, project_id=google_project_id, use_flex_paygo=use_flex_paygo
            )
            images = await client.generate_images_batch(
                prompt,
                model,
                int(batch_size),
                resolution,
                aspect_ratio,
                gcs_bucket,
                use_gcs_for_refs,
                img_paths,
            )

            cost_per_img = config.COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
            if use_flex_paygo:
                cost_per_img = cost_per_img * config.FLEX_PAYGO_DISCOUNT
            stat_key = google_api_key if google_api_key else google_project_id
    except Exception as e:
        raise gr.Error(str(e))

    total_cost = cost_per_img * len(images)

    saved_paths = []
    os.makedirs("outputs", exist_ok=True)

    for i, img in enumerate(images):
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"outputs/img_{date}_{i}.png"
        img.save(filepath)
        saved_paths.append(filepath)
        database.cache_image(prompt, filepath, model, resolution)

    database.update_stats(stat_key, len(images), total_cost)

    return saved_paths, *get_stats_display(stat_key)


# --- Batch Processing Handler Functions (Google Only) ---


async def submit_batch_task(
    google_api_key: str,
    google_project_id: str,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    batch_size: int,
    gcs_bucket: str,
    input_images: list,
):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not gcs_bucket.strip():
        raise gr.Error("Google Cloud Storage Bucket Name is required for Batch jobs.")
    if not google_api_key.strip() and not google_project_id.strip():
        raise gr.Error("Missing Credentials. Please provide an API Key or Project ID in Settings.")

    img_paths = [img.name for img in input_images] if input_images else None
    if img_paths and len(img_paths) > 16:
        raise gr.Error("Maximum 16 reference images allowed.")

    try:
        client = NanoBananaClient(google_api_key, google_project_id)
        job_id = await client.submit_batch_job(
            prompt=prompt,
            model_name=model,
            batch_size=int(batch_size),
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            gcs_bucket_name=gcs_bucket,
            input_image_paths=img_paths,
        )
        # Store model and resolution in job_cache for usage stat calculations later
        job_cache.append([job_id, prompt, "PENDING", gcs_bucket, model, resolution])

        # Update the Gradio Dataframe with the new job entry and a preview of the prompt
        table_rows = [
            [j[0], j[1][:35] + "..." if len(prompt) > 35 else prompt, j[2]] for j in job_cache
        ]
        return (
            gr.update(value=table_rows),
            f"Success! Job '{job_id}' submitted to queue.",
        )
    except Exception as e:
        raise gr.Error(f"Batch Submission Error: {str(e)}")


async def refresh_job_statuses(google_api_key: str, google_project_id: str):
    updated_cache = []
    client = NanoBananaClient(google_api_key, google_project_id)

    global job_cache
    for job in job_cache:
        job_id = job[0]
        prompt = job[1]
        prompt_preview = prompt[:35] + "..." if len(prompt) > 35 else prompt
        current_status = job[2]
        bucket = job[3]
        model = job[4] if len(job) > 4 else config.GEMINI_IMAGE_MODELS[0]
        resolution = job[5] if len(job) > 5 else "1K"

        if current_status in [
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        ]:
            updated_cache.append(job)
            continue
        try:
            status = await client.get_batch_job_status(job_id)
            updated_cache.append(
                [
                    job_id,
                    prompt_preview,
                    status.split("_")[-1],
                    bucket,
                    model,
                    resolution,
                ]
            )
        except Exception:
            updated_cache.append(job)

    job_cache = updated_cache
    table_rows = [[j[0], j[1], j[2]] for j in job_cache]
    return gr.update(value=table_rows)


async def fetch_completed_job(
    google_api_key: str, google_project_id: str, job_id: str, gcs_bucket: str
):
    if not job_id.strip():
        raise gr.Error("Please enter a valid Job ID.")
    if not gcs_bucket.strip():
        raise gr.Error("Please enter the GCS Bucket Name.")

    global job_cache

    # 1. Explicitly check if the job exists in the active session cache
    # (prevents re-downloading consumed/invalid jobs)
    matched_job = None
    for job in job_cache:
        if job[0] == job_id:
            matched_job = job
            break

    if not matched_job:
        raise gr.Error(
            "Error: This job has already been consumed, does not exist, "
            "or is no longer active in the queue."
        )

    try:
        client = NanoBananaClient(google_api_key, google_project_id)

        # 2. Download images from GCS(will raise RuntimeError if missing/already deleted)
        images = await client.download_batch_results(job_id, gcs_bucket)

        # 3. Extract model and resolution metadata from the matched cache entry
        prompt = matched_job[1] if len(matched_job) > 1 else "Unknown Prompt"
        model = matched_job[4] if len(matched_job) > 4 else config.GEMINI_IMAGE_MODELS[0]
        resolution = matched_job[5] if len(matched_job) > 5 else "1K"

        saved_paths = []
        os.makedirs("outputs", exist_ok=True)

        for i, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/batch_img_{timestamp}_{i}.png"
            img.save(filepath)
            saved_paths.append(filepath)
            database.cache_image("Batch" + "|" + job_id + "|" + prompt, filepath, model, resolution)

        # 4. Calculate discounted Batch API cost and update usage statistics
        # (only runs on first valid fetch)
        cost_per_img = config.BATCH_COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
        total_cost = int(cost_per_img * len(images))

        stat_key = google_api_key if google_api_key else google_project_id
        database.update_stats(stat_key, len(images), total_cost)

        # 5. Clean up GCS files so they cannot be fetched again
        await client.delete_batch_job_files(job_id, gcs_bucket)

        # 6. Remove job from the active dashboard permanently (job_cache)
        job_cache = [j for j in job_cache if j[0] != job_id]
        table_rows = [[j[0], j[1], j[2]] for j in job_cache]

        return (
            saved_paths,
            f"Successfully downloaded {len(saved_paths)} images. Job consumed and removed.",
            *get_stats_display(stat_key),
            gr.update(value=table_rows),  # Pass the fresh table data to Gradio
        )
    except Exception as e:
        raise gr.Error(f"{str(e)}")


async def discard_batch_job(
    google_api_key: str, google_project_id: str, job_id: str, gcs_bucket: str
):
    if not job_id.strip():
        raise gr.Error("Please enter a valid Job ID.")
    if not gcs_bucket.strip():
        raise gr.Error("Please enter the GCS Bucket Name.")

    global job_cache

    try:
        client = NanoBananaClient(google_api_key, google_project_id)
        # 1. Clean up input and output JSONL artifacts from GCS
        await client.delete_batch_job_files(job_id, gcs_bucket)
    except Exception:
        # If files were already deleted or missing in GCS, log error and proceed to purge cache
        pass

    # 2. Remove job from active session dashboard tracking
    job_cache = [j for j in job_cache if j[0] != job_id]
    table_rows = [[j[0], j[1][:35] + "...", j[2]] for j in job_cache]

    return (
        f"Job '{job_id}' discarded and GCS files cleaned up.",
        gr.update(value=table_rows),
    )


def get_stats_display(key: str):
    stats = database.get_stats(key)
    return (
        stats["total_img"],
        stats["monthly_img"],
        format_currency(stats["total_cost"]),
        format_currency(stats["monthly_cost"]),
    )


def load_initial_stats(google_api_key: str, google_project_id: str):
    """Fetches usage stats based on the available active credential."""
    stat_key = google_api_key if google_api_key else google_project_id
    return get_stats_display(stat_key)


def clear_ui_prompt():
    return ""


def load_history():
    """Fetches history from the database and formats it for the Gallery and Dataframe."""
    rows = database.get_cached_history()
    gallery_items = []
    table_data = []

    for row in rows:
        image_path, prompt, model, resolution, timestamp = row
        gallery_items.append((image_path, prompt))
        table_data.append([prompt, model, resolution, timestamp])

    return gallery_items, table_data


def handle_clear_cache():
    database.clear_cache()
    gr.Info("Local cache and images successfully cleared.")
    return (
        config.API_STATUS_MESSAGES["gemini"]["default"],
        config.API_STATUS_MESSAGES["byteplus"]["default"],
        [],
        [],
    )


# Dynamic Model List based on Engine
def update_model_list(engine_choice):
    if engine_choice == "Local Inference":
        local_models = config.get_local_models()
        if not local_models:
            return gr.update(
                choices=["NO MODELS FOUND IN /models"], value="NO MODELS FOUND IN /models"
            )
        return gr.update(choices=local_models, value=local_models[0])
    elif engine_choice == "BytePlus Cloud (Seedream)":
        return gr.update(choices=config.SEEDREAM_MODELS, value=config.SEEDREAM_MODELS[0])
    else:
        return gr.update(choices=config.GEMINI_IMAGE_MODELS, value=config.GEMINI_IMAGE_MODELS[0])


# --- Application Initialization ---
app_handlers = {
    "estimate_cost_display": estimate_cost_display,
    "estimate_batch_cost_display": estimate_batch_cost_display,
    "save_settings": save_settings,
    "check_connection_all": check_connection_all,
    "clear_ui_prompt": clear_ui_prompt,
    "handle_clear_cache": handle_clear_cache,
    "load_initial_stats": load_initial_stats,
    "load_history": load_history,
    "update_model_list": update_model_list,
    "process_generation": process_generation,
    "submit_batch_task": submit_batch_task,
    "refresh_job_statuses": refresh_job_statuses,
    "fetch_completed_job": fetch_completed_job,
    "discard_batch_job": discard_batch_job,
    "load_settings_ui": load_settings_ui,
}

ui_layout = UILayout(app_settings, app_handlers)
ui = ui_layout.build()


if __name__ == "__main__":
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Default(primary_hue="blue"),
        share=False,
        debug=True,
    )
