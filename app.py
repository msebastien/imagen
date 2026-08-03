"""
app.py
Gradio User Interface and event wiring for Imagen AI Studio.
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
        "byteplus_api_key": "",
    }


def save_settings(
    google_api_key: str,
    google_project_id: str,
    gcs_bucket: str,
    use_gcs_for_refs: bool,
    byteplus_api_key: str,
):
    """Saves application settings to a local JSON file."""
    settings = {
        "google_api_key": google_api_key.strip(),
        "google_project_id": google_project_id.strip(),
        "gcs_bucket": gcs_bucket.strip(),
        "use_gcs_for_refs": use_gcs_for_refs,
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


def estimate_cost_display(model: str, resolution: str, batch_size: int) -> str:
    if not batch_size or batch_size < 1:
        batch_size = 1

    cost_per_img = config.COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
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

            client = NanoBananaClient(google_api_key, google_project_id)
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
            stat_key = google_api_key if google_api_key else google_project_id
    except Exception as e:
        raise gr.Error(str(e))

    total_cost = cost_per_img * len(images)

    saved_paths = []
    os.makedirs("outputs", exist_ok=True)

    for i, img in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"outputs/img_{timestamp}_{i}.png"
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


# --- Gradio UI Layout ---
with gr.Blocks() as ui:
    gr.Markdown("# 🍌 Imagen AI Studio")
    gemini_status_indicator = gr.Markdown(config.API_STATUS_MESSAGES["gemini"]["default"])
    byteplus_status_indicator = gr.Markdown(config.API_STATUS_MESSAGES["byteplus"]["default"])

    with gr.Tabs():
        # --- Real-Time Generation Tab ---
        with gr.Tab("Generate (On-Demand)"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_box = gr.Textbox(
                        label="Generation Prompt",
                        placeholder="Describe the image you want to create in detail...",
                        lines=4,
                    )
                    with gr.Row():
                        btn_send = gr.Button("🚀 Send Request", variant="primary")
                        btn_clear_prompt = gr.Button("🗑️ Clear Prompt")

                    # New Engine Toggle
                    engine_radio = gr.Radio(
                        choices=[
                            "Google Cloud (Nano Banana)",
                            "BytePlus Cloud (Seedream)",
                            "Local Inference",
                        ],
                        value="Google Cloud (Nano Banana)",
                        label="Inference Engine",
                    )

                    with gr.Accordion("Advanced Parameters", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=config.GEMINI_IMAGE_MODELS,
                            value=config.GEMINI_IMAGE_MODELS[0],
                            label="Model",
                        )
                        with gr.Row():
                            res_radio = gr.Radio(
                                choices=config.RESOLUTIONS,
                                value="2K",
                                label="Resolution",
                            )
                            ar_dropdown = gr.Dropdown(
                                choices=config.ASPECT_RATIOS,
                                value="16:9",
                                label="Aspect Ratio",
                            )
                            batch_slider = gr.Slider(
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=1,
                                label="Batch Size",
                            )

                        input_gallery = gr.File(
                            label="Input/Reference Images (Max 16)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        cost_indicator = gr.Markdown("**Estimated Cost:** $0.00")

                with gr.Column(scale=3):
                    output_gallery = gr.Gallery(label="Generated Outputs", columns=2, height="auto")

        # --- Batch Queue Tab ---
        with gr.Tab("Batch Queue (Google Models)"):
            gr.Markdown(
                "Submit large or background image tasks to the Google Cloud Batch API. "
                "Enjoy a **50% discount** on generation costs. "
                "Results are saved in Google Cloud Storage."
            )
            with gr.Row():
                # Left Column: Submission Form
                with gr.Column(scale=2):
                    b_prompt_box = gr.Textbox(
                        label="Batch Prompt",
                        placeholder="Describe the batch image generation prompt...",
                        lines=4,
                    )
                    with gr.Row():
                        b_btn_send = gr.Button("🚀 Submit Batch Job", variant="primary")
                        b_btn_clear_prompt = gr.Button("🗑️ Clear Prompt")

                    with gr.Accordion("Advanced Parameters", open=True):
                        b_model_dropdown = gr.Dropdown(
                            choices=config.GEMINI_IMAGE_MODELS,
                            value=config.GEMINI_IMAGE_MODELS[0],
                            label="Model",
                        )
                        with gr.Row():
                            b_res_radio = gr.Radio(
                                choices=config.RESOLUTIONS,
                                value="2K",
                                label="Resolution",
                            )
                            b_ar_dropdown = gr.Dropdown(
                                choices=config.ASPECT_RATIOS,
                                value="16:9",
                                label="Aspect Ratio",
                            )
                            b_batch_slider = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=10,
                                label="Batch Size",
                            )

                        b_input_gallery = gr.File(
                            label="Input/Reference Images (Max 16)",
                            file_count="multiple",
                            file_types=["image"],
                        )

                        b_cost_indicator = gr.Markdown("**Estimated Batch Cost (~50% off):** $0.00")

                    b_status_msg = gr.Textbox(label="Submission Status", interactive=False)

                # Right Column: Dashboard & Fetching
                with gr.Column(scale=3):
                    gr.Markdown("### Active Job Dashboard")
                    job_table = gr.Dataframe(
                        headers=["Job ID", "Prompt Preview", "Status"],
                        interactive=False,
                        wrap=True,
                    )
                    b_refresh_btn = gr.Button("🔄 Refresh Statuses")

                    gr.Markdown("### Manage Completed Results")
                    with gr.Row():
                        fetch_job_id = gr.Textbox(
                            label="Job ID", placeholder="Paste Job ID here...", scale=2
                        )
                        fetch_btn = gr.Button("📥 Download Images", scale=1)
                        discard_btn = gr.Button("🗑️ Discard Job", scale=1, variant="stop")

                    fetch_msg = gr.Textbox(label="Action Status", interactive=False)
                    batch_gallery = gr.Gallery(
                        label="Batch Output Gallery", columns=3, height="auto"
                    )

        # --- History & Cache Tab ---
        with gr.Tab("History & Cache"):
            btn_refresh_history = gr.Button("🔄 Refresh History")

            history_gallery = gr.Gallery(label="Image Cache", columns=4, height="auto")

            history_table = gr.Dataframe(
                headers=["Prompt", "Model", "Resolution", "Date"],
                interactive=False,
                wrap=True,
            )

        # --- Settings Tab ---
        with gr.Tab("Settings"):
            gr.Markdown("### 1. Google Cloud Platform Authentication")
            google_api_key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                value=app_settings.get("google_api_key", ""),
            )
            google_project_id_input = gr.Textbox(
                label="Google Cloud Project ID (Vertex AI Postpay Routing)",
                value=app_settings.get("google_project_id", ""),
            )
            gcs_bucket_input = gr.Textbox(
                label="Google Cloud Storage Bucket Name",
                value=app_settings.get("gcs_bucket", ""),
            )
            use_gcs_for_refs_input = gr.Checkbox(
                label="Upload Real-Time Reference Images to Google Cloud Storage "
                "(Bypasses payload limits)",
                value=app_settings.get("use_gcs_for_refs", False),
            )

            gr.Markdown("### 2. BytePlus Authentication")
            byteplus_api_key_input = gr.Textbox(
                label="BytePlus API Key (Seedream V3 Endpoint)",
                type="password",
                value=app_settings.get("byteplus_api_key", ""),
            )

            with gr.Row():
                btn_save_settings = gr.Button("💾 Save Settings", variant="primary")
                btn_test_conn = gr.Button("Test Connection")

            gr.Markdown("### Cache Management")
            btn_clear_cache = gr.Button("Clear SQLite Image Cache", variant="stop")

        # --- Stats Tab ---
        with gr.Tab("Usage Statistics"):
            gr.Markdown(
                "Metrics are tied to your specific API Key / Project ID. "
                "Monthly counters reset on the 1st."
            )
            with gr.Row():
                stat_tot_img = gr.Number(label="Total Images Generated", interactive=False)
                stat_mon_img = gr.Number(label="Images Generated This Month", interactive=False)
            with gr.Row():
                stat_tot_cost = gr.Textbox(label="Total Cost (USD)", interactive=False)
                stat_mon_cost = gr.Textbox(label="Monthly Cost (USD)", interactive=False)
            btn_refresh_stats = gr.Button("Refresh Statistics")

    # --- Event Wiring ---

    # Real-Time Cost Estimation
    inputs_for_cost = [model_dropdown, res_radio, batch_slider]
    for component in inputs_for_cost:
        component.change(fn=estimate_cost_display, inputs=inputs_for_cost, outputs=cost_indicator)

    # Batch Cost Estimation
    inputs_for_batch_cost = [b_model_dropdown, b_res_radio, b_batch_slider]
    for component in inputs_for_batch_cost:
        component.change(
            fn=estimate_batch_cost_display,
            inputs=inputs_for_batch_cost,
            outputs=b_cost_indicator,
        )

    # Save Settings
    btn_save_settings.click(
        fn=save_settings,
        inputs=[
            google_api_key_input,
            google_project_id_input,
            gcs_bucket_input,
            use_gcs_for_refs_input,
            byteplus_api_key_input,
        ],
    )

    # Connection Test & Clear Prompt
    btn_test_conn.click(
        fn=check_connection_all,
        inputs=[google_api_key_input, google_project_id_input, byteplus_api_key_input],
        outputs=[gemini_status_indicator, byteplus_status_indicator],
    )

    btn_clear_prompt.click(fn=clear_ui_prompt, outputs=prompt_box)
    b_btn_clear_prompt.click(fn=clear_ui_prompt, outputs=b_prompt_box)

    # Cache Management
    btn_clear_cache.click(
        fn=handle_clear_cache,
        outputs=[
            gemini_status_indicator,
            byteplus_status_indicator,
            history_gallery,
            history_table,
        ],
    )

    # Usage Stats
    btn_refresh_stats.click(
        fn=load_initial_stats,
        inputs=[google_api_key_input, google_project_id_input],
        outputs=[stat_tot_img, stat_mon_img, stat_tot_cost, stat_mon_cost],
    )

    # History Refresh
    btn_refresh_history.click(fn=load_history, outputs=[history_gallery, history_table])

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
            return gr.update(
                choices=config.GEMINI_IMAGE_MODELS, value=config.GEMINI_IMAGE_MODELS[0]
            )

    engine_radio.change(fn=update_model_list, inputs=engine_radio, outputs=model_dropdown)

    # Real-Time Generation Execution
    btn_send.click(
        fn=process_generation,
        inputs=[
            engine_radio,
            google_api_key_input,
            google_project_id_input,
            byteplus_api_key_input,
            prompt_box,
            model_dropdown,
            res_radio,
            ar_dropdown,
            batch_slider,
            gcs_bucket_input,
            use_gcs_for_refs_input,
            input_gallery,
        ],
        outputs=[
            output_gallery,
            stat_tot_img,
            stat_mon_img,
            stat_tot_cost,
            stat_mon_cost,
        ],
    ).then(fn=load_history, outputs=[history_gallery, history_table])

    # Batch Job Execution & Dashboard Wiring
    b_btn_send.click(
        fn=submit_batch_task,
        inputs=[
            google_api_key_input,
            google_project_id_input,
            b_prompt_box,
            b_model_dropdown,
            b_res_radio,
            b_ar_dropdown,
            b_batch_slider,
            gcs_bucket_input,
            b_input_gallery,
        ],
        outputs=[job_table, b_status_msg],
    )

    b_refresh_btn.click(
        fn=refresh_job_statuses,
        inputs=[google_api_key_input, google_project_id_input],
        outputs=[job_table],
    )

    # Batch File Fetching & Cleanup
    fetch_btn.click(
        fn=fetch_completed_job,
        inputs=[google_api_key_input, google_project_id_input, fetch_job_id, gcs_bucket_input],
        outputs=[
            batch_gallery,
            fetch_msg,
            stat_tot_img,
            stat_mon_img,
            stat_tot_cost,
            stat_mon_cost,
            job_table,  # Send the updated Dashboard UI table
        ],
    ).then(fn=load_history, outputs=[history_gallery, history_table])

    # Discard Batch Job without downloading
    discard_btn.click(
        fn=discard_batch_job,
        inputs=[google_api_key_input, google_project_id_input, fetch_job_id, gcs_bucket_input],
        outputs=[
            fetch_msg,
            job_table,
        ],
    )

    # Auto-load history and stats on app startup / page refresh
    ui.load(fn=load_history, outputs=[history_gallery, history_table]).then(
        fn=load_initial_stats,
        inputs=[google_api_key_input, google_project_id_input],
        outputs=[stat_tot_img, stat_mon_img, stat_tot_cost, stat_mon_cost],
    )

if __name__ == "__main__":
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Default(primary_hue="blue"),
        share=False,
        debug=True,
    )
