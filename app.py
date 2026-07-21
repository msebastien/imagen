"""
app.py
Gradio User Interface and event wiring for Nano Banana Vertex AI Studio.
Includes real-time generation, local history/caching, usage statistics,
and asynchronous Google Cloud Batch API processing.
"""

import gradio as gr
import asyncio
import os
from datetime import datetime
from PIL import Image

import database
import config
from api_client import NanoBananaClient

# Initialize local SQLite DB on startup
database.init_db()

# In-memory session job tracking for active Batch API jobs
job_cache = []


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


async def check_connection(api_key: str, project_id: str) -> str:
    client = NanoBananaClient(api_key, project_id)
    is_reachable = await client.check_reachability()
    if is_reachable:
        return "🟢 **API Status:** Connected & Reachable"
    return "🔴 **API Status:** Disconnected / Invalid Credentials"


async def process_generation(
    api_key: str,
    project_id: str,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    batch_size: int,
    input_images: list,
):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not api_key.strip() and not project_id.strip():
        raise gr.Error(
            "Missing Credentials. Please provide an API Key or Project ID in Settings."
        )
    if batch_size < 1 or batch_size > 8:
        raise gr.Error("Batch size must be between 1 and 8.")

    img_paths = [img.name for img in input_images] if input_images else None
    if img_paths and len(img_paths) > 16:
        raise gr.Error("Maximum 16 reference images allowed.")

    try:
        client = NanoBananaClient(api_key, project_id)
        images = await client.generate_images_batch(
            prompt, model, int(batch_size), resolution, aspect_ratio, img_paths
        )
    except Exception as e:
        raise gr.Error(str(e))

    cost_per_img = config.COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
    total_cost = cost_per_img * len(images)

    saved_paths = []
    os.makedirs("outputs", exist_ok=True)

    for i, img in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jpg" if img.format == "JPEG" else img.format.lower()
        filepath = f"outputs/img_{timestamp}_{i}.{ext}"
        img.save(filepath)
        saved_paths.append(filepath)
        database.cache_image(prompt, filepath, model, resolution)

    stat_key = api_key if api_key else project_id
    database.update_stats(stat_key, len(images), total_cost)

    return saved_paths, *get_stats_display(stat_key)


# --- Batch Processing Handler Functions ---


async def submit_batch_task(
    api_key: str,
    project_id: str,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    batch_size: int,
    gcs_bucket: str,
):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not gcs_bucket.strip():
        raise gr.Error("Google Cloud Storage Bucket Name is required for Batch jobs.")
    if not api_key.strip() and not project_id.strip():
        raise gr.Error(
            "Missing Credentials. Please provide an API Key or Project ID in Settings."
        )

    try:
        client = NanoBananaClient(api_key, project_id)
        job_id = await client.submit_batch_job(
            prompt=prompt,
            model_name=model,
            batch_size=int(batch_size),
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            gcs_bucket_name=gcs_bucket,
        )
        job_cache.append([job_id, prompt[:35] + "...", "PENDING", gcs_bucket])
        table_rows = [[j[0], j[1], j[2]] for j in job_cache]
        return (
            gr.update(value=table_rows),
            f"Success! Job '{job_id}' submitted to queue.",
        )
    except Exception as e:
        raise gr.Error(f"Batch Submission Error: {str(e)}")


async def refresh_job_statuses(api_key: str, project_id: str):
    updated_cache = []
    client = NanoBananaClient(api_key, project_id)

    global job_cache
    for job in job_cache:
        job_id, prompt_preview, current_status, bucket = job
        if current_status in [
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        ]:
            updated_cache.append(job)
            continue
        try:
            status = await client.get_batch_job_status(job_id)
            updated_cache.append([job_id, prompt_preview, status, bucket])
        except Exception:
            updated_cache.append(job)

    job_cache = updated_cache
    table_rows = [[j[0], j[1], j[2]] for j in job_cache]
    return gr.update(value=table_rows)


async def fetch_completed_job(
    api_key: str, project_id: str, job_id: str, gcs_bucket: str
):
    if not job_id.strip():
        raise gr.Error("Please enter a valid Job ID.")
    if not gcs_bucket.strip():
        raise gr.Error("Please enter the GCS Bucket Name.")

    try:
        client = NanoBananaClient(api_key, project_id)
        images = await client.download_batch_results(job_id, gcs_bucket)

        saved_paths = []
        os.makedirs("outputs", exist_ok=True)

        for i, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/batch_img_{timestamp}_{i}.jpg"
            img.save(filepath)
            saved_paths.append(filepath)
            database.cache_image(
                "Batch Job: " + job_id, filepath, "Batch Model", "Batch Res"
            )

        return saved_paths, f"Successfully downloaded {len(saved_paths)} images."
    except Exception as e:
        raise gr.Error(f"{str(e)}")


def get_stats_display(key: str):
    stats = database.get_stats(key)
    return (
        stats["total_img"],
        stats["monthly_img"],
        format_currency(stats["total_cost"]),
        format_currency(stats["monthly_cost"]),
    )


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
    return gr.Info("Local cache and images successfully cleared."), [], []


# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as ui:
    gr.Markdown("# 🍌 Nano Banana Vertex AI Studio")
    status_indicator = gr.Markdown("⚪ **API Status:** Waiting for credentials...")

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

                    with gr.Accordion("Advanced Parameters", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=config.AVAILABLE_MODELS,
                            value=config.AVAILABLE_MODELS[0],
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
                    output_gallery = gr.Gallery(
                        label="Generated Outputs", columns=2, height="auto"
                    )

        # --- Batch Queue Tab (NEW) ---
        with gr.Tab("Batch Queue (50% Discount)"):
            gr.Markdown(
                "Submit large or background image tasks to the Google Cloud Batch API. "
                "Enjoy a **50% discount** on generation costs. Results are saved in Google Cloud Storage."
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
                            choices=config.AVAILABLE_MODELS,
                            value=config.AVAILABLE_MODELS[0],
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

                        b_bucket_input = gr.Textbox(
                            label="Google Cloud Storage Bucket Name",
                            placeholder="e.g. my-imagen-batch-bucket",
                        )
                        b_cost_indicator = gr.Markdown(
                            "**Estimated Batch Cost (~50% off):** $0.00"
                        )

                    b_status_msg = gr.Textbox(
                        label="Submission Status", interactive=False
                    )

                # Right Column: Dashboard & Fetching
                with gr.Column(scale=3):
                    gr.Markdown("### Active Job Dashboard")
                    job_table = gr.Dataframe(
                        headers=["Job ID", "Prompt Preview", "Status"],
                        interactive=False,
                        wrap=True,
                    )
                    b_refresh_btn = gr.Button("🔄 Refresh Statuses")

                    gr.Markdown("### Download Completed Results")
                    with gr.Row():
                        fetch_job_id = gr.Textbox(
                            label="Job ID", placeholder="Paste Job ID here...", scale=3
                        )
                        fetch_btn = gr.Button("📥 Download Images", scale=1)

                    fetch_msg = gr.Textbox(label="Download Status", interactive=False)
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
            gr.Markdown("### Authentication & Routing")
            api_key_input = gr.Textbox(
                label="API Key", type="password", placeholder="Enter Gemini API Key..."
            )
            project_id_input = gr.Textbox(
                label="Google Cloud Project ID (Vertex AI Postpay Routing)",
                placeholder="YOUR_PROJECT_ID (Optional)",
            )
            btn_test_conn = gr.Button("Test Connection")

            gr.Markdown("### Cache Management")
            btn_clear_cache = gr.Button("Clear SQLite Image Cache", variant="stop")

        # --- Stats Tab ---
        with gr.Tab("Usage Statistics"):
            gr.Markdown(
                "Metrics are tied to your specific API Key / Project ID. Monthly counters reset on the 1st."
            )
            with gr.Row():
                stat_tot_img = gr.Number(
                    label="Total Images Generated", interactive=False
                )
                stat_mon_img = gr.Number(
                    label="Images Generated This Month", interactive=False
                )
            with gr.Row():
                stat_tot_cost = gr.Textbox(label="Total Cost (USD)", interactive=False)
                stat_mon_cost = gr.Textbox(
                    label="Monthly Cost (USD)", interactive=False
                )
            btn_refresh_stats = gr.Button("Refresh Statistics")

    # --- Event Wiring ---

    # Real-Time Cost Estimation
    inputs_for_cost = [model_dropdown, res_radio, batch_slider]
    for component in inputs_for_cost:
        component.change(
            fn=estimate_cost_display, inputs=inputs_for_cost, outputs=cost_indicator
        )

    # Batch Cost Estimation
    inputs_for_batch_cost = [b_model_dropdown, b_res_radio, b_batch_slider]
    for component in inputs_for_batch_cost:
        component.change(
            fn=estimate_batch_cost_display,
            inputs=inputs_for_batch_cost,
            outputs=b_cost_indicator,
        )

    # Connection Test & Clear Prompt
    btn_test_conn.click(
        fn=check_connection,
        inputs=[api_key_input, project_id_input],
        outputs=status_indicator,
    )

    btn_clear_prompt.click(fn=clear_ui_prompt, outputs=prompt_box)
    b_btn_clear_prompt.click(fn=clear_ui_prompt, outputs=b_prompt_box)

    # Cache Management
    btn_clear_cache.click(
        fn=handle_clear_cache,
        outputs=[
            status_indicator,
            history_gallery,
            history_table,
        ],
    )

    # Usage Stats
    btn_refresh_stats.click(
        fn=get_stats_display,
        inputs=[api_key_input],
        outputs=[stat_tot_img, stat_mon_img, stat_tot_cost, stat_mon_cost],
    )

    # History Refresh
    btn_refresh_history.click(fn=load_history, outputs=[history_gallery, history_table])

    # Real-Time Generation Execution
    btn_send.click(
        fn=process_generation,
        inputs=[
            api_key_input,
            project_id_input,
            prompt_box,
            model_dropdown,
            res_radio,
            ar_dropdown,
            batch_slider,
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
            api_key_input,
            project_id_input,
            b_prompt_box,
            b_model_dropdown,
            b_res_radio,
            b_ar_dropdown,
            b_batch_slider,
            b_bucket_input,
        ],
        outputs=[job_table, b_status_msg],
    )

    b_refresh_btn.click(
        fn=refresh_job_statuses,
        inputs=[api_key_input, project_id_input],
        outputs=[job_table],
    )

    fetch_btn.click(
        fn=fetch_completed_job,
        inputs=[api_key_input, project_id_input, fetch_job_id, b_bucket_input],
        outputs=[batch_gallery, fetch_msg],
    ).then(fn=load_history, outputs=[history_gallery, history_table])

    # Auto-load history on app startup
    ui.load(fn=load_history, outputs=[history_gallery, history_table])

if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1", server_port=7860)
