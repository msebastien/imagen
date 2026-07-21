"""
app.py
Gradio User Interface and event wiring.
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


def format_currency(cents: int) -> str:
    return f"${cents / 100:.2f}"


def estimate_cost_display(model: str, resolution: str, batch_size: int) -> str:
    if not batch_size or batch_size < 1:
        batch_size = 1

    cost_per_img = config.COST_TABLE_CENTS.get(model, {}).get(resolution, 0)
    total_cents = cost_per_img * batch_size
    return f"**Estimated Cost:** {format_currency(total_cents)}"


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
        filepath = f"outputs/img_{timestamp}_{i}.jpg"
        img.save(filepath)
        saved_paths.append(filepath)
        # We now pass the model and resolution to the database
        database.cache_image(prompt, filepath, model, resolution)

    stat_key = api_key if api_key else project_id
    database.update_stats(stat_key, len(images), total_cost)

    return saved_paths, get_stats_display(stat_key)


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
        # Add to Gallery with the prompt as a caption
        gallery_items.append((image_path, prompt))
        # Add to Dataframe
        table_data.append([prompt, model, resolution, timestamp])

    return gallery_items, table_data


def handle_clear_cache():
    database.clear_cache()
    # Return a success message, an empty gallery, and an empty dataframe
    return gr.Info("Local cache and images successfully cleared."), [], []


# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as ui:
    gr.Markdown("# 🍌 Nano Banana Vertex AI Studio")
    status_indicator = gr.Markdown("⚪ **API Status:** Waiting for credentials...")

    with gr.Tabs():
        # --- Generation Tab ---
        with gr.Tab("Generate"):
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

        # --- History & Cache Tab (NEW) ---
        with gr.Tab("History & Cache"):
            btn_refresh_history = gr.Button("🔄 Refresh History")

            # Displays the images with the prompt as the hover caption
            history_gallery = gr.Gallery(label="Image Cache", columns=4, height="auto")

            # Displays the exact text metadata for easy reading
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

    inputs_for_cost = [model_dropdown, res_radio, batch_slider]
    for component in inputs_for_cost:
        component.change(
            fn=estimate_cost_display, inputs=inputs_for_cost, outputs=cost_indicator
        )

    btn_test_conn.click(
        fn=check_connection,
        inputs=[api_key_input, project_id_input],
        outputs=status_indicator,
    )

    btn_clear_prompt.click(fn=clear_ui_prompt, outputs=prompt_box)

    # Updated to clear the history tab visuals as well
    btn_clear_cache.click(
        fn=handle_clear_cache,
        outputs=[
            status_indicator,
            history_gallery,
            history_table,
        ],  # We can pipe info to a neutral element or omit, Gradio.Info handles toast natively.
    )

    btn_refresh_stats.click(
        fn=get_stats_display,
        inputs=[api_key_input],
        outputs=[stat_tot_img, stat_mon_img, stat_tot_cost, stat_mon_cost],
    )

    # Load history when the history refresh button is clicked
    btn_refresh_history.click(fn=load_history, outputs=[history_gallery, history_table])

    # Also automatically refresh the history when a new generation finishes
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

    # Auto-load history on app start (useful if there's already cached data)
    ui.load(fn=load_history, outputs=[history_gallery, history_table])

if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1", server_port=7860)
