"""
ui_layout.py
Dedicated UI builder classes for Imagen AI Studio.
Encapsulates Gradio layout construction and event wiring for modularity.
"""

import gradio as gr
import config


class UILayout:
    """
    Constructs the Gradio user interface layout and maps interaction events
    to the provided backend controller handlers.
    """

    def __init__(self, app_settings: dict, handlers: dict):
        self.settings = app_settings
        self.handlers = handlers
        self.ui = gr.Blocks()

    def build(self) -> gr.Blocks:
        with self.ui:
            self._build_header()

            with gr.Tabs():
                self._build_realtime_tab()
                self._build_batch_tab()
                self._build_history_tab()
                self._build_settings_tab()
                self._build_stats_tab()

            self._wire_events()

        return self.ui

    def _build_header(self):
        gr.Markdown("# 🍌 Imagen AI Studio")
        self.gemini_status = gr.Markdown(config.API_STATUS_MESSAGES["gemini"]["default"])
        self.byteplus_status = gr.Markdown(config.API_STATUS_MESSAGES["byteplus"]["default"])

    def _build_realtime_tab(self):
        with gr.Tab("Generate (On-Demand)"):
            with gr.Row():
                with gr.Column(scale=2):
                    self.prompt_box = gr.Textbox(
                        label="Generation Prompt",
                        placeholder="Describe the image you want to create in detail...",
                        lines=4,
                    )
                    with gr.Row():
                        self.btn_send = gr.Button("🚀 Send Request", variant="primary")
                        self.btn_clear_prompt = gr.Button("🗑️ Clear Prompt")

                    self.engine_radio = gr.Radio(
                        choices=[
                            "Google Cloud (Nano Banana)",
                            "BytePlus Cloud (Seedream)",
                            "Local Inference",
                        ],
                        value="Google Cloud (Nano Banana)",
                        label="Inference Engine",
                    )

                    with gr.Accordion("Advanced Parameters", open=True):
                        self.model_dropdown = gr.Dropdown(
                            choices=config.GEMINI_IMAGE_MODELS,
                            value=config.GEMINI_IMAGE_MODELS[0],
                            label="Model",
                        )
                        with gr.Row():
                            self.res_radio = gr.Radio(
                                choices=config.RESOLUTIONS,
                                value="2K",
                                label="Resolution",
                            )
                            self.ar_dropdown = gr.Dropdown(
                                choices=config.ASPECT_RATIOS,
                                value="16:9",
                                label="Aspect Ratio",
                            )
                            self.batch_slider = gr.Slider(
                                minimum=1, maximum=8, step=1, value=1, label="Batch Size"
                            )

                        self.input_gallery = gr.File(
                            label="Input/Reference Images (Max 16)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        self.cost_indicator = gr.Markdown("**Estimated Cost:** $0.00")

                with gr.Column(scale=3):
                    self.output_gallery = gr.Gallery(
                        label="Generated Outputs", columns=2, height="auto"
                    )

    def _build_batch_tab(self):
        with gr.Tab("Batch Queue (Google Models)"):
            gr.Markdown(
                "Submit large or background image tasks to the Google Cloud Batch API. "
                "Enjoy a **50% discount** on generation costs. "
                "Results are saved in Google Cloud Storage."
            )
            with gr.Row():
                # Left Column: Submission Form
                with gr.Column(scale=2):
                    self.b_prompt_box = gr.Textbox(
                        label="Batch Prompt",
                        placeholder="Describe the batch image generation prompt...",
                        lines=4,
                    )
                    with gr.Row():
                        self.b_btn_send = gr.Button("🚀 Submit Batch Job", variant="primary")
                        self.b_btn_clear_prompt = gr.Button("🗑️ Clear Prompt")

                    with gr.Accordion("Advanced Parameters", open=True):
                        self.b_model_dropdown = gr.Dropdown(
                            choices=config.GEMINI_IMAGE_MODELS,
                            value=config.GEMINI_IMAGE_MODELS[0],
                            label="Model",
                        )
                        with gr.Row():
                            self.b_res_radio = gr.Radio(
                                choices=config.RESOLUTIONS, value="2K", label="Resolution"
                            )
                            self.b_ar_dropdown = gr.Dropdown(
                                choices=config.ASPECT_RATIOS, value="16:9", label="Aspect Ratio"
                            )
                            self.b_batch_slider = gr.Slider(
                                minimum=1, maximum=100, step=1, value=10, label="Batch Size"
                            )

                        self.b_input_gallery = gr.File(
                            label="Input/Reference Images (Max 16)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        self.b_cost_indicator = gr.Markdown(
                            "**Estimated Batch Cost (~50% off):** $0.00"
                        )

                    self.b_status_msg = gr.Textbox(label="Submission Status", interactive=False)

                # Right Column: Dashboard & Fetching
                with gr.Column(scale=3):
                    gr.Markdown("### Active Job Dashboard")
                    self.job_table = gr.Dataframe(
                        headers=["Job ID", "Prompt Preview", "Status"], interactive=False, wrap=True
                    )
                    self.b_refresh_btn = gr.Button("🔄 Refresh Statuses")

                    gr.Markdown("### Manage Completed Results")
                    with gr.Row():
                        self.fetch_job_id = gr.Textbox(
                            label="Job ID", placeholder="Paste Job ID here...", scale=2
                        )
                        self.fetch_btn = gr.Button("📥 Download Images", scale=1)
                        self.discard_btn = gr.Button("🗑️ Discard Job", scale=1, variant="stop")

                    self.fetch_msg = gr.Textbox(label="Action Status", interactive=False)
                    self.batch_gallery = gr.Gallery(
                        label="Batch Output Gallery", columns=3, height="auto"
                    )

    def _build_history_tab(self):
        with gr.Tab("History & Cache"):
            self.btn_refresh_history = gr.Button("🔄 Refresh History")
            self.history_gallery = gr.Gallery(label="Image Cache", columns=4, height="auto")
            self.history_table = gr.Dataframe(
                headers=["Prompt", "Model", "Resolution", "Date"], interactive=False, wrap=True
            )

    def _build_settings_tab(self):
        with gr.Tab("Settings"):
            gr.Markdown("### 1. Google Cloud Platform Authentication")
            self.google_api_key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                value=self.settings.get("google_api_key", ""),
            )
            self.google_project_id_input = gr.Textbox(
                label="Google Cloud Project ID (Vertex AI Postpay Routing)",
                value=self.settings.get("google_project_id", ""),
            )
            self.gcs_bucket_input = gr.Textbox(
                label="Google Cloud Storage Bucket Name",
                value=self.settings.get("gcs_bucket", ""),
            )
            self.use_gcs_for_refs_input = gr.Checkbox(
                label="Upload Real-Time Reference Images to Google Cloud Storage "
                "(Bypasses payload limits)",
                value=self.settings.get("use_gcs_for_refs", False),
            )
            self.use_flex_paygo_input = gr.Checkbox(
                label="Enable Gemini Flex PayGo (50% Cost Reduction for Vertex AI On-Demand)",
                value=self.settings.get("use_flex_paygo", False),
            )

            gr.Markdown("### 2. BytePlus Authentication")
            self.byteplus_api_key_input = gr.Textbox(
                label="BytePlus API Key (Seedream V3 Endpoint)",
                type="password",
                value=self.settings.get("byteplus_api_key", ""),
            )

            with gr.Row():
                self.btn_save_settings = gr.Button("💾 Save Settings", variant="primary")
                self.btn_test_conn = gr.Button("Test Connection")

            gr.Markdown("### Cache Management")
            self.btn_clear_cache = gr.Button("Clear SQLite Image Cache", variant="stop")

    def _build_stats_tab(self):
        with gr.Tab("Usage Statistics"):
            gr.Markdown(
                "Metrics are tied to your specific API Key / Project ID. "
                "Monthly counters reset on the 1st."
            )
            with gr.Row():
                self.stat_tot_img = gr.Number(label="Total Images Generated", interactive=False)
                self.stat_mon_img = gr.Number(
                    label="Images Generated This Month", interactive=False
                )
            with gr.Row():
                self.stat_tot_cost = gr.Textbox(label="Total Cost (USD)", interactive=False)
                self.stat_mon_cost = gr.Textbox(label="Monthly Cost (USD)", interactive=False)
            self.btn_refresh_stats = gr.Button("Refresh Statistics")

    def _wire_events(self):
        h = self.handlers

        # Real-Time Costs
        rt_cost_inputs = [
            self.engine_radio,
            self.model_dropdown,
            self.res_radio,
            self.batch_slider,
            self.use_flex_paygo_input,
        ]
        for comp in rt_cost_inputs:
            comp.change(
                fn=h["estimate_cost_display"], inputs=rt_cost_inputs, outputs=self.cost_indicator
            )

        # Batch Costs
        b_cost_inputs = [self.b_model_dropdown, self.b_res_radio, self.b_batch_slider]
        for comp in b_cost_inputs:
            comp.change(
                fn=h["estimate_batch_cost_display"],
                inputs=b_cost_inputs,
                outputs=self.b_cost_indicator,
            )

        # Settings
        self.btn_save_settings.click(
            fn=h["save_settings"],
            inputs=[
                self.google_api_key_input,
                self.google_project_id_input,
                self.gcs_bucket_input,
                self.use_gcs_for_refs_input,
                self.use_flex_paygo_input,
                self.byteplus_api_key_input,
            ],
        )

        # Connection Test
        self.btn_test_conn.click(
            fn=h["check_connection_all"],
            inputs=[
                self.google_api_key_input,
                self.google_project_id_input,
                self.byteplus_api_key_input,
            ],
            outputs=[self.gemini_status, self.byteplus_status],
        )

        # Clear Prompt
        self.btn_clear_prompt.click(fn=h["clear_ui_prompt"], outputs=self.prompt_box)
        self.b_btn_clear_prompt.click(fn=h["clear_ui_prompt"], outputs=self.b_prompt_box)

        # Clear Cache
        self.btn_clear_cache.click(
            fn=h["handle_clear_cache"],
            outputs=[
                self.gemini_status,
                self.byteplus_status,
                self.history_gallery,
                self.history_table,
            ],
        )

        # Usage Stats
        self.btn_refresh_stats.click(
            fn=h["load_initial_stats"],
            inputs=[self.google_api_key_input, self.google_project_id_input],
            outputs=[self.stat_tot_img, self.stat_mon_img, self.stat_tot_cost, self.stat_mon_cost],
        )

        # History Refresh
        self.btn_refresh_history.click(
            fn=h["load_history"], outputs=[self.history_gallery, self.history_table]
        )

        # Engine Selection Change (Updates Model List)
        self.engine_radio.change(
            fn=h["update_model_list"], inputs=self.engine_radio, outputs=self.model_dropdown
        )

        # Real-Time Generation
        self.btn_send.click(
            fn=h["process_generation"],
            inputs=[
                self.engine_radio,
                self.google_api_key_input,
                self.google_project_id_input,
                self.byteplus_api_key_input,
                self.use_flex_paygo_input,
                self.prompt_box,
                self.model_dropdown,
                self.res_radio,
                self.ar_dropdown,
                self.batch_slider,
                self.gcs_bucket_input,
                self.use_gcs_for_refs_input,
                self.input_gallery,
            ],
            outputs=[
                self.output_gallery,
                self.stat_tot_img,
                self.stat_mon_img,
                self.stat_tot_cost,
                self.stat_mon_cost,
            ],
        ).then(fn=h["load_history"], outputs=[self.history_gallery, self.history_table])

        # Batch Generation
        # Batch Job Execution & Dashboard Wiring
        self.b_btn_send.click(
            fn=h["submit_batch_task"],
            inputs=[
                self.google_api_key_input,
                self.google_project_id_input,
                self.b_prompt_box,
                self.b_model_dropdown,
                self.b_res_radio,
                self.b_ar_dropdown,
                self.b_batch_slider,
                self.gcs_bucket_input,
                self.b_input_gallery,
            ],
            outputs=[self.job_table, self.b_status_msg],
        )

        self.b_refresh_btn.click(
            fn=h["refresh_job_statuses"],
            inputs=[self.google_api_key_input, self.google_project_id_input],
            outputs=[self.job_table],
        )

        # Batch File Fetching & Cleanup
        self.fetch_btn.click(
            fn=h["fetch_completed_job"],
            inputs=[
                self.google_api_key_input,
                self.google_project_id_input,
                self.fetch_job_id,
                self.gcs_bucket_input,
            ],
            outputs=[
                self.batch_gallery,
                self.fetch_msg,
                self.stat_tot_img,
                self.stat_mon_img,
                self.stat_tot_cost,
                self.stat_mon_cost,
                self.job_table,  # Send the updated Dashboard UI table
            ],
        ).then(fn=h["load_history"], outputs=[self.history_gallery, self.history_table])

        # Discard Batch Job without downloading
        self.discard_btn.click(
            fn=h["discard_batch_job"],
            inputs=[
                self.google_api_key_input,
                self.google_project_id_input,
                self.fetch_job_id,
                self.gcs_bucket_input,
            ],
            outputs=[self.fetch_msg, self.job_table],
        )

        # Lifecycle Loading
        # Reload settings into UI inputs, history, and usage statistics
        # on app startup and page refresh
        self.ui.load(
            fn=h["load_settings_ui"],
            outputs=[
                self.google_api_key_input,
                self.google_project_id_input,
                self.gcs_bucket_input,
                self.use_gcs_for_refs_input,
                self.byteplus_api_key_input,
            ],
        ).then(fn=h["load_history"], outputs=[self.history_gallery, self.history_table]).then(
            fn=h["load_initial_stats"],
            inputs=[self.google_api_key_input, self.google_project_id_input],
            outputs=[self.stat_tot_img, self.stat_mon_img, self.stat_tot_cost, self.stat_mon_cost],
        )
