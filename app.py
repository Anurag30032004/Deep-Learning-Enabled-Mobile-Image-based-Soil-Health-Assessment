import gradio as gr
import os
import shutil
import datetime

from model.predictor import load_trained_model, predict_images
from model.config import MODEL_PATH, UPLOAD_DIR
from model.feedback import save_feedback

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = load_trained_model(MODEL_PATH)


def save_uploaded_files(files):
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if files:
        for file in files:
            filename = os.path.basename(file)
            shutil.copy(file, os.path.join(UPLOAD_DIR, filename))

    return files


def run_prediction(files, enable_calibration=True):
    if not files:
        return [], [], []
    save_uploaded_files(files)
    results, calibration_reports = predict_images(UPLOAD_DIR, model, enable_calibration=enable_calibration)
    formatted = []
    explanations = []
    for row in results:
        if len(row) == 8:
            name, soil, conf, moisture, salinity, om_index, ph_tendency, health = row
            kb_explanation = ""
        else:
            name, soil, conf, moisture, salinity, om_index, ph_tendency, health, kb_explanation = row
        formatted.append([
            name,
            soil,
            f"{conf:.2%}",
            f"{moisture:.2f}",
            f"{salinity:.2f}",
            f"{om_index:.2f}",
            ph_tendency,
            health
        ])
        if kb_explanation and name == "Final Decision":
            explanations.append(kb_explanation)
    return formatted, explanations, calibration_reports


# ── Calibration Report Formatter ────────────────────────────────────────

CONDITION_LABELS = {
    'overexposed': '☀️ Overexposed',
    'underexposed': '🌑 Underexposed',
    'warm_cast': '🔶 Warm Color Cast',
    'cool_cast': '🔷 Cool Color Cast',
    'low_contrast': '🌫️ Low Contrast',
    'normal': '✅ Normal',
    'error': '❌ Error',
}

def format_calibration_report(calibration_reports):
    """Build a Markdown calibration report from the reports list."""
    if not calibration_reports:
        return ""

    lines = ["## 🔬 Calibration Report\n"]
    lines.append("| Image | Condition | Adjustments | Brightness Δ |")
    lines.append("|-------|-----------|-------------|-------------|")
    for rpt in calibration_reports:
        meta = rpt.get('metadata')
        if meta is None:
            lines.append(f"| {rpt['image']} | ⏭️ Skipped | Calibration disabled | — |")
            continue
        cond = CONDITION_LABELS.get(meta['condition'], meta['condition'])
        adjustments = "; ".join(meta.get('adjustments', []))
        delta = meta.get('brightness_delta', 0)
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        lines.append(f"| {rpt['image']} | {cond} | {adjustments} | {delta_str} |")
    lines.append("")
    return "\n".join(lines)


def build_before_after_gallery(calibration_reports):
    """Build a list of (image_path, label) tuples for the before/after gallery."""
    gallery_items = []
    for rpt in calibration_reports:
        original = rpt.get('original_path', '')
        calibrated = rpt.get('calibrated_path', '')
        name = rpt['image']
        if original and os.path.isfile(original):
            gallery_items.append((original, f"Original — {name}"))
        if calibrated and os.path.isfile(calibrated) and calibrated != original:
            gallery_items.append((calibrated, f"Calibrated — {name}"))
    return gallery_items


# ── Custom CSS ──────────────────────────────────────────────────────────

CUSTOM_CSS = """
#preview {
    overflow-x: auto;
    white-space: nowrap;
    background: #f5f7fa;
    border-radius: 10px;
    padding: 10px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
#preview img {
    display: inline-block;
    margin: 0 6px;
    border-radius: 8px;
    border: 2px solid #e0e7ef;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    transition: transform 0.2s;
}
#preview img:hover {
    transform: scale(1.04);
    border-color: #4f8cff;
}
.gr-button-primary {
    background: linear-gradient(90deg, #4f8cff 0%, #38b6ff 100%);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(79,140,255,0.08);
}
.gr-button-secondary {
    background: #e0e7ef;
    color: #222;
    border: none;
    border-radius: 6px;
    font-weight: 500;
}
.gr-dataframe {
    background: #fafdff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.gr-markdown {
    color: #2a2e35;
    font-size: 1.2em;
    font-weight: 600;
}
.better-table th, .better-table td {
    padding: 8px 16px;
    font-size: 1.05em;
    border-bottom: 1px solid #e0e7ef;
}
.better-table {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    background: #181c24;
}
.calibration-toggle {
    border: 2px solid #4f8cff;
    border-radius: 8px;
    padding: 6px 12px;
    background: rgba(79,140,255,0.06);
}
"""


# ── Build UI ────────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("""
    # Soil Classification Explorer
    Effortlessly classify soil types from your images using our advanced AI model. Upload your images, preview them, and get instant, accurate predictions.
    """)

    with gr.Row():
        # LEFT PANEL
        with gr.Column(scale=1):
            image_input = gr.File(
                label="Upload Soil Images",
                file_types=["image"],
                file_count="multiple",
                interactive=True,
                show_label=True,
                elem_id="image_input"
            )
            gallery = gr.Gallery(
                label="Image Preview",
                elem_id="preview",
                columns=6,
                height=220,
                show_label=True
            )
            calibration_toggle = gr.Checkbox(
                label="🔬 Enable Auto-Calibration (normalizes lighting & color)",
                value=True,
                elem_classes=["calibration-toggle"],
                elem_id="calibration_toggle"
            )
            clear_btn = gr.Button("Reset", elem_classes=["gr-button-secondary"])

        # RIGHT PANEL
        with gr.Column(scale=1):
            discover_btn = gr.Button("Determine Soil Health", elem_classes=["gr-button-primary"])
            results_table = gr.Dataframe(
                headers=["Property", "Value"],
                datatype=["str", "str"],
                label="Classification Results",
                interactive=False,
                wrap=True,
                elem_classes=["better-table"],
                show_label=False,
                elem_id="results_table"
            )
            status = gr.Markdown("", visible=False)
            feedback_comment = gr.Textbox(label="User Comment (optional)", lines=2)
            feedback_correction = gr.Textbox(label="Correction (if any)", lines=1)
            explanation_md = gr.Markdown("", visible=True)

    # Preview updates instantly
    def update_gallery(files):
        return files

    image_input.change(
        fn=update_gallery,
        inputs=image_input,
        outputs=gallery
    )

    RESULT_HEADERS = ["Image Name", "Soil Type", "Confidence", "Moisture", "Salinity", "OM Index", "pH Tendency", "Soil Health"]

    def predict_and_feedback(files, user_comment, user_correction, enable_cal):
        if not files:
            return (
                gr.update(visible=True, value="**Please upload at least one image to classify.**"),
                [], [], gr.update(visible=False), "", "", []
            )
        results, explanations, calibration_reports = run_prediction(files, enable_calibration=enable_cal)
        if not results:
            return (
                gr.update(visible=True, value="**No predictions could be made.**"),
                [], [], gr.update(visible=False), "", "", []
            )
        final_row = None
        individual_rows = []
        for row in results:
            if str(row[0]).strip().lower() == 'final decision':
                final_row = row
            else:
                individual_rows.append(row)

        # Build calibration report markdown
        cal_report_md = format_calibration_report(calibration_reports)

        # Build before/after gallery
        ba_gallery = build_before_after_gallery(calibration_reports)

        if final_row:
            explanation_md_val = "\n".join([f"- {ex}" for ex in explanations]) if explanations else ""
            save_feedback(*final_row, explanation_md_val, user_comment, user_correction)
            property_value_rows = [[header, str(val)] for header, val in zip(RESULT_HEADERS, final_row)]
            return (
                gr.update(visible=False),
                property_value_rows,
                gr.update(visible=True, value=individual_rows),
                gr.update(visible=True),
                explanation_md_val,
                cal_report_md,
                ba_gallery
            )
        else:
            return (
                gr.update(visible=False),
                [],
                gr.update(visible=True, value=individual_rows),
                gr.update(visible=True),
                "",
                cal_report_md,
                ba_gallery
            )

    with gr.Accordion("Show Individual Image Results", open=False) as indiv_section:
        indiv_table = gr.Dataframe(
            headers=[
                "Image Name", "Soil Type", "Confidence",
                "Moisture", "Salinity", "OM Index", "pH Tendency", "Soil Health"
            ],
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            label="Individual Results",
            interactive=False,
            wrap=True,
            visible=True,
            elem_classes=["better-table"]
        )

    with gr.Accordion("🔬 Calibration Report", open=False, visible=True) as cal_section:
        calibration_report_md = gr.Markdown("", elem_id="calibration_report")
        ba_gallery = gr.Gallery(
            label="Before / After Comparison",
            columns=4,
            height=260,
            show_label=True,
            elem_id="ba_gallery"
        )

    discover_btn.click(
        fn=predict_and_feedback,
        inputs=[image_input, feedback_comment, feedback_correction, calibration_toggle],
        outputs=[status, results_table, indiv_table, indiv_section, explanation_md, calibration_report_md, ba_gallery]
    )

    def reset_all():
        if os.path.exists(UPLOAD_DIR):
            files = os.listdir(UPLOAD_DIR)
            if files:
                history_dir = os.path.join(os.path.dirname(UPLOAD_DIR), "history")
                os.makedirs(history_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = os.path.join(history_dir, f"session_{timestamp}")
                os.makedirs(session_dir, exist_ok=True)
                for f in files:
                    shutil.move(os.path.join(UPLOAD_DIR, f), os.path.join(session_dir, f))
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        return None, [], gr.update(visible=False), [], gr.update(visible=False), "", "", []

    clear_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[image_input, gallery, status, results_table, indiv_table, explanation_md, calibration_report_md, ba_gallery]
    )

demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)
