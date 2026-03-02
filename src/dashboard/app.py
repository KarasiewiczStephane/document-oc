"""Streamlit dashboard for document OCR result visualization.

Displays an image upload area, OCR extracted text, extracted fields
with confidence scores, and bounding box visualization using demo data.

Run with: streamlit run src/dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw


def generate_demo_extracted_fields() -> pd.DataFrame:
    """Generate synthetic extracted fields with confidence scores.

    Returns:
        DataFrame with field_name, value, confidence, source, and is_valid.
    """
    return pd.DataFrame(
        [
            {
                "field_name": "Invoice Number",
                "value": "INV-2024-00847",
                "confidence": 0.97,
                "source": "rule_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Date",
                "value": "2024-11-15",
                "confidence": 0.95,
                "source": "rule_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Vendor Name",
                "value": "Acme Corporation",
                "confidence": 0.92,
                "source": "ml_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Total Amount",
                "value": "$4,250.00",
                "confidence": 0.98,
                "source": "rule_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Tax Amount",
                "value": "$382.50",
                "confidence": 0.91,
                "source": "rule_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Subtotal",
                "value": "$3,867.50",
                "confidence": 0.89,
                "source": "ml_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Due Date",
                "value": "2024-12-15",
                "confidence": 0.93,
                "source": "rule_extractor",
                "is_valid": True,
            },
            {
                "field_name": "PO Number",
                "value": "PO-2024-1234",
                "confidence": 0.85,
                "source": "ml_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Payment Terms",
                "value": "Net 30",
                "confidence": 0.78,
                "source": "ml_extractor",
                "is_valid": True,
            },
            {
                "field_name": "Email",
                "value": "billing@acme.com",
                "confidence": 0.72,
                "source": "ml_extractor",
                "is_valid": False,
            },
        ]
    )


DEMO_OCR_TEXT = """INVOICE

Acme Corporation
123 Business Ave, Suite 400
New York, NY 10001

Invoice Number: INV-2024-00847
Date: November 15, 2024
Due Date: December 15, 2024
PO Number: PO-2024-1234

Bill To:
Widget Industries
456 Commerce Blvd
Chicago, IL 60601

Description                  Qty    Unit Price    Amount
-----------------------------------------------------------
Software License (Annual)      1    $2,500.00    $2,500.00
Professional Services         10      $125.00    $1,250.00
Support Package                1      $117.50      $117.50

                              Subtotal:         $3,867.50
                              Tax (9%):           $382.50
                              TOTAL:            $4,250.00

Payment Terms: Net 30
Email: billing@acme.com
"""


def generate_demo_bounding_boxes() -> list[dict]:
    """Generate synthetic bounding box data for document fields.

    Returns:
        List of dicts with label, x, y, width, height, and confidence.
    """
    return [
        {
            "label": "Invoice Number",
            "x": 350,
            "y": 140,
            "width": 200,
            "height": 25,
            "confidence": 0.97,
        },
        {
            "label": "Date",
            "x": 350,
            "y": 170,
            "width": 180,
            "height": 25,
            "confidence": 0.95,
        },
        {
            "label": "Due Date",
            "x": 350,
            "y": 200,
            "width": 180,
            "height": 25,
            "confidence": 0.93,
        },
        {
            "label": "Vendor Name",
            "x": 50,
            "y": 60,
            "width": 180,
            "height": 25,
            "confidence": 0.92,
        },
        {
            "label": "Total Amount",
            "x": 400,
            "y": 450,
            "width": 150,
            "height": 25,
            "confidence": 0.98,
        },
        {
            "label": "Tax Amount",
            "x": 400,
            "y": 425,
            "width": 150,
            "height": 25,
            "confidence": 0.91,
        },
        {
            "label": "Subtotal",
            "x": 400,
            "y": 400,
            "width": 150,
            "height": 25,
            "confidence": 0.89,
        },
    ]


def create_demo_document_image() -> Image.Image:
    """Create a synthetic document image for demonstration.

    Returns:
        PIL Image of a simple invoice document.
    """
    width, height = 600, 550
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    lines = [
        (50, 20, "INVOICE"),
        (50, 60, "Acme Corporation"),
        (50, 80, "123 Business Ave, Suite 400"),
        (50, 100, "New York, NY 10001"),
        (350, 140, "Invoice: INV-2024-00847"),
        (350, 170, "Date: 2024-11-15"),
        (350, 200, "Due: 2024-12-15"),
        (50, 260, "Bill To: Widget Industries"),
        (50, 280, "456 Commerce Blvd, Chicago, IL 60601"),
        (50, 320, "Description             Qty   Price      Amount"),
        (50, 345, "Software License          1   $2,500    $2,500.00"),
        (50, 365, "Professional Services    10   $125      $1,250.00"),
        (50, 385, "Support Package           1   $117.50     $117.50"),
        (350, 400, "Subtotal:  $3,867.50"),
        (350, 425, "Tax (9%):    $382.50"),
        (350, 450, "TOTAL:     $4,250.00"),
        (50, 490, "Payment Terms: Net 30"),
        (50, 510, "Email: billing@acme.com"),
    ]

    for x, y, text in lines:
        draw.text((x, y), text, fill="black")

    draw.line([(50, 335), (550, 335)], fill="gray", width=1)
    draw.line([(50, 395), (550, 395)], fill="gray", width=1)

    return img


def draw_bounding_boxes(
    img: Image.Image,
    boxes: list[dict],
    show_labels: bool = True,
) -> Image.Image:
    """Draw bounding boxes on a document image.

    Args:
        img: Input PIL Image.
        boxes: List of bounding box dicts.
        show_labels: Whether to draw labels on boxes.

    Returns:
        PIL Image with boxes drawn.
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
    ]

    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        if show_labels:
            label = f"{box['label']} ({box['confidence']:.0%})"
            draw.text((x, y - 12), label, fill=color)

    return annotated


def render_upload_section() -> tuple[Image.Image | None, bool]:
    """Render the image upload section.

    Returns:
        Tuple of (uploaded image or None, whether to use demo data).
    """
    st.header("Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a document image (PNG, JPG, PDF)",
        type=["png", "jpg", "jpeg", "tiff"],
    )

    use_demo = st.checkbox("Use demo invoice data", value=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return image, False

    return None, use_demo


def render_ocr_output(text: str) -> None:
    """Render OCR extracted text section."""
    st.header("OCR Extracted Text")
    st.text_area(
        "Raw OCR Output",
        value=text,
        height=300,
        disabled=True,
    )

    word_count = len(text.split())
    line_count = len(text.strip().splitlines())
    col1, col2, col3 = st.columns(3)
    col1.metric("Words Extracted", word_count)
    col2.metric("Lines", line_count)
    col3.metric("Characters", len(text))


def render_extracted_fields(df: pd.DataFrame) -> None:
    """Render extracted fields table with confidence scores."""
    st.header("Extracted Fields")

    col1, col2, col3 = st.columns(3)
    avg_conf = df["confidence"].mean()
    valid_count = df["is_valid"].sum()
    col1.metric("Average Confidence", f"{avg_conf:.1%}")
    col2.metric("Fields Extracted", len(df))
    col3.metric("Valid Fields", f"{valid_count}/{len(df)}")

    def color_confidence(val: float) -> str:
        if val >= 0.9:
            return "background-color: #d4edda"
        if val >= 0.8:
            return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    styled = df.style.applymap(color_confidence, subset=["confidence"]).format(
        {"confidence": "{:.1%}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.sort_values("confidence"),
        x="confidence",
        y="field_name",
        orientation="h",
        color="source",
        title="Field Confidence Scores",
        labels={"confidence": "Confidence", "field_name": "Field"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_bounding_boxes(
    img: Image.Image,
    boxes: list[dict],
) -> None:
    """Render bounding box visualization section."""
    st.header("Bounding Box Visualization")

    show_labels = st.checkbox("Show field labels", value=True)
    annotated = draw_bounding_boxes(img, boxes, show_labels)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Document")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("Detected Fields")
        st.image(annotated, use_container_width=True)


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, str, list[dict], np.ndarray]:
    """Load all demo data for the dashboard.

    Returns:
        Tuple of (fields_df, ocr_text, bounding_boxes, document_image_array).
    """
    fields = generate_demo_extracted_fields()
    boxes = generate_demo_bounding_boxes()
    img = create_demo_document_image()
    return fields, DEMO_OCR_TEXT, boxes, np.array(img)


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Document OCR Dashboard",
        page_icon="📄",
        layout="wide",
    )

    st.title("Document OCR Dashboard")
    st.markdown(
        "Upload a document image to extract text and structured fields, "
        "or explore the demo invoice data below."
    )

    uploaded_image, use_demo = render_upload_section()

    if use_demo:
        fields_df, ocr_text, boxes, img_array = load_demo_data()
        demo_img = Image.fromarray(img_array)

        st.divider()
        render_ocr_output(ocr_text)
        st.divider()
        render_extracted_fields(fields_df)
        st.divider()
        render_bounding_boxes(demo_img, boxes)
    elif uploaded_image is not None:
        st.divider()
        st.image(uploaded_image, caption="Uploaded Document", use_container_width=True)
        st.info(
            "In production, the OCR pipeline would process this image. "
            "Enable 'Use demo invoice data' to see example results."
        )
    else:
        st.info("Upload a document image or enable demo mode to get started.")


if __name__ == "__main__":
    main()
