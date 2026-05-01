import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# GREY UI THEME
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #2f2f2f;
    color: white;
}
h1, h2, h3, p, span {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = [
    'basophil','eosinophil','erythroblast','ig',
    'lymphocyte','monocyte','neutrophil','platelet'
]

# =========================
# MEDICAL REPORT
# =========================
CELL_REPORTS = {
    "basophil": {"description":"Allergic response cell","function":"Releases histamine","note":"High in allergy"},
    "eosinophil": {"description":"Parasite defense cell","function":"Kills parasites","note":"High in infection"},
    "erythroblast": {"description":"Immature RBC","function":"Forms RBC","note":"Bone marrow activity"},
    "ig": {"description":"Immature granulocyte","function":"Early immune response","note":"Infection marker"},
    "lymphocyte": {"description":"Adaptive immunity cell","function":"Antibodies","note":"Viral infection indicator"},
    "monocyte": {"description":"Macrophage precursor","function":"Cleans pathogens","note":"Chronic infection"},
    "neutrophil": {"description":"Bacterial defense","function":"Kills bacteria","note":"Bacterial infection"},
    "platelet": {"description":"Clotting cell","function":"Blood clotting","note":"Bleeding disorder if low"}
}

st.title("🩸 Blood Cell AI Diagnosis System")

# =========================
# MODEL LOADING (CONVNEXT)
# =========================
@st.cache_resource
def load_model():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    num_classes = len(CLASS_NAMES)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    path = r"C:\Users\Asus\Documents\Bloodclassfication\convnexttiny_best.pth"

    if not os.path.exists(path):
        st.error("Model not found!")
        return model

    checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

model = load_model()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# UPLOAD IMAGE
# =========================
file = st.file_uploader("Upload Blood Cell Image", type=["jpg","png","jpeg"])

# =========================
# PDF REPORT FUNCTION
# =========================
def generate_pdf(label, report, confidence):
    filename = "blood_report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Blood Cell Diagnosis Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Prediction: {label}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Description: {report['description']}", styles["Normal"]))
    content.append(Paragraph(f"Function: {report['function']}", styles["Normal"]))
    content.append(Paragraph(f"Note: {report['note']}", styles["Normal"]))

    doc.build(content)
    return filename

# =========================
# PREDICTION
# =========================
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

        label = CLASS_NAMES[pred.item()]
        confidence = conf.item() * 100

    # =========================
    # RESULT
    # =========================
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

    # =========================
    # REPORT
    # =========================
    st.subheader("🧾 Medical Report")
    report = CELL_REPORTS[label]

    st.write("**Description:**", report["description"])
    st.write("**Function:**", report["function"])
    st.write("**Note:**", report["note"])

    # =========================
    # PROBABILITY CHART
    # =========================
    st.subheader("📊 Class Probabilities")

    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, probs.numpy()[0])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # =========================
    # PDF DOWNLOAD
    # =========================
    if st.button("📄 Download Report PDF"):
        file_path = generate_pdf(label, report, confidence)
        with open(file_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="blood_report.pdf"
            )