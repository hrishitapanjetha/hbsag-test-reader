import glob
import io
import json
import os
import tempfile
import zipfile

import streamlit as st

from pipeline import HBsAgPipeline, draw_final
from report import DISCLAIMER, annotated_jpeg, build_docx, build_pdf, hw_text, results_frame, summary_counts

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "hbsag_model_v2.pt")
SAMPLE_DIR = os.path.join(HERE, "samples")
SAMPLES = sorted(f for f in glob.glob(os.path.join(SAMPLE_DIR, "*.*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
try:
    SAMPLE_NOTES = json.load(open(os.path.join(SAMPLE_DIR, "samples.json")))
except Exception:
    SAMPLE_NOTES = {}

st.set_page_config(page_title="HBsAg Test Reader", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Loading the detection model...")
def load_pipeline(weights):
    return HBsAgPipeline(weights, load_readers=False)


ON_CLOUD = os.path.exists("/mount/src") or os.environ.get("HOME") == "/home/adminuser"


def ensure_readers(p, handwriting):
    if p.reader is None or (handwriting and p.trocr is None):
        with st.spinner("Loading the text readers (first run only)..."):
            p._load_readers(easyocr_reader=True, handwriting=handwriting)


st.title("HBsAg Rapid Test Reader")
st.caption("Upload photographs of HBsAg rapid tests. The tool finds each device, reads the result and the specimen information, and prepares a report.")

with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Model weights", DEFAULT_WEIGHTS)
    conf = st.slider("Detection confidence", 0.10, 0.60, 0.25, 0.05, help="Lower finds more faint lines but may add false detections.")
    rule = st.selectbox("Photo with two devices", ["any_positive", "all_positive"], format_func=lambda x: "Positive if either device is positive" if x == "any_positive" else "Positive only if both devices are positive")
    read_ids = st.checkbox("Read barcode and test type", value=True, help="Reads the barcode and the printed test name. Slower, especially on the first run.")
    use_hw = st.checkbox("Also read handwritten ID", value=not ON_CLOUD, disabled=not read_ids, help="Uses a larger model and needs more memory. Handwriting reading is the least reliable part, so it is only a secondary check.")
    operator = st.text_input("Prepared by (optional)")
    st.divider()
    st.caption(DISCLAIMER)

if not os.path.exists(weights):
    st.error(f"Model weights not found at {weights}")
    st.stop()

tab_demo, tab_upload = st.tabs(["Try the sample photographs", "Upload your own photographs"])
run_items = None

with tab_demo:
    if SAMPLES:
        st.write("No photographs to hand? Pick some of the sample photographs below and press Analyse to see how the tool works.")
        names = [os.path.basename(f) for f in SAMPLES]
        default = [n for n in ("509341479.jpg", "302237023.jpg", "468319602.jpeg") if n in names] or names[:3]
        chosen = st.multiselect("Sample photographs", names, default=default, help="Choose one or more.")
        cols = st.columns(3)
        for k, n in enumerate(chosen):
            with cols[k % 3]:
                st.image(os.path.join(SAMPLE_DIR, n), caption=f"{n}: {SAMPLE_NOTES.get(n, '')}", use_container_width=True)
        if st.button(f"Analyse {len(chosen)} sample photograph(s)", type="primary", disabled=not chosen, key="run_demo"):
            run_items = [(n, open(os.path.join(SAMPLE_DIR, n), "rb").read()) for n in chosen]
    else:
        st.info("No sample photographs are available in this copy of the app.")

with tab_upload:
    uploads = st.file_uploader("Upload test photographs", type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"], accept_multiple_files=True)
    if len(uploads) > 40:
        st.warning("More than 40 photographs selected. Processing may be slow, so consider running them in smaller batches.")
    if st.button(f"Analyse {len(uploads)} photograph(s)" if uploads else "Analyse", type="primary", disabled=not uploads, key="run_upload"):
        run_items = [(u.name, u.getvalue()) for u in uploads]

items = run_items or []
run = bool(items)

if run:
    p = load_pipeline(weights)
    p.conf, p.photo_rule = conf, rule
    if read_ids:
        ensure_readers(p, use_hw)
    results = []
    bar = st.progress(0.0, text="Starting...")
    tmp = tempfile.mkdtemp()
    for i, (name, data) in enumerate(items):
        bar.progress(i / len(items), text=f"Reading {name} ({i + 1} of {len(items)})")
        path = os.path.join(tmp, f"{i:03d}_{os.path.basename(name)}")
        with open(path, "wb") as fh:
            fh.write(data)
        try:
            row, ex = p.process(path, read_ids=read_ids, use_handwriting=use_hw)
            rgb = draw_final(ex["image"], ex)
            results.append(dict(name=name, row=row, rgb=rgb, jpeg=annotated_jpeg(rgb)))
        except Exception as e:
            st.error(f"Could not read {name}: {e}")
    bar.progress(1.0, text="Done")
    st.session_state["results"] = results
    st.session_state["operator"] = operator

results = st.session_state.get("results")
if results:
    c = summary_counts(results)
    cols = st.columns(5)
    cols[0].metric("Photographs", c["photos"])
    cols[1].metric("Positive", c["positive"])
    cols[2].metric("Negative", c["negative"])
    cols[3].metric("Invalid", c["invalid"])
    cols[4].metric("Need review", c["review"])
    if c["review"]:
        st.warning(f"{c['review']} photograph(s) need a person to check them. They are marked below.")

    st.subheader("Results")
    st.dataframe(results_frame(results), use_container_width=True, hide_index=True)

    st.subheader("Report")
    op = st.session_state.get("operator", "")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download report (Word)", build_docx(results, operator=op), "HBsAg_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    d2.download_button("Download report (PDF)", build_pdf(results, operator=op), "HBsAg_report.pdf", "application/pdf")
    d3.download_button("Download table (CSV)", results_frame(results).to_csv(index=False).encode(), "HBsAg_results.csv", "text/csv")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        for r in results:
            z.writestr(f"annotated_{os.path.splitext(r['name'])[0]}.jpg", r["jpeg"])
    d4.download_button("Download annotated photos (ZIP)", zbuf.getvalue(), "HBsAg_annotated.zip", "application/zip")

    st.subheader("Photograph details")
    for r in results:
        x = r["row"]
        flag = " (needs review)" if x["needs_review"] else ""
        with st.expander(f"{r['name']}: {x['photo_result'].upper()}{flag}", expanded=len(results) <= 2):
            left, right = st.columns([2, 1])
            left.image(r["rgb"], use_container_width=True)
            right.markdown(f"**Result:** {x['photo_result'].upper()}")
            right.markdown(f"**Devices:** {x['n_devices']} ({', '.join(d.upper() for d in x['device_results']) or 'none found'})")
            right.markdown(f"**Specimen ID (barcode):** {x['specimen_id'] or 'not read'}")
            right.markdown(f"**Handwritten ID:** {hw_text(x)}")
            right.markdown(f"**Test type:** {x['test_type']}")
            for reason in x["reasons"]:
                right.warning(reason)
            if not x["reasons"]:
                right.success("No points to check.")
    st.caption(DISCLAIMER)
else:
    st.info("Upload one or more photographs and press Analyse.")
