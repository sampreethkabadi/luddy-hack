import base64
import time
import requests
import streamlit as st
from PIL import Image
import io

OCR_URL  = "http://localhost:8001"
COMP_URL = "http://localhost:8002"

st.set_page_config(page_title="Neural Compression Pipeline", page_icon="🗜️", layout="wide")

st.title("2-Stage Neural Compression Pipeline")
st.caption("CNN OCR → Adaptive Huffman Encoding · Luddy Hack 2026")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    algo = st.radio("Huffman Algorithm", ["fgk", "vitter"], index=0,
                    help="FGK (Faller-Gallager-Knuth) or Vitter's Algorithm V")
    noise_type = st.selectbox("Noise Type Hint", [None, "f", "w", "c", "p"],
                              format_func=lambda x: {
                                  None: "Auto", "f": "Folded",
                                  "w": "Wrinkled", "c": "Coffee", "p": "Footprint"
                              }[x])
    st.divider()
    st.subheader("Service Status")
    try:
        r1 = requests.get(f"{OCR_URL}/health",  timeout=2).json()
        st.success(f"OCR ✓  backend: {r1.get('backend','?')}")
    except:
        st.error("OCR service offline")
    try:
        requests.get(f"{COMP_URL}/health", timeout=2)
        st.success("Compression ✓")
    except:
        st.error("Compression service offline")

# ── Main ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a noisy document image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        st.caption(f"{image.size[0]}×{image.size[1]} px")

    with col2:
        if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
            total_start = time.perf_counter()

            # Stage 1 — OCR
            with st.spinner("Stage 1: Running OCR..."):
                uploaded.seek(0)
                files = {"image": (uploaded.name, uploaded.read(), "image/png")}
                data  = {"noise_type": noise_type} if noise_type else {}
                try:
                    resp = requests.post(f"{OCR_URL}/ocr", files=files, data=data, timeout=60)
                    resp.raise_for_status()
                    ocr = resp.json()
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    st.stop()

            extracted_text = ocr["text"]
            ocr_latency    = ocr.get("latency_ms", 0)

            # Stage 2a — Compress
            with st.spinner("Stage 2: Compressing..."):
                try:
                    resp = requests.post(f"{COMP_URL}/compress",
                                         json={"text": extracted_text, "algo": algo}, timeout=10)
                    resp.raise_for_status()
                    comp = resp.json()
                except Exception as e:
                    st.error(f"Compression failed: {e}")
                    st.stop()

            # Stage 2b — Decompress
            with st.spinner("Verifying lossless round-trip..."):
                try:
                    resp = requests.post(f"{COMP_URL}/decompress",
                                         json={"payload_b64": comp["payload_b64"], "algo": algo}, timeout=10)
                    resp.raise_for_status()
                    decomp = resp.json()
                except Exception as e:
                    st.error(f"Decompression failed: {e}")
                    st.stop()

            total_ms   = (time.perf_counter() - total_start) * 1000
            lossless   = decomp["text"] == extracted_text

            # ── Results ──────────────────────────────────────────────────────
            st.success("Pipeline complete!" + (" ✓ Lossless" if lossless else " ✗ Mismatch!"))

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OCR Latency",      f"{ocr_latency:.0f} ms")
            m2.metric("Compression Ratio", f"{comp['ratio']:.3f}x")
            m3.metric("Entropy",           f"{comp['entropy']:.3f} bits/sym")
            m4.metric("End-to-End",        f"{total_ms:.0f} ms")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Efficiency",        f"{comp['efficiency']:.1%}")
            m6.metric("Compressed Bits",   f"{comp['bits']:,}")
            m7.metric("Characters",        f"{len(extracted_text):,}")
            m8.metric("Lossless",          "✓ Yes" if lossless else "✗ No")

            st.divider()

            tab1, tab2, tab3 = st.tabs(["📄 Extracted Text", "🗜️ Compressed", "♻️ Recovered"])

            with tab1:
                st.subheader(f"OCR Output  ·  backend: {ocr.get('backend','?')}")
                st.text_area("Extracted text", extracted_text, height=200)

            with tab2:
                st.subheader("Compressed Payload (base64)")
                payload = comp["payload_b64"]
                st.text_area("payload_b64", payload[:500] + ("..." if len(payload) > 500 else ""), height=100)
                st.download_button("Download compressed bytes",
                                   data=base64.b64decode(payload),
                                   file_name="compressed.bin",
                                   mime="application/octet-stream")

            with tab3:
                st.subheader("Decompressed Text")
                st.text_area("Recovered text", decomp["text"], height=200)
                if lossless:
                    st.success("Byte-for-byte identical to OCR output.")
                else:
                    st.error("Round-trip mismatch detected!")

else:
    st.info("Upload a document image above to run the pipeline.")

    # Demo with plain text when no image
    st.divider()
    st.subheader("Or test compression directly")
    sample = st.text_area("Enter text to compress", value="The quick brown fox jumps over the lazy dog.", height=100)
    if st.button("Compress Text", use_container_width=True):
        try:
            resp = requests.post(f"{COMP_URL}/compress", json={"text": sample, "algo": algo}, timeout=10)
            resp.raise_for_status()
            comp = resp.json()
            resp2 = requests.post(f"{COMP_URL}/decompress",
                                  json={"payload_b64": comp["payload_b64"], "algo": algo}, timeout=10)
            resp2.raise_for_status()
            decomp = resp2.json()

            c1, c2, c3 = st.columns(3)
            c1.metric("Compression Ratio", f"{comp['ratio']:.3f}x")
            c2.metric("Entropy",           f"{comp['entropy']:.3f} bits/sym")
            c3.metric("Efficiency",        f"{comp['efficiency']:.1%}")

            lossless = decomp["text"] == sample
            if lossless:
                st.success("✓ Lossless round-trip verified")
            else:
                st.error("Round-trip mismatch!")
        except Exception as e:
            st.error(f"Service error: {e}. Make sure both services are running.")
