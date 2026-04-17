"""End-to-end pipeline client: image -> OCR -> compress -> decompress -> assert.

Owner: Sampreeth

CLI usage (shape to confirm during impl):
    python -m orchestrator.run_pipeline \
        --image path/to/FontLre_Noisec_TE.png \
        --ocr-url http://localhost:8001 \
        --huffman-url http://localhost:8002

Steps:
1. POST the image to the OCR service -> extracted_text
2. POST extracted_text to /compress  -> payload_b64, metrics
3. POST payload_b64 to /decompress   -> recovered_text
4. Assert recovered_text == extracted_text (losslessness)
5. Print summary: text preview, compression ratio, total latency

Used by both the demo video and the benchmark harness.

Not yet implemented.
"""
