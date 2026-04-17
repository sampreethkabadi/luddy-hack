"""CNN OCR model definitions.

Owner: Anuj

Three variants to benchmark (see PLAN.md §3 Experiment Matrix):
- PureCNN_CTC:     conv stack -> sequence projection -> CTC head
- CRNN_CTC:        conv stack -> BiLSTM -> CTC head
- UNetDenoiser_CRNN: U-Net denoiser + CRNN head, dual-loss (MSE + CTC)

Alphabet: lowercase + uppercase + digits + punctuation observed in labels.json
+ space + CTC blank token. Emit the final character set from labels.py output.

Not yet implemented.
"""
