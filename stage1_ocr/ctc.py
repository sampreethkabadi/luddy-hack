"""CTC decoding utilities.

Owner: Anuj

- greedy_decode: per-timestep argmax -> collapse repeats -> drop blanks
- beam_decode (optional stretch): prefix beam search for better accuracy

Also: helpers for mapping char <-> index given the alphabet.

Not yet implemented.
"""
