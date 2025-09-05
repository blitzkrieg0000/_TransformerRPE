# Relative Positional Encoding (RPE) Implementations

This repo contains multiple **Relative Positional Encoding (RPE)** variants for Transformer models, implemented in Python.  
RPE improves Transformer performance on sequences by encoding **relative positions** between tokens rather than absolute positions.


## Overview
| File | Description | Reference Paper |
|------|-------------|----------------|
| `RPE00.py` | Baseline / simple RPE implementation | – |
| `RPE01_Shaw.py` | Shaw et al., 2018 RPE for self-attention | Shaw, Uszkoreit & Vaswani (2018) [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) |
| `RPE02_AnnaHuang2018.py` | Huang et al., 2018 relative attention variant | Anna Huang et al., 2018 [Improved Transformer with RPE](https://arxiv.org/abs/1805.08318) |
| `RPE03_XL.py` | Transformer-XL style RPE for long context | Dai et al., 2019 [Transformer-XL](https://arxiv.org/abs/1901.02860) |
| `RPE04_T5.py` | T5 style relative bias implementation | Raffel et al., 2020 [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683) |
