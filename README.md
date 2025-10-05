PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting

This repository provides the official implementation of PhaseFormer, a model for time series forecasting that works in the phase domain instead of traditional patch/segment-based methods. 
The model is built with PyTorch Lightning to facilitate training and evaluation.
This repo is compatible with the Time-Series Library framework.

PhaseFormer introduces:
- Phase tokenization: Splits a time series into fixed cycles (period_len) and encodes each cycle as a phase token.
- ross-phase interaction: Exchanges information between phases via a cross-phase routing mechanism.
- Phase blocks stacking: Layers multiple phase blocks to predict future phases and reconstruct them into time series.

This approach improves both efficiency and effectiveness compared with standard segment-based forecasting.

Example scripts are provided in the /scripts directory. You can run PhaseFormer on Traffic by:

```bash
bash ./scripts/run_traffic.py
```

