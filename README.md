# Flywire-Brain-Model
Uses flywire data to model a fruit fly brain

The separate [Eon comparison baseline](hybrid_fly/EON_METHODOLOGY.md) uses Shiu
neural dynamics and 15 ms brain/body coupling. Its documented scope is locomotion
with additional neural readouts; it is not a full reproduction of Eon's demo.

Run the Eon model on the NVIDIA GPU from PowerShell:

```powershell
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --backend gpu --seconds 1.5 --viewer
```

The neural backend already defaults to GPU; `--backend gpu` selects it explicitly.
The runner prints the selected backend and GPU. Body physics and CSV loading
still run on CPU.
