# SAC-DBS

Soft Actor-Critic controller for closed-loop Deep Brain Stimulation (DBS) using a Kuramoto-based tremor digital twin.

## What changed
- Uses the UCI HandPD wearable inertial dataset to calibrate tremor frequency and noise; falls back to synthetic 4-6 Hz tremor if offline.
- Saves training curves to `training_curve.png` (Monitor logs under `logs/sac`).
- Evaluation saves `results_plot.png` (tremor severity, stimulation voltage, medication level overlay).

## Setup
```bash
pip install gymnasium stable-baselines3 matplotlib numpy scipy pandas
```

## Train
```bash
python train_agent.py
```
Outputs:
- `sac_dbs_model.zip` trained policy.
- `training_curve.png` reward vs. timesteps plot.
- Monitor/TensorBoard logs in `logs/sac`.

## Evaluate and plot
```bash
python evaluate_and_plot.py
```
Outputs:
- `results_plot.png` dual-axis plot (order parameter R in red, voltage in blue, medication level dashed gray).

## Notes on the real dataset
- Source: UCI HandPD smartphone inertial data (`HandPD_sensors_data.zip`).
- Loader extracts the first numeric timeseries, computes vector magnitude, infers sampling rate if a time column exists, then estimates dominant tremor frequency (Welch) and noise variance.
- If the download or parse fails, the loader reverts to a synthetic 60 s band-passed noise signal in the 4-6 Hz Parkinsonian tremor band.
