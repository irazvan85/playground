# fly-drone-pilot

A Drosophila-connectome-constrained spiking neural network piloting a
simulated drone.

Real photoreceptor-to-descending-neuron optic-flow circuitry -- built from
genuine, individually identified Drosophila neurons and their real
hex-lattice retinotopic anatomy (Nern et al. 2024; Matsliah et al. 2024,
*Nature*) -- drives a leaky-integrate-and-fire network whose descending-
neuron output is decoded into roll/pitch/yaw/thrust commands for a
6-DOF drone flying through a procedural obstacle corridor.

```
real photoreceptor columns (compound eye)
        |  raw luminance samples, no precomputed "flow"
        v
L1 / L2 / L3  (lamina)
        v
Mi1/Mi4/Mi9 (ON)      Tm1/Tm4/Tm9 (OFF)
        v                     v
     T4a-d                 T5a-d          <- direction selectivity emerges here,
        \                   /                from real spatial offsets (graph.py)
         \                 /                 + a synaptic delay line (lif.py)
          v               v
       HS (HSN/HSE/HSS)   VS (VS1/VS4/VS6)   <- graded (non-spiking), as these
                \         /                     real cells actually are
                 v       v
          descending-neuron pool (DN_yaw_*, DN_roll_*, DN_pitch_*, DN_thrust_mod)
                    v
        trained readout -> [roll, pitch, yaw, thrust]
                    v
             6-DOF drone dynamics
```

## Why this is "real" and where it isn't

This sandbox's network policy blocks live access to both connectome
databases that could supply a true per-synapse weight matrix
(`neuprint.janelia.org` and `codex.flywire.ai` both return HTTP 403). So:

- **Real**: every modeled neuron's identity, cell type, and retinotopic
  hex-column position (`data/connectome/raw/*.csv`) comes from the actual
  published FlyWire/hemibrain optic-lobe connectome. The direction-
  selectivity wiring (spatial offsets + delay line) and the HS/VS
  graded-not-spiking behavior are also real, cited anatomical/physiological
  findings, not artistic license.
- **Literature-derived, not live-queried**: the synaptic weights/signs in
  `data/connectome/connectivity_table.csv` are a normalized scale reflecting
  the qualitative structure reported in the cited papers, not live
  per-synapse counts.
- **Not connectome-derived at all**: the final DN-pool -> command readout
  (trained, ~30 parameters) and the drone's rigid-body dynamics -- neither
  claims to be biological.

Full citations, licensing notes, and exactly how to swap in a live
`neuprint-python` pull once you have network access are in
[`data/connectome/SOURCES.md`](data/connectome/SOURCES.md).

## Running it

```bash
pip install -e ".[dev]"
pytest -q                      # unit + smoke tests
python scripts/demo.py         # trains the readout, flies the corridor, saves plots/ to output/
python scripts/demo.py --skip-training --n-steps 300   # quick untrained run
```

`scripts/demo.py` prints forward progress, collision status, and minimum
clearance, and saves four plots to `output/`: the flight trajectory
(top-down + side view), a spike raster of the real-cell-type units, the
descending-neuron pool's firing-rate traces, and a snapshot of the compound
eye's real hex-column luminance layout mid-flight.

## Layout

```
data/connectome/          vendored real anatomy data + literature connectivity table + SOURCES.md
src/fly_drone_pilot/
  connectome/             loader (real data + literature weights) and instance-graph builder
  model/                  LIF/graded network, readout layer, readout training
  vision/                 compound-eye sampling (real hex-column layout)
  drone/                  6-DOF dynamics + command mapping (no biology here)
  sim/                    procedural environment + the closed control loop
  viz/                    plots
scripts/                  demo.py, fetch_connectome_data.py (live-neuprint instructions)
tests/                    pytest suite
```
