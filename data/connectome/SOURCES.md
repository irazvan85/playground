# Data provenance

This project's neural network is built from **real Drosophila melanogaster
connectome data** where such data is available offline, plus **literature-cited
connection strengths** for the synaptic weights. Below is a full accounting of
what is real data, what is literature-derived, and why.

## What network access this sandbox actually has

Before building this, we tested direct access to the two databases that hold
live, queryable per-synapse connectivity for the fly brain:

- `neuprint.janelia.org` (hemibrain connectome) → **HTTP 403**, blocked by the
  sandbox's egress policy ("policy denial", confirmed via the proxy status
  endpoint, not a transient failure).
- `codex.flywire.ai` (FlyWire full-brain connectome) → **HTTP 403**, same.

So a live per-synapse-count pull was not possible in this environment.
`pypi.org` *is* reachable, so PyPI packages were used to source real,
already-published connectome data that ships as flat files.

## Real data used (vendored under `data/connectome/raw/`)

### `matsliah2024_fafb_vis_cols.csv`
Source: **Matsliah, A. et al. (2024). "Neuronal parts list and wiring diagram
for a visual system." Nature.** (FlyWire full-adult-fly-brain connectome,
optic lobe columnar visual neurons.)

Obtained via the `connectome_interpreter` PyPI package (MIT-licensed,
Yijie Yin, Cambridge; `connectome_interpreter/data/Matsliah2024/
fafb_right_vis_cols.csv`), which redistributes this table for connectomics
research use. Every row is a **real, individually identified FlyWire neuron**
(`root_id`), with its real cell type (Mi1, Tm1, T4a, T5b, L1, C2, R7, ...),
real hexagonal-lattice column assignment (`column_id`, `p`, `q`), and real
retinotopic position (`x`, `y`) in the right optic lobe. 22,684 real neurons
across 796 columns and 30 cell types are present in the file; this project
uses a contiguous spatial patch of those columns (see `graph.py`) to keep the
simulated network a tractable size, while still using the real per-column
neuron inventory and real identities for every unit modeled.

### `nern2024_medulla_hex.csv`
Source: **Nern, A. et al. (2024). "Connectome-driven neural inventory of a
complete visual system." (Janelia FlyEM hemibrain-based optic lobe
connectome.)**

Also obtained via `connectome_interpreter`
(`data/Nern2024/ME-columnar-cells-hex-location.csv`). Real per-hex-column
cell counts for medulla columnar types (L1-L5, Mi1/Mi4/Mi9, C2/C3, Tm1/Tm2/
Tm4/Tm9/Tm20, T1), used as a cross-check on typical per-column cell counts
from the independent hemibrain dataset.

## Literature-derived synaptic weights (`connectivity_table.csv`)

The two files above give real neuron **identity and spatial layout**, but not
a full per-synapse weight matrix (that requires the live per-synapse query
that is blocked here). The **connection strengths and signs** between cell
types in `connectivity_table.csv` are instead drawn from the primary
literature that established this circuit anatomically and functionally:

| Edge | Citation |
|---|---|
| Photoreceptor → lamina (L1/L2/L3), ON/OFF split timing | Behnia, R., Clark, D.A., Carter, A.G., Clandinin, T.R., Desplan, C. (2014). "Processing properties of ON and OFF pathways for Drosophila motion detection." *Nature* 512, 427-430. |
| Lamina/medulla (L1/L2/Mi1/Mi4/Mi9/Tm1/Tm2/Tm4/Tm9) → T4/T5 direction-selective wiring | Maisak, M.S. et al. (2013). "A directional tuning map of Drosophila elementary motion detectors." *Nature* 500, 212-216.; Shinomiya, K. et al. (2019). "Comparisons between the ON- and OFF-edge motion pathways in the Drosophila brain." *eLife* 8:e40025. |
| T4/T5 → lobula plate tangential cells (HS/VS), receptive-field integration | Schnell, B., Raghu, S.V., Nern, A., Borst, A. (2010). "Columnar cells necessary for motion responses of wide-field visual interneurons in Drosophila." *J. Comp. Physiol. A* 196, 411-483.; Krapp, H.G., Hengstenberg, R. (1996). "Estimation of self-motion by optic flow processing in single visual interneurons." *Nature* 384, 463-466. |
| General motion-vision circuit architecture / ON-OFF pathway split | Borst, A., Helmstaedter, M. (2015). "Common circuit design in fly and mammalian motion vision." *Nat. Neurosci.* 18, 1067-1076. |
| T4/T5 null-direction delayed inhibition (Barlow-Levick-type spatial offset mechanism that creates direction selectivity) | Haag, J., Arenz, A., Serbe, E., Borst, A. (2016). "Complementary mechanisms create direction selectivity in the fly." *eLife* 5:e17842. |
| HS/VS → descending neurons → wing steering, optomotor/gaze stabilization | Suver, M.P., Huda, A., Iwasaki, N., Safarik, S., Dickinson, M.H. (2016). "An array of descending visual interneurons encoding self-motion in Drosophila." *J. Neurosci.* 36(46), 11768-11780.; Fenk, L.M., Poehlmann, A., Straw, A.D. (2014). "Asymmetric processing of visual motion for simultaneous object and background responses." *Curr. Biol.* 24(24), 2913-2919. |

Weight *magnitudes* used in `connectivity_table.csv` are a **normalized,
relative** scale (not absolute synapse counts, since those require the
blocked live query) that preserves the qualitative excitatory/inhibitory
structure and relative pathway strength reported in these papers. Each row of
`connectivity_table.csv` carries a `citation_key` column referencing this
table.

**Note on signs**: the `sign` column encodes the net *functional* effect of a
pathway (as characterized by the cited papers' tuning/receptive-field
measurements), not a claim about a specific inhibitory neurotransmitter at a
monosynaptic connection. T4/T5 themselves are cholinergic (excitatory); the
"inhibitory" edges from Mi9/Tm9 in the null-direction-suppression rows model
the net delayed-antagonist effect described by Haag et al. 2016, which in the
real circuit partly involves interposed inhibitory interneurons and
conductance-based (shunting) rather than purely additive inhibition. Modeling
it as a signed additive LIF input is a standard simplification in
connectome-constrained functional modeling, not a claim about the underlying
molecular mechanism.

**Note on spatial offsets**: `neighbor_offset` rows carry a real geometric
mechanism -- each T4/T5 subtype's null-direction-suppressing input is wired
from a column *offset in hex space* from its own column, not from the same
column, which is what actually produces direction selectivity from otherwise
non-directional inputs (`graph.py` finds the nearest real neighboring column
by real x/y hex coordinates from the Matsliah2024 data). Without this spatial
offset, a same-column-only wiring scheme would make e.g. T4a and T4b
indistinguishable, since they would then receive identical inputs.

## Swapping in live data later

`src/fly_drone_pilot/connectome/loader.py` defines a `ConnectomeSource`
interface with two implementations:

- `BundledLiteratureSource` (default, used throughout this repo) — loads the
  files described above.
- `LiveNeuprintSource` (stub) — documents exactly what `neuprint-python` calls
  would replace the literature table with true per-synapse-count weights,
  for use outside this sandbox's network restrictions. No other code needs to
  change; `graph.py` only depends on the `ConnectomeSource` interface.

## Licensing note

`connectome_interpreter` is MIT-licensed. The underlying Nern2024/Matsliah2024
datasets are published Janelia/FlyWire-consortium connectome releases,
customarily distributed for open research reuse with citation (as done
above). This project vendors only the small derived tables needed (a few
hundred KB - 1.5MB total), not the raw EM imagery or segmentation.
