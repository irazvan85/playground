"""Shared constants describing the modeled circuit's non-columnar (wide-field) units.

These are the lobula plate tangential cells (HS/VS), the descending-neuron
pool, and the motor-pool proxies -- real, individually identified cell
classes in the fly brain (Schnell et al. 2010; Krapp & Hengstenberg 1996;
Suver et al. 2016 -- see data/connectome/SOURCES.md) that are *not* organized
retinotopically column-by-column the way the early visual pathway is, so
they are modeled as one instance per named cell rather than one per hex
column.
"""

LPTC_TYPES = ["HSN", "HSE", "HSS", "VS1", "VS4", "VS6"]

DN_TYPES = [
    "DN_yaw_L",
    "DN_yaw_R",
    "DN_roll_L",
    "DN_roll_R",
    "DN_pitch_up",
    "DN_pitch_down",
    "DN_thrust_mod",
]

SINGLETON_TYPES = LPTC_TYPES + DN_TYPES

# Readout uses the descending-neuron pool's firing rates, in this fixed order.
READOUT_INPUT_TYPES = DN_TYPES
