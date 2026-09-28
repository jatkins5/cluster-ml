# Vendored reference code (do not edit)

Verbatim copies of Chuiyang's (ckong13) radio-model scripts, taken
2026-09-06. They are the provenance for `Radio_Data/radio_FOF*.npz` and are
kept here so our own variants can be diffed against the original without
touching her working directory, which is read-only to us.

| file | source |
|---|---|
| `ckong13_Radio_generation_tngcluster.py` | `/oscar/data/idellant/Chuiyang/Radio_Data/Radio_generation.py` |
| `ckong13_Radio_generation_tngcluster.ipynb` | same directory, notebook form |
| `ckong13_Radio_generation_camels.py` | `/oscar/data/idellant/Chuiyang/Camels/Radio/Radio_generation.py` |

These are checked in unmodified and marked read-only. **Make changes in
`build_radio_cells.py`**, which reimplements the same weight formula (agreeing
with the stored weights to 5.2e-05, i.e. float32 storage precision) and adds
the per-cell volume and the gas-phase cuts.

Key line, `ckong13_Radio_generation_tngcluster.py`:

```python
shock = (Machnumber > 1.3) & (EnergyDissipation > 0)
w = 5.2e+23 * E * (B_uG**(1.0 + 0.5*s)) / (B_uG**2 + Bcmb**2) * phi
```

There is no temperature or density cut, which is why the weight is carried
by cold, 379x overdense, Mach-26 gas -- stripped ISM of infalling galaxies
rather than the ICM in which radio relics form.

## Redaction

The upstream TNG-Cluster radio script hard-codes its author's personal TNG
API key. In these copies it is replaced by `REDACTED-TNG-API-KEY` (the only
change from the originals), because this repository is public. The copies
are for reading the model, not running the download step; anyone running
it needs their own key from tng-project.org.
