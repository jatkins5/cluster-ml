# Relic pilot and matched radio/X-ray follow-ups

This directory contains the six-halo relic pilot's review material and the
scripts/provenance for the matched CNN comparisons. **All relic-to-event
associations remain unvalidated.** Five halos have candidate opposite-side
pairs at snapshot 99; this is not a measurement of a separation–age relation.

The PDFs and tables can be inspected locally after `git pull`; no simulation
data or Python environment is needed to open them.

## Open the diagnostic panels

| Halo | Diagnostic PDF | Candidate pair at snapshot 99 |
| --- | --- | --- |
| 2380799 | [Open panels](relic_pilot/halo_2380799.pdf) | Yes |
| 5873398 | [Open panels](relic_pilot/halo_5873398.pdf) | Yes |
| 12871237 | [Open panels](relic_pilot/halo_12871237.pdf) | Yes |
| 17129688 | [Open panels](relic_pilot/halo_17129688.pdf) | No |
| 17469179 | [Open panels](relic_pilot/halo_17469179.pdf) | Yes |
| 19262770 | [Open panels](relic_pilot/halo_19262770.pdf) | Yes |

Rows show snapshots 91 and 99; columns show xy, yz, and xz projections.
Cyan/green contours are projections of the selected 3-D voxel groups;
matching crosses mark luminosity-weighted centres after bright-cell removal.
The dashed white line is the proposed collision axis, and white plus signs
mark recorded main/subcluster positions where available. No contour means
the selection did not accept a component, not that no emission exists.

The background is 1.4 GHz luminosity per projected bin, using the regenerated
synchrotron prescription and the existing density cut. Colour limits are
automatic, and some panels use different voxel sizes. Use these panels for
morphology; brightness evolution needs common luminosity-per-area scales.
Coordinates are relative to each snapshot's halo centre and R500.

## Review procedure

1. Inspect all three projections for coherent extended fronts, faint bridges
   joining distinct structures, fragmentation, and overlooked counterparts.
   Include the halo without an accepted pair in this check.
2. Assess whether the two structures plausibly belong together. Unequal sizes
   and brightnesses are allowed. Collision-axis agreement is supporting
   evidence, not an automatic acceptance or rejection criterion.
3. Compare alternative events in the competing-event table, including mergers
   below the mass-ratio threshold used to select the six pilot halos.
4. Record morphology quality, segmentation issues, candidate event IDs,
   association confidence, and a reason before inspecting any fitted
   separation–age relation. Keep ambiguous cases flagged.

Snapshots 91 and 99 are about 1.34 Gyr apart. Snapshot 91 precedes the selected
collision in every pilot halo, so these panels alone cannot establish which
collision generated a front. Closer snapshots, subcluster trajectories, and
gas/shock context are needed for stronger event associations. The temporal
review should check for already-existing relics, following the logic in
[Lee et al., section IV](https://arxiv.org/html/2510.21632v1#S4).

## Tables and reproducibility

- [Candidate sample](relic_pilot/candidate_sample.csv): eligible halos and the
  six selected using merger histories alone.
- [Structure/event candidates](relic_pilot/structure_event_candidates.csv):
  per-halo, per-snapshot detections, proposed event ages, separations, and angles.
- [Structures](relic_pilot/structures.csv): individual component properties.
- [Competing events](relic_pilot/competing_events.csv): other catalogue events
  in the selected halos, including lower-mass mergers.
- `relic_pilot/members_*_snap*.npz`: cleaned component memberships, positions
  relative to the halo centre in physical kpc, and cell luminosities in W/Hz.
  Cell indices refer to the generated, density-selected shock-cell arrays,
  not original snapshot particle IDs.
- [Pilot manifest](relic_pilot/manifest.json): input provenance and limitations.
- [Run manifest](run_manifest.json): job IDs and frozen source hashes.
- [Run notes](RUN_NOTES.txt): protocols, dependencies, and expected CNN outputs.
- `preflight/`: matched-dataset verification and expected outer/inner splits.
- `code/` and the submission scripts: frozen copies used for these jobs.

Job states in the manifest/notes are the recorded submission-check states,
not a live status display. Large simulation inputs remain on Oscar and are
required to rerun the experiments. The review bundle includes all six PDFs,
four review tables, and twelve small membership archives.
