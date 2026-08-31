# Figure provenance and post-processing

## Canonical figure sets

- `code_generated/` contains the immutable analytical output generated directly from repository code.
- `submission_artwork/` contains the files intended for manuscript submission. At repository creation, these are byte-for-byte copies of the code-generated PDFs.
- `edit_log.csv` records every later post-processing operation.

## Permitted post-processing

Editorial post-processing may adjust panel spacing, typography, line weight, colour consistency, legend placement, clipping margins and export settings. It must not change:

- numerical values or point positions;
- axis limits, transforms or tick meaning;
- uncertainty intervals or sample sizes;
- category membership or analytical thresholds;
- geographic boundaries or station locations;
- the presence, absence or apparent prominence of observations.

## AI-assisted editing

Generative image editing should not be used on data-bearing figures. If an AI-assisted but non-generative layout or accessibility tool is used, retain the code-generated original, record the tool and version in `edit_log.csv`, describe the exact operation, and verify the final artwork against the underlying source data. Any use must also be checked against the target journal's policy at the time of submission.

## Final verification

Before replacing a manuscript figure with submission artwork:

1. compare every panel with its code-generated source;
2. confirm that all labels, values, axes and legends are unchanged;
3. inspect the PDF at final print size;
4. record the edit and reviewer in `edit_log.csv`;
5. retain both versions in the tagged release.
