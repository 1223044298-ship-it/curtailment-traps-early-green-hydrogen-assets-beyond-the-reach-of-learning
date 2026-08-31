# Retired official-map registration reference

These files document an earlier Figure 1 registration workflow. The current
Figure 1 uses OpenStreetMap coastline and administrative geometry throughout;
none of the files in this directory is read by the active plotting function.

## Preferred current boundary master

- Source: Standard Map Service System, Ministry of Natural Resources of China (`http://bzdt.ch.mnr.gov.cn/`).
- Product: China boundary map, 1:7,400,000, no neighbouring countries, no rivers, linework version 1.
- Standard-map source number printed on the product: `GS(2023)2767`.
- EPS export date: 8 August 2023.
- Author-held source files: `MNR_China_boundary_1_7.4m_GS2023_2767.eps` and its original ZIP archive. The third-party EPS and ZIP are not redistributed in this repository; their preview, integrity hashes and cleaned analytical derivatives are included.
- Geographic content: national and provincial boundaries, Taiwan, and a standard South China Sea inset.

This was the boundary master used in an earlier Figure 1a candidate. Its labels, province fills and decorative outer halo were removed; the national and provincial linework, Taiwan and the native South China Sea representation were retained for historical comparison only.

## Secondary 2023 full-extent reference

- Product: China map, 1:10,000,000, no neighbouring countries, uncoloured linework version 1.
- Standard-map source number printed on the product: `GS(2023)2763`.
- Author-held source files: `MNR_China_1_10m_GS2023_2763.eps` and its original ZIP archive. These third-party source files are not redistributed here; the embedded-preview PNG and provenance are retained.
- Use: independent cross-check of current national and provincial boundary geometry. Its full South China Sea extent is inefficient for the compact Figure 1 layout.

## User-downloaded compact layout reference

- Source: Standard Map Service System, Ministry of Natural Resources of China (`http://bzdt.ch.mnr.gov.cn/`).
- Downloaded: 19 August 2026.
- Product: China map with provincial colouring, 1:16,000,000, landscape, no neighbouring countries, linework version 1.
- Standard-map source number printed on the product: `GS(2016)2923`.
- Original archive identifier: `4o28b0625501ad13015501ad2bfc0135b`.
- Original production file: `MNR_China_1_16m_GS2016_2923.eps`, 1208 x 859 pt bounding box; Adobe InDesign CS6 export.
- Geographic content: national and provincial boundaries, Taiwan, and the standard South China Sea inset.

## Historical use in Figure 1

The retired candidate derived visible national and provincial boundaries, Taiwan and the South China Sea representation from `GS(2023)2767`. The active Figure 1 no longer uses this affine-registration workflow; its geometry and projection are documented in `../osm_china_boundaries/README.md`.

Reproducibility files are `GS2023_2767_clean.svg`, `GS2023_2767_paths.json`, `GS2023_2767_registration.json` and `GS2023_2767_registration_diagnostic.png`. After the full-boundary refinement, the clipped registration root-mean-square separation is approximately 9.8 source-page units, or 0.43% of the 2,269-unit map width. Removing labels and adding thematic content changes the published map, so `GS(2023)2767` documents source provenance and must not be presented as approval of the modified figure. Current status is recorded in `../Figure1_authorised_map_provenance.txt`.

The derived OSM mask and its source, licence, transformation and integrity hash are documented in `../osm_coastline/README.md`.

## Integrity hashes (SHA-256)

- Archive: `EC5DFB91444A3F97CF4FE57E39C68221CF867D7E74D873EA7B3AF000D8DE6AB0`
- EPS: `3799D853EF4CEC1836DAE941FFC913A57B748EA3E78E893AA9673AEA3CF6267C`
- Embedded-preview PNG: `6C0BE2A61ECDDA82F72D0319FD2E9314C79DA58D570310568B04DD62B2273A4A`

Preferred 2023 reference:

- `GS(2023)2767` archive: `1C8F7F8B0402D5D3A32A917E16F6347A4220A05D2F56F55D4E473783D3770F40`
- `GS(2023)2767` EPS: `8709AA9590ACAEF2926FAB9AD6979665C7CAF8469EC7186EA33EDEB9838368CC`
- `GS(2023)2767` embedded-preview PNG: `CE019F7363F9B5EFE0D59F841AC9DA7CD821282C0A77B241D9A339A60C723A48`

Secondary 2023 reference:

- Archive: `EB2452D803B4A78721513C6295E6A5AADE613741ECAE60F0E378AFB1AE9E305B`
- EPS: `9875E44F9AA76AEE270C341164BA7EC8A8ACE43F43E84E6A55A60F5C4B94A73A`
- Embedded-preview PNG: `8B52E6A30A2AE6ABC908E31E0B29E2987D1D234DA8D5BA7CD885CE2C6E458492`
