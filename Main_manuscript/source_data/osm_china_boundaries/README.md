# OpenStreetMap China map geometry provenance

Figure 1a uses one OpenStreetMap geometry stack for the land mask, coastline,
national outline and province boundaries. Project records and every map layer
are transformed with the same China Albers equal-area projection, so no
cross-source affine registration is applied.

## Sources

- Administrative data: OpenStreetMap relations requested through the Overpass
  API on 28 August 2026.
- OSM data timestamp: `2026-08-28T08:59:21Z`.
- China country relation: `270056`.
- Taiwan and Macao geometry relations: `449220` and `1867188`.
- Province-level relations: 32 OSM `admin_level=4` relations returned within
  the China area query.
- Coastline land source: the OSM coastline-derived land polygons documented in
  `../osm_coastline/README.md`.
- Attribution: copyright OpenStreetMap contributors.
- Licence: Open Database License 1.0 (`ODbL 1.0`).

The main Overpass query was:

```text
[out:json][timeout:300];
rel["boundary"="administrative"]["admin_level"="2"]["ISO3166-1"="CN"]->.country;
.country map_to_area->.countryarea;
(.country;rel["boundary"="administrative"]["admin_level"="4"](area.countryarea););
out geom;
```

Taiwan and Macao relations were requested separately by their `ISO3166-1`
tags. The processing script is
`analysis_code/workflows/20260818_figures/code/prepare_osm_china_boundaries.py`.

## Derived geometry

- File: `osm_china_admin_2_4_20260828.geojson`.
- SHA-256: `46CD7E8687452A24405C0A0E7F0635ECA124E73823A9CACB91ECF59B10D5F626`.
- Contents: coastline-clipped land, country outline and province linework.
- Simplification: topology-preserving tolerance of 0.003 degrees.
- Display extent: contiguous China, Hainan and Taiwan; remote South China Sea
  polygons below the main Hainan extent are not shown in this scientific
  locator map.

OSM administrative geometry is used only as cartographic context. It is not a
legal boundary adjudication and does not enter any record-level calculation.
