# OpenStreetMap coastline mask provenance

Figure 1a uses a land--ocean mask derived from the OpenStreetMap coastline
processing service. At render time the mask is intersected with OpenStreetMap
national administrative relations so that land outside the display extent is
not shown. OpenStreetMap admin-level-2 and admin-level-4 relations provide the
visible country and province linework. Project records and all map layers share
the same projection.

- Source: `https://osmdata.openstreetmap.de/data/land-polygons.html`
- Download: `simplified-land-polygons-complete-3857.zip`
- Source last modified: 25 August 2026
- Attribution: © OpenStreetMap contributors
- Licence: Open Database License 1.0 (`ODbL 1.0`)
- Derived file: `osm_land_east_asia_20260825.geojson`
- Transformation: crop the EPSG:3857 land polygons to 68--142 degrees east
  and 4--60 degrees north, inverse-project to WGS84, repair geometry and apply
  a topology-preserving 0.004-degree simplification.

The geometry is used as scientific cartographic context and not as legal
boundary adjudication. It is not used in any analytical classification.
