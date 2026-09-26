# Multispectral Imagery (Sentinel-2, Landsat, HLS; UAS multispectral)

Registry id `hls` · planned **v0.4.0** · status **candidate**.

Satellite multispectral resolution (10–30 m) is too coarse for individual
ordinary waves. It can still capture:

- surf-zone width and breaking patterns (foam is bright in NIR and SWIR);
- turbidity and sediment plumes (context for water colour and domain shift);
- land/water masks and shoreline position.

The project's imagery schema (`schemas/imagery.py`) records bands with centre
wavelength, bandwidth, and radiometric quantity (DN, radiance, reflectance), so
multispectral products share the acquisition framework without being treated
as equivalent to RGB video.

**Open question:** is multispectral information useful for *wave inference*
itself, or mainly as *domain context* for explaining generalisation failures?
