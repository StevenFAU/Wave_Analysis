# ERA5 fixture

`era5_waves_oak_island_20260615_20260915.nc` (125,787 bytes, sha256
`6dddf426ca2e5e24d9a7418a2823548ba5f0fb48ed462c2156da28f33c106237`) is an
unmodified file returned by the Copernicus Climate Data Store on 2026-09-26 for
this request to `reanalysis-era5-single-levels`:

```json
{
  "product_type": ["reanalysis"],
  "variable": ["significant_height_of_combined_wind_waves_and_swell", "peak_wave_period",
               "mean_wave_period", "mean_wave_period_based_on_first_moment",
               "mean_zero_crossing_wave_period", "mean_wave_direction",
               "significant_height_of_wind_waves", "mean_period_of_wind_waves",
               "mean_direction_of_wind_waves", "significant_height_of_total_swell",
               "mean_period_of_total_swell", "mean_direction_of_total_swell"],
  "year": ["2026"], "month": ["06", "09"], "day": ["15"], "time": ["00:00", "12:00"],
  "area": [34.9, -79.2, 32.9, -77.2],
  "data_format": "netcdf", "download_format": "unarchived"
}
```

The CDS snapped the area to the 0.5 deg grid (34.5 to 33.0 N, 79.0 to 77.5 W:
16 points, 4 of them land). The June times are final ERA5 (`expver` 0001), the
September times preliminary ERA5T (`expver` 0005). The box covers the WebCOOS
Oak Island cameras.

Contains modified Copernicus Climate Change Service information 2026. Licence:
CC-BY 4.0. Citation: Hersbach et al. (2023), ERA5 hourly data on single levels
from 1940 to present, C3S Climate Data Store, DOI 10.24381/cds.adbb2d47.
