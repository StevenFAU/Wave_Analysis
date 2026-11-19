# EarthScope vertical ground motion

Waimea Bay pairs with IU.KIP.00.BHZ on Oahu. Oak Island pairs with N4.Y60A.00.HHZ in North Carolina. Acquire one historic UTC day with `scripts/seismic_day.py STATION YYYY-MM-DD`. The source adapter stores the served miniSEED in `data/raw/earthscope/` with a per-station retrieval ledger. EarthScope recommends limiting each request to one station and one day. [EarthScope dataselect](https://service.earthscope.org/fdsnws/dataselect/1/)

Read each day with `uv run --no-project --with obspy python scripts/seismic_day_qc.py FILE`. Check station, channel, rate, boundaries, gaps and overlaps. The initial Welch check uses detrended 256-second windows of raw digital counts and selects the strongest 0.1–0.3 Hz peak in each complete hour. Local geology, instrument response, earthquakes and distant seas influence this signal. Obtain epoch-matched StationXML and correct instrument response before comparing amplitudes across instruments or deployments.

Two open 24-hour samples passed the continuity check. At IU.KIP (2011-01-15, 20 Hz), the median hourly secondary-band peak was 0.168 Hz; at N4.Y60A (2023-01-15, 100 Hz), 0.217 Hz. These demonstrate coverage and readable signal. A matched buoy comparison on held-out dates is needed to measure predictive skill.
