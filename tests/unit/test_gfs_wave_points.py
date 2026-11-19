"""Check missing totals, overflow flags and short-lead timestamps in station bulletins."""

from wave_analysis.sources.gfs_wave import short_leads


def test_first_seven_hours():
    text = "Location : 41025\nCycle : 20260930 18 UTC\n"
    for lead in range(7):
        day, hour = (30, 18 + lead) if lead < 6 else (1, 0)
        cells = [
            f"{day} {hour}",
            "0" if lead == 0 else "1.30 7 1",
            "* 1.20 7.4 250" if lead else "",
            "",
            "",
            "",
            "",
            "",
        ]
        text += " | " + " | ".join(cells) + " |\n"
    rows = short_leads(text)
    assert rows[0]["hs_m"] is None and rows[0]["systems"] == 0
    assert rows[1]["system1_wind"] and rows[1]["extra_systems"] == 1
    assert rows[-1]["valid_utc"].month == 10 and rows[-1]["lead_h"] == 6
