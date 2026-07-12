# Data

The raw UCI household power consumption file is not committed to this repository because it is large and can be downloaded from the original source.

Dataset source:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The Streamlit app downloads the ZIP file on first run and caches reusable aggregate series under `.cache/`.

Git ignores raw data files and cache outputs so the repository stays lightweight.
