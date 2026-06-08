# SPS Settlement 3D

Interactive 3D visualization of foundation settlement at Amundsen-Scott South Pole Station. Built with Dash and Three.js.

## Features

- 3D pile and beam model with vertical exaggeration
- Three view modes: Fixed Datum, Settlement Bowl, Relative to Mean
- Piles colored by cumulative settlement or settlement rate
- Beams colored by floor differential (red ≥ 2 in)
- Grade-level beam connections (Dec 2017+)
- Toggleable reference planes: mean, best-fit all, Pod A, Pod B
- Plan view settlement heatmap
- Settlement history sparklines with linear forecast
- Hover tooltips on piles (settlement, rate, shim pack) and beams (differential)
- HTML report export

## Setup

```bash
conda create -n sps-settlement python=3.11
conda activate sps-settlement
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8050 in your browser.

## Usage

1. Click **Upload Survey Excel** and select your survey workbook
2. Click **Compute**
3. Use the date slider to step through survey history and forecast dates

The workbook must contain three sheets: `SURVEY DATA`, `TRUSS DATA`, and `SHIM DATA`.  
`SP_BeamArrowLabels.csv` must remain in the same directory as `app.py`.
