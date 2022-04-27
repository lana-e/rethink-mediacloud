# rethink-mediacloud
Repo for ReThink's MediaCloud API, built to analyze 9/11 anniversary coverage and expanded for more general use. Objectives include:
- Create custom news outlet lists to search (e.g. ReThink’s standard Novetta media outlet lists)
- Output “news hole” line graphs
- Use word counts to create word clouds

The main notebooks to use in the `notebooks/` directory are `ReThink MediaCloud API User Guide.ipynb` (this notebook includes full docstrings for the analysis and datavis functions) and `ReThink MediaCloud API User Template.ipynb` (this notebook is built to work out-of-the-box for specific queries). To use the notebooks, the user must have a `.env` file with a MediaCloud API key defined as `MC_API_KEY` within the working environment.
