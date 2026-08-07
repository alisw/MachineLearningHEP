# Verify different stages of MLHEP processing

## Check MLHEP output data files

File: `check_parquet.py`<br>
Usage: `python check_parquet.py in_file.parquet`

The Python script contains some examples of how to read from a parquet file, print some useful information, and plot histograms.

It can be used to check MLHEP skimming, training, and application outputs by testing individual parquet files.

## Compare prompt fractions calculated with different inputs or methods

Files: `plot_prompt_fraction_vs_crosssec_configs.py`, `config_fraction_vs_crosssec_configs.json`<br>
Usage: `python plot_prompt_fraction_vs_crosssec_configs.py config_fraction_vs_crosssec_configs.json`

Adjust the JSON config. You can provide as many histogram files in the `hists` dictionary as you want.
By adjusting `histoname`, you can plot also the non-prompt fraction.

## Plot prompt fraction vs different BDT cuts

Files: `plot_prompt_fraction_vs_bdt_cuts.py`, `config_fraction_vs_bdt_cuts.json`<br>
Usage: `python plot_prompt_fraction_vs_bdt_cuts.py config_fraction_vs_bdt_cuts.json`

Adjust the JSON config. Here, you provide a glob pattern to all files of interest.
By adjusting `histoname`, you can plot also the non-prompt fraction.
