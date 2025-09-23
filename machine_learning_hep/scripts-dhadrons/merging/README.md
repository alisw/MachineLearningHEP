# Merging histograms and files

## Merge multiple histograms from multiple input files

Files: `merge_histos.py`, `merge-cutvar.sh`, `merge-yields.sh`
Usage: `python merge_histos.py -o out_file.root -n histName1 -n histName2 -i in_file1.root -i in_file2.root`

You can provide as many histogram names as you want. All histograms should be 1-dimensional and have the same x-axis. If no histogram name is provided, the script will merge all 1-dimensional histograms from the input files.

Provide one input file per x-axis bin. File names can be repeated.

Merge histograms `histName1` and `histName2` from the input files and save them in the output file. For each histogram name provided, e.g., `histName1`, "merging" means creation of a single output histogram with bin 1 content set to the content of bin 1 in `histName1` in `in_file1.root`, bin 2 content set to the content of bin 2 in `histName1` in `in_file2.root`, and so on. Particularly, the x-axis can represent pT, and the script can be used to merge results obtained for different pT bins.

The bash files `merge-cutvar.sh` and `merge-yields.sh` provide examples of using this Python script for merging cut variation results and O2Physics D2H fitter results, respectively.

## Merge the outputs of the MLHEP histomass step

Files: `merge_histomass.py`, `merge-mlhep.sh`
Usage: `python merge_histomass.py -o out_file.root -n histName1 -n histName2 -i in_file1.root -i -in_file2.root`

This script is different from the previous one as it is adjusted to the layout of MLHEP `masshisto.root` files, which contain 1 invariant mass histogram per pT bin.

Histogram names `histName1`, `histName2` are treated as patterns (substrings) of histograms to merge. For `masshisto.root` files, the pattern can be `hmassfPt`, which matches all histograms like `hmassfPt0_1_0.010.000.000`, `hmassfPt1_2_0.020.400.000`, and so on.

You can provide as many histogram name pattern as you want.
Provide one input file per pT bin. Each file should contain one matching histogram per pT bin. File names can be repeated.

The merging creates a single output file with histogram for the 1st pT bin from `in_file1.root`, histogram for the 2nd pT bin from `in_file2.root`, and so on.

`merge-mlhep.sh` is an example that uses `merge_histomass.py` to obtain a single invariant mass file for the O2Physics D2H mass fitter. The script makes also use of `merge_histos.py` to get a single efficiencies file to be used in the cut variation macro.

## Gather MLHEP efficiencies and mass fits for cut variation

File: `gather-inputs-cutvar.sh`
Usage: `./gather-inputs-cutvar.sh`

To get MLHEP results for different non-prompt cuts, different output directories must be set. Otherwise, the results get overwritten. However, the cut variation script requires the input efficiency and mass fit files to be in a single directory.

This script takes all `efficienciesLcpKpiRun3analysis.root` and `yields_LcpKpi_Run3analysis.root` MLHEP output files from the directories that match `RESDIR_PATTERN`, and puts them in the `OUTPUT_DIR`. To differentiate the files, the suffix made of the corresponding directory name with `RESDIR_PATTERN` removed is appended to a file name.<br>
`PERM_PATTERN` is also used to match directories, but it is not removed from the suffix.

For example, given `RESDIR_PATTERN`: `/data/MLHEP/results-today_`, `PERM_PATTERN`: `non-prompt_`, and directories like: `/data/MLHEP/results-today_non-prompt_0.1`, `/data/MLHEP/results-today_-prompt_0.2`, the resulting efficiency file names are: `efficienciesLcpKpiRun3analysis_non-prompt_0.1.root`, `efficienciesLcpKpiRun3analysis_non-prompt_0.2.root`.

Adjust `MLHEP_DIR`, `OUTPUT_DIR`, `RESDIR_PATTERN` and `PERM_PATTERN` in the script.

You might also need to adjust the regular expression in line 12 and file paths in the for loop. 
