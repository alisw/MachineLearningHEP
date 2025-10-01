# Scripts for ALICE preliminary plots

## Invariant mass fits

File: `plot_invmass_fit_dzero_dplus_lambdac.py`<br>
Usage: `python plot_invmass_fit_dzero_dplus_lambdac.py config_invmass_preliminary.yml`

Example config in `config_invmass_preliminary.yml`. It was used to draw the plots:
- <https://alice-figure.web.cern.ch/node/34090>
- <https://alice-figure.web.cern.ch/node/34089>
- <https://alice-figure.web.cern.ch/node/34088>

The script is passed in different versions around the D2H people. Here, it contains my few improvements, e.g., configurable multiplicity label.<br>
I also commented out lines related to non-prompt particles as we had results only for the prompt case.

You still need to adjust the script in several places:
- comment/uncomment the lines related to prompt/non-prompt particles
- adjust the output directory in line 177
- input filename in `get_name_infile()`
- histogram names in `main()`

## Cut variation results

File: `DrawCutVarFit.C`<br>
Usage: `root -x DrawCutVarFit.C` in the ROOT / O2 shell

Used to draw the plot <https://alice-figure.web.cern.ch/node/31345>.

Adjust the script:
- set the `bdtScoreCuts_...` variables to your final BDT cuts
- set `binMin` and `binMax` to the pT bin you want to plot
- set `bdtScoreCuts` to the proper `bdtScoreCuts_...` variable
- adjust `bdtScoreCuts_toPlot` and the corresponding indices in `bdtScoreCuts_toPlot_ind`; they are the cuts to label on the x-axis
- adjust the input file name and histogram names in `DrawCutVarFit()`
- adjust x-axis title, if needed
