# Multitrial systematics with MLHEP

## Generate configurations (MLHEP yml databases) for each trial

File: `run-mlhep-fitter-multitrial.py`<br>
Usage: `python run-mlhep-fitter-multitrial.py database_file in_db_dir out_db_dir mlhep_results_dir_pattern`

Arguments:
- `database_file`: filename of the template database without the .yml extension, e.g., `database_ml_parameters_LcToPKPi`
- `in_db_dir`: path to the directory containing the database, e.g., `data/data_run3`
- `out_db_dir`: path to the directory for output multitrial databases, e.g., `multitrial_db`
- `mlhep_results_dir_pattern`: prefix of output directory name for fit results; for each trial, the trial name is appended to the directory name, and the resulting directory name is written under `Run3analysis/{data,mc}/prefix_dir_res` in the database file 

Adjust `DIR_PATH` in the script. It is the path to the base directory where you store directories with MLHEP results.

This script needs to be ran only once to generate databases.

Currently, the trials are hardcoded in the Python script. To add or modify a trial, you need to adjust `BASE_TRIALS` variable and the `process_trial` function.

## Get mass fits for each trial

File: `run-mlhep-fitter-multitrial.sh`<br>
Usage: `./run-mlhep-fitter-multitrial.sh`

The `submission/analyzer.yml` config is used.
The script automates running MLHEP for each trial. Mass histograms are copied before each MLHEP invocation, so as only the quick fit steps needs to be activated in `submission/analyzer.yml`

Adjust the variables before the `for` loop.<br>
The script includes also a call to `run-mlhep-fitter-multitrial.py`, which can be commented out. In this case, make sure to pass the same `OUT_DB_DIR`, `DB_PATTERN`, `DIR_PATTERN` values to the two scripts.

Before running, you need to create the directory structure for each MLHEP output. You can, for example, run the `.sh` script with the `cp` lines commented out. Then, MLHEP creates directories for each trial and fails quietly. Next, run the script with `cp` lines uncommented, and you will get the final output.

## Plot multitrial results

Files: `multitrial.py`, `config_multitrial.json`<br>
Usage: `python3 multitrial.py config_multitrial.json`

Adjust the sample `config_multitrial.json` to your needs.
