# Run MLHEP

## A simple shortcut with default running options

File: `run-mlhep.sh`<br>
Usage: `./run-mlhep.sh my_database.yml my_run_config.yml logfile.log`

It calls:
```
mlhep --log-file logfile.log \
  -a Run3analysis \
  --run-config my_run_config.yml \
  --database-analysis my_database.yml
```

## Run MLHEP in batch for various BDT cuts

File: `run-mlhep-batch.sh`<br>
Usage: `./run-mlhep-batch.sh`

The script requires an MLHEP database with %resdir%, %bkg...%, and %fd% placeholders.
You can see the examples of placeholders in the `data/data_run3/database_ml_parameters_LcToPKPi_multiclass_fdd.yml` database.

The script loops over different cuts defined with `seq`. By default, they are non-prompt cuts. To run for prompt cuts, you need simply to put the %fd% placeholders at the place of prompt cuts in `probcutoptimal` variables in the database. Background cuts are set to the `bkg` value.

For each non-prompt cut, the MLHEP workflow is launched with %resdir% output directory set to `${RESDIR}/${RESDIR_PATTERN}${suffix}`, where `RESDIR` is the main MLHEP output directory, `RESDIR_PATTERN` is the prefix of the output directory name, and suffix is `fd_${fd}`, where `${fd}` is the current cut value.

The MLHEP workflow is defined by `submission/analyzer.yml` file. Usually, you would enable `histomass` and `efficiency` steps for data and MC, and `fit` and `efficiency` steps in the "Inclusive hadrons" section.

Adjust the script variables and the `submission/analyzer.yml` file to your needs.
