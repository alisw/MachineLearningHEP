# Validation

## Check for duplicated event IDs

The script `find_duplicates_event.py` checks whether there are duplicates in the `AnalysisResultsEvtOrig.pkl.lz4` based on the combination of `["ev_id", "ev_id_ext", "run_number"]`.

It only relies on the database to be used and checked. To customise, open the script and scroll down to the section **MAIN** and put values accordingly. If you also want to see all the values of duplicated event IDs, set `EXTRACT_DUPL_INFO = True`

The summary is dumped into a `YAML` file whose name can also be customised as you can see.

By default, all files with their `dupl/all` ratio are summarised and to make it easier when searching the file, a flag `has_duplicates` is `true` whenever the previous ratio is `!=0`.

If `EXTRACT_DUPL_INFO = True` in the script, also the duplicated `["ev_id", "ev_id_ext", "run_number"]` are added.
