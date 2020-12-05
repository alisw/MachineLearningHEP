# Derive MC event weights


First you need configuration files like those for [V0M percentile](perc_v0m).

A complete `YAML` configuration looks like

```yaml


database: /my/database/path/database_ml_parameters_hadron.yml

analysis: MBvspt_perc_v0m                           # The analysis in the database one is interested in.
                                                    # The trigger flags are derived from that


data_or_mc: data                                    # or mc

# optional
query_all: "is_ev_rej == 0"                         # The entire loaded dataframe is skimmed with that,
                                                    # one can use any existing column
                                                    # Afterwards, the dataframe is skimmed down to
                                                    # required columns.

# optional
required_columns:
    - col_name1
    - col_name2
    - col_name23

# optional
slice_cuts:                                         # The distribution of the same variable is derived in
                                                    # these slices. So one ends up with n_slices plots per
                                                    # year of data taking. Column names used here must be
                                                    # specified in "required_columns" field.
    - "col_name1 > val1 and col_name2 <= val2"
    - "col_name23 != val23"
    - null

distribution: n_tracklets                           # The column name of the distribution to be plotted.
                                                    # No need to specify this in "required_columns"

x_range:                                            # x-axis range (bins, low, up)
    - 100
    - 0
    - 100


# optional
use_mass_window: false                              # Whether or not to use a mass window cut on selected

                                                    # candidates.
#optional
use_ml_selection: false                             # Whether or not to use ML selection if model is available


out_file: n_tracklets_MB_D0_mc.root                 # File where the histograms are written to


```

Then, run each of these with

```bash
python derive_weights.py distr <path_to_config>
```

Afterwards, you will have the correspoding `ROOT` files with the distributions per data taking year.

Then run

```bash
python derive_weights.py weights <path_to_data_root> <path_to_mc_root>
```

of those you want to divide. This will produce `weights_<path_to_data_root>` with the weights per period.
