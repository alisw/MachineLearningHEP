# Derive MC event weights


First you need configuration files like those for [V0M percentile](perc_v0m).

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
