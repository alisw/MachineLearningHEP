# Obtain various comparison plots, esp. for systematics and final analysis results

File: `compare_fractions.py`
Usage: `python compare_fractions.py config.json`

All JSON files in this directory provide various configuration examples for different use cases.

The script can read both model histograms and analysis result histograms, as specified in the JSON configuration. It plots all histograms on a single plot with different colours, and calculates the systematic errors, if no systematics is provided. They are printed in the console. The systematic errors can be provided in the JSON config, and then they are drawn as boxes around the central points.

The script plots also a separate plot with the ratios of other histograms to the central histogram. The central histogram is the one specified as "default" in the JSON.

The histogram labels in legend are taken from the dictionary labels in the JSON, which can be specified with the TLatex syntax.

It is also possible to specify the `y_axis` title and an additional description under the "ALICE Preliminary" header (`alice_text` variable in the config). The header itself and its position can be adjusted in the `get_alice_text` function in the Python script.

Colors and markers can be adjusted at the beginning of the script. 
