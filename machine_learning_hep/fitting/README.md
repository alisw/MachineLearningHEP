# Fitting

## Introduction

The fitting sub-module is split into two parts `fitting.py`, `utils.py` and `helpers.py`. The first two are in principle independent of this package and do not know anything about the structure. For instance, there is no usage of any database configuration. Hence, these classes and functions found in `fitting.py` and `utils.py` are self-consistent.
On the other hand, `helper.py` contains the interfaces between this package and the fit classes. Here, the desired configuration and initialisation for the fitters is extracted and the fit objects are instantiated accordingly. Finally, the class `MLFitter` is a wrapper around all fits needed in this package.

## Structure

### `fitting.py`

**FitBase**
This is the base class all concrete fit classes derive from. Each fit then has a kernel which can be accessed via `self.kernel`. This object will be defined and set in deriving classes and is responsible for the actual fitting procedure.

**FitROOT**
This is the base class for all fits depending on `ROOT`. Such an object collects at least `ROOT` objects in `self.root_objects` which can be serialised along with a corresponding fit object in order to recover the fit object in a later run again.

**FitAliHF**
This class uses the `AliInvMassFitter` as it is defined in [AliPhysics](https://github.com/alisw/AliPhysics/blob/master/PWGHF/vertexingHF/AliHFInvMassFitter.h) and hence it comes with all its features.

**FitROOTGauss**
This class implements a simple Gaussian fit.

### `utils.py`

**save_fit** and **load_fit** are used to serialise a fit object to disk or to read a serialised configuration back and construct a fit object from that.

### `helper.py`
Here, all `MLHEP` specific classes and helper functions are defined.

**MLFitParsFactory**
This class builds an abstraction layer and is responsible to understand the fit configuration given in the databases (such as `database_ml_parameters_<particlename>.yml`). Fit configurations are packed in a unified way to be used further to create and initialise fit according to those defined in `fitting.py`.

**MLFitter**
All fits used in an analysis run are handled here.


## Example usage

Here is a small example (as it actually looks like in the package but with some further comments)

```python

"""
Create an MLFitter object given
1. config_database: the configuration dictionary where the fit parameters can be found. This is forwarded to an MLFitParsFactory object internally.
2. analysis_type: the specified analysis-section to be lookd up in the config_database where the fit parameters for the specified analysis are defined
3. histogram_filepath_data, histogram_filepath_mc: file paths to ROOT files where histograms can be found which should be fitted.
"""
fitter = MLFitter(config_database, analysis_type, histogram_filepath_data, histogram_filepath_mc)

"""
This performs fits in inclusive bins of the second binning variable defined in the analysis section
of the database. These pre-fits are usedto initialise the central fit. How to do that is derived
from the database parameters and handled by the MLFitter and MLFitParsFactory objects.
"""
fitter.perform_pre_fits()

# Central fits are performed.
fitter.perform_central_fits()

# Specify a file where fit summary plots will be saved.
fileout_name = "summary_fit_plots.root"
fileout = TFile(fileout_name, "RECREATE")

"""
Fit plots are saved in the directory fit_plots_save_dir and summary plots are also saved in the
specified ROOT file (can also be an abstract TDirectory) given that it is not None.
"""
fitter.draw_fits(fit_plots_save_dir, fileout)
fileout.Close()

# Serialize all fits to the directory fit_save_dir
self.fitter.save_fits(fit_save_dir)

# ... do something in the meantime or re-start the analysis workflow ...

# Look for the fitter
if not fitter:
    fitter = MLFitter(config_database, analysis_type, histogram_filepath_data, histogram_filepath_mc)
    # Read back fits serialised to fit_save_dir if possible
    if not fitter.load_fits(fit_save_dir):
        print(f"FATAL: Cannot load fits from dir {fit_save_dir}")
        return

# Get a fit passing the bins ibin1 and ibin2 of the fit variables the fits where done in differentially
fit = fitter.get_central_fit(ibin1, ibin2)

# If the fit could not be loaded or was not successful back then, return (or do something else...)
if not fit:
    print(f"FATAL: Cannot access fit in bins ({ibin1}, {ibin2})")
    return
if not fit.success:
    print(f"Fit in bins ({ibin1}, {ibin2}) not successful, skip...")
    return
```
    






