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
