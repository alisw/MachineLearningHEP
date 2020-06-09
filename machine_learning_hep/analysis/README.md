# Analysis and systematics

## Overview

First of all, everything in here is basically an **Analyzer**. These objects can be handled by an `AnalysisManager`. 


## Applying additional analysis cuts

In order to place additional cuts before a mass histogram is filled, those have to be set in the corresponding analysis section in the database. There, one cut must be put per analysis pT bin. If no cut should be applied, just put `Null`. The flag `use_cuts` controls whether the cuts should be applied or not. Otherwise, cuts are formulated as strings which are directly used in a `pandas.DataFrame.query` meaning that all names used **must** exist as a column in the dataframe in the analysis. An example implementation in the database could look like

```yaml
# within an analysis section, assuming 4 pT bins
  use_cuts: True
  cuts:
    - "p_prong0 > 2 or p_prong1 < 1"
    - Null
    - "abs(eta_cand) < 1.2"
    - Null
```

The cuts can then be accessed in `processer_<type>.process_histomass_single`. The database flag `use_cuts` is translated into the member `self.do_custom_analysis_cuts` and should be checked whether it's `True` in order to not circumvent it's purpose. Then, there is a helper function in `Processer` so if you have a dataframe corresponding to a certain pT bin, you can just do

## Using efficiencies from another analysis

To use the efficiencies from another analysis for a certain multiplicity bin one can use the fields `path_eff` and `mult_bin_eff` when using the analyzer class `AnalyzerDhadrons_mult`. When using this feature, both fields have to contain a list as long as the number of multiplicity bins. The first list lists the corresponding file to be used and  the second list entries are integers referring to the i'th multiplicity bin histogram inside the file. `null` entries can be used to use the efficiencies of this very analysis multiplicity bin (which is of course the default when none of the lists is present).


```python
if self.do_custom_analysis_cuts:
    df = self.apply_cuts_ptbin(df, ipt)

```

which would apply the cuts defined for the `ipt`'th bin and returns the skimmed dataframe. Nothing is done when there was no cut defined and you would just get back the dataframe you put in.

## Analysis and systematic implementation and workflow

A specific analysis or systematics is derived from `Analyzer`. This `AnalyzerDerived` can then implement any analysis step method. Note, that passing arguments to those methods is at the moment not supported. However, as they have access to the entire configuration via the database dictionary, this will probably not be needed as all specifics can be derived from that database.
An object of `AnalyzerDerived` can be tried to run the step `ana_step` by doing

```python
method_executed = analyzer_derived.analysis_step("ana_step")
```

If a method with this name is found, it will be executed. If not, this returns `False`. (Of course, `analyzer_derived.ana_step()` works just as well...).

`Analyzer` objects can be wrapped into an `AnalyzerManager` which has the following constructor

```python
class AnalyzerManager:
    def __init__(self, database, case, typean, doperiodbyperiod, *args):
        ...
```

where `database` is the dictionary derived from the `database_ml_parameters_<particle>.yml` and contain hence all information needed. `case` is passed for backwards-compatibility in some sense but might become obsolete at some point as it is basically the particle name and not used to specifically run/determine anything. On the other hand, `typean` is the analysis to be run and needed to pick up the correct parameters from `database` as they might differ from analysis to analysis. The boolean variable `doperiodbyperiod` specifies whether analyses should be run for each single specified data-taking period specified in `database`. `*args` gives the opportunity to specify arguments which should be passed to each `Analyzer` in addition.

Say now, we have a class `AwesomeAnalyzer` deriving from `Analyzer` so we can do the follwing

```python
ana_mgr = AnalyzerManager(AwesomeAnalyzer, database, case, typean, add_arg1, add_arg2)
ana_steps = ["fit", "make_cross_section", "plot_all"]

# The "*" is absolutely necessary here as it kind of resolves the list into single arguments...
ana_mgr.analyze(*ana_steps)

# ...so you could also do...
ana_mgr.analyze("fit", "make_cross_section", "plot_all")

# ...or only
ana_mgr.analyze("summarize")
```

Note that the analysis steps are executed in the exact same order they were passed. The `AnalyzerManager` then first loops over each step and inside this loop it loops over all `Analyzer`s. The other way round might be dangerous as some steps might depend on others and it is could be the case that it is not accounted for that in the specific implementation of the `AwesomeAnalyzer` (ok, then it might be not so awesome...).

In the very same way it works with systematics. A class handling those would as well derive from `Analyzer` and it can hence as well be treated by an `AnalyzerManager` object in the very same way.

## Implementing an after-burner

If `doperiodbyperiod` was enabled, it might be necessary to run some final jobs, for instance in order to merge data. Therefore, `Analyzer` implements a method `get_after_burner(self)` which, in the default implementation, just returns `None`. If so, the `AnalyzerManager` is smart enough to not run any after-burner steps. If the method is implemented, an object deriving from `AnalyzerAfterBurner` is expected (which in turn derives actually from `Analyzer`).

Everytime a specific analysis step has been run for all period-analyses, the after-burner is invoked, basically by 

```python
after_burner.analysis_step("fit")
```
Classes deriving from `AnalyzerAfterBurner` have access to all per-period `Analyzer`s through the member list `analyzers`. Hence, they can access all of them in the corresponding method. There are two things to note here:

1. The after-burner is called for each analysis step, however it is save in case that is not implemented - simply nothing happens.
2. To be meaningful, the after-burner method has to have the same name as the individual analysis step done before.

One use-case of the after-burner is for example the class `Systematics` in `systematics.py` which at that moment does systematic studies of the ML working point (basically a variation) and of the MC pT shape.

## Implementing an Analyzer

Any analyzer or systematic class derived from `Analyzer`. That means, you start off like this

```python
from machine_learning_hep.analysis.analyzer import Analyzer

class AwesomeAnalyzer(Analyzer):
    def __init__(self, datap, case, typean, period, few, more, arguments):
        super().__init__(datap, case, typean, period)

    # awesome implementations
```

It is required hat the base class gets the database dictionary, the analysis type, the particle case and the period. Hence, these four arguments need to correspond to the first four arguments of your `AwesomeAnalyzer`. After the base classe's `__init__` has been called these are automatically available in your `AwesomeAnalyzer` as class members

* `self.datap`
* `self.case`
* `self.typean`
* `self.period`

In addition there is also a logger in `self.logger` you can use to issue more important output for the user.

`self.period` will have the period number ranging from `0` to `n_period - 1`. It is `None` is this an analyzer has to expect merged period input. This info can be used if, for instance, a method should only be executed for a certain pariod or a period-merged analysis. At the beginning of such a method you might write

```python
    def my_analysis_step(self):
        if self.period is None:
            return
        # Following implementation only run for per-period run
```



