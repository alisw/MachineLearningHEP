# Analysis and systematics

## Overview

First of all, everything in here is basically an **Analyzer**. These objects can be handled by an `AnalysisManager`. 


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

