# SUViz

This is a tool to process and plot the SU standard particle distribution file used in the [UK-XFEL conversion scripts](https://github.com/UKFELs/FXFEL). it consists of a series of modules that can be easily used to automatically plot the most common parameters and return arrays of the values for your own analysis. It also includes routines to plot the particle 2D phase spaces and/or slice properties in interactive html files for viewing in a web browser, or as simple png's. It includes some iPython Notebooks for examples of use cases, and an example `plots.py` script to quickly produce plots from an SU format particle file.

## To Install

Do

```
pip install .
```

in this directory to install as a Python package. The Python package `accsviz` will then be available for use in your own Python scripts or functions (see the supplied exemplar `plots.py` script for an example of use).

## Dependencies:

 - Numpy
 - pytables
 - Bokeh
 - SciPy
 - Pandas
