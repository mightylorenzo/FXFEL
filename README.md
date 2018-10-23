# SUViz #

This is a tool to process and plot the SU standard particle distribution file used in the [UK-XFEL conversion scripts](https://github.com/UKFELs/FXFEL). it consists of a series of modules that can be easily used to automatically plot the most common parameters and return arrays of the values for your own analysis. It also includes routines to plot the particle 2D phase spaces and/or slice properties in interactive html files for viewing in a web browser, or as simple png's. It includes some iPython Notebooks for examples of use cases, and an example `plots.py` script to quickly produce plots from an SU format particle file.

To run `plots.py`, simply pass it the name of the file you wish to analyze and plot. So, for instance, if the SU file `test.h5` is in our current directory, do
```
python plots.py test.h5
```
and an .html file will be opened in your browser containing the plots. Note the tabs at the top which allow you to view different types of plots.

There are some Jupyter Notebooks in the `notebooks` directory, which include examples of use of the API. `Default Plots.ipynb` contains simple instructions for generating the default set of plots in an `.html` file - the example script `plots.py` mentioned above basically accumulates those commands into one script. `Particle Distribution Visualization.ipynb` contains a more in-depth usage and customization of plots and calculations. Before using the example notebooks, install the package as shown below.

## To Install

We recommend using Anaconda, upon which this software has been tested and used. Anaconda will also allow the use of the Jupyter Notebook examples in the `notebooks` directory. Grab this source code by cloning the repo or downloading a zip file of the source. To clone, do
```
git clone https://github.com/mightylorenzo/FXFEL.git fxfelviz
```

which will create the `fxfelviz` directory and download the file into it. Then, go into the new directory, and run `pip` to install, so:

```
cd fxfelviz
pip install .
```

Note the period `.` in the last line. The Python package `accsviz` will then be available for use in your own Python scripts or functions (see the supplied exemplar `plots.py` script for an example of use).

## Dependencies

 - Numpy
 - pytables
 - Bokeh
 - SciPy
 - Pandas
