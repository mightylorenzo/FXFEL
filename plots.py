from accsviz import processing_tools as pt
import sys

def plots(filepath):

# Init FEL plotting object

    FEL = pt.ProcessedData(filepath,num_slices=50,undulator_period=0.0275,k_fact=1.13)

# plot default FEL plots

    plots = pt.Bokeh_Plotting(FEL)  #to save pngs you need 'npm install -g phantomjs-prebuilt'
    plots.prepare_defaults(file_name='test')
    plots.plot_defaults(show_html=True) #show_html=True


if __name__ == '__main__':

    if len(sys.argv)==2:
        fname = sys.argv[1]
        print 'Processing file:', fname
        plots(fname)
    else:
        print 'Usage: plots.py <FileName> \n'
        sys.exit(1)
