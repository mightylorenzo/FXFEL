import processing_tools as pt 
import glob, os, shutil
import re

def sort_nicely(strings):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    strings.sort( key=alphanum_key )


def directory_list(directory, file_ending='.h5'):
    '''returns a list of files with a given ending'''
    files = []
    for file in os.listdir(directory):
        if file.endswith(file_ending):
            files.append(os.path.join(directory, file))
    return files



def make_directories(directory, file_ending='.h5'):
    '''Makes a directory for every file with a certain ending split
    at the full stop'''
    full_path = os.path.join(directory, '*'+file_ending)
    print full_path
    for file_path in glob.glob(full_path):
        new_dir = file_path.rsplit('.', 1)[0]
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)



def move_plots(directory):
    '''Moves all png files and html files into their respective
    base filename directories'''
    for file_path in glob.glob(os.path.join(directory, '*.h5')):
        new_dir = file_path.rsplit('.', 1)[0]
        file_path_1 = file_path.rsplit('.', 1)[0]+'*'+'.png'
        file_path_2 = file_path.rsplit('.', 1)[0]+'*'+'.html'
        for files in glob.glob(file_path_1):
            shutil.move(files, new_dir)
        for files in glob.glob(file_path_2):
            shutil.move(files, new_dir)

def plot_defaults(directory, file_ending='.h5',
                  undulator_period=0.0275, peak_field=0,
                  k_fact=1, num_slices=False):
    '''Plots every file in a directory and moves it into a folder'''
    make_directories(directory, file_ending='.h5')

    files = directory_list(directory,file_ending)

    for i in files:
        data = pt.ProcessedData(
            i, undulator_period, 
            peak_field, k_fact, num_slices)
        plots = pt.Panda_Plotting(data)
        plots.plot_defaults()
        bokeh = pt.Bokeh_Plotting(data)
        bokeh.prepare_defaults(file_name=i[:-3])
        bokeh.plot_defaults()

    move_plots(directory)
