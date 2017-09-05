import processing_tools as pt 
import glob, os, shutil
import re

def sort_nicely(strings):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    strings.sort( key=alphanum_key )


def directory_list(directory,file_ending='.h5'):
    files = []
    for file in os.listdir(directory):
        if file.endswith(file_ending):
            files.append(os.path.join(directory, file))
    return files



def make_directories(directory):
    for file_path in glob.glob(os.path.join(directory, '*.*')):
        new_dir = file_path.rsplit('.', 1)[0]
        try:
            os.mkdir(os.path.join(directory, new_dir))
        except OSError:
            pass
def move_plots(directory):
    for file_path in glob.glob(os.path.join(directory, '*.h5')):
        new_dir = file_path.rsplit('.', 1)[0]
        file_path_1 = file_path.rsplit('.', 1)[0]+'*'+'.png'
        file_path_2 = file_path.rsplit('.', 1)[0]+'*'+'.html'
        for files in glob.glob(file_path_1):
            shutil.move(files, new_dir)
        for files in glob.glob(file_path_2):
            shutil.move(files, new_dir)

folder = '/home/daniel_b/Documents/Summer_project/test'
def plot_defaults(directory,file_ending='.h5'):
    x = directory_list(directory,file_ending)
    make_directories(directory)
    for i in x:
        data = pt.ProcessedData(i,0.00275, num_slices=100)
        plots = pt.Panda_Plotting(data)
        plots.plot_defaults()
        bokeh = pt.Bokeh_Plotting(data)
        bokeh.prepare_defaults(file_name=i)
        bokeh.plot_defaults()
    move_plots(directory)
