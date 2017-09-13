# -*- coding: utf-8 -*-
"""
# Copyright (c) 2012-2017, University of Strathclyde
# Authors: Daniel Bultrini
# License: BSD-3-Clause
"""

import numpy as np
import pandas as pd

from bokeh.layouts import layout
from bokeh.layouts import widgetbox

from bokeh.embed import file_html

from bokeh.io import show
from bokeh.io import output_notebook

from bokeh.models import Text
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Line
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import HoverTool
from bokeh.models import LinearAxis
from bokeh.models import ColumnDataSource
from bokeh.models import SingleIntervalTicker

from bokeh.palettes import Spectral6
import processing_tools as pt



def interactive_plot(filenames,x_axis,y_axis,num_slices=100, undulator_period=0.0275,k_fact=1):
    ''' Creates an interactive plot that slides throuigh a list of files
    uses java magic and is more for convenience/one offs as exporting these plots 
    tends to break a few things'''
    sources = {}

    data_array = []
    obj_array = [0]*len(filenames)
    for i, filename in enumerate(filenames):
        obj_array[i] = pt.ProcessedData(filename, num_slices=num_slices, undulator_period=undulator_period, k_fact=k_fact)
        if x_axis != 'z_pos':
            data_array.append(obj_array[i].DistFrame())
        if y_axis == 'MX_gain' or y_axis == '1D_gain' or y_axis == 'pierce':
            data_array.append(obj_array[i].FELFrame())
        else:
            data_array.append(obj_array[i].StatsFrame())

    sections = range(len(data_array))
    rng_x, rng_y = [9999.0,-9999.0], [99999.0,-9999.0]
    
    for i in data_array:
        if i[x_axis].max() > rng_x[1]:
            rng_x[1] = i[x_axis].max()
        if i[x_axis].min() < rng_x[0]:
            rng_x[0] = i[x_axis].min()
        if i[y_axis].max() > rng_y[1]:
            rng_y[1] = i[y_axis].max()
        if i[y_axis].min() < rng_y[0]:
            rng_y[0] = i[y_axis].min()

    rng_x = (rng_x[0],rng_x[1])
    rng_y = (rng_y[0],rng_y[1])



    for section in xrange(len(data_array)):
        x       = data_array[section][x_axis]
        x.name  = x_axis
        y            = data_array[section][y_axis]
        y.name       = y_axis
        #z      = data_array[section]['z']
        #z.name = 'z'


        new_df = pd.concat(
                    [x, y],
                    axis=1
        )
        sources['_' + str(section)] = ColumnDataSource(new_df)

    dict_of_sources = dict(zip(
                        [i for i in sections],
                        ['_%s' % i for i in sections])
                        )

    js_source_array = str(dict_of_sources).replace("'", "")
    xdr  = Range1d(*rng_x)
    ydr  = Range1d(*rng_y)
    plot = Plot(
        x_range=xdr,
        y_range=ydr,
        plot_width=800,
        plot_height=400,
        outline_line_color=None,
        #toolbar_location=None,
        min_border=20,
    )

    AXIS_FORMATS = dict(
        minor_tick_in=None,
        minor_tick_out=None,
        major_tick_in=None,
        major_label_text_font_size="10pt",
        major_label_text_font_style="normal",
        axis_label_text_font_size="10pt",

        axis_line_color='#AAAAAA',
        major_tick_line_color='#AAAAAA',
        major_label_text_color='#666666',

        major_tick_line_cap="round",
        axis_line_cap="round",
        axis_line_width=1,
        major_tick_line_width=1,
    )

    xaxis = LinearAxis(
        #ticker     = SingleIntervalTicker(interval=1),
        axis_label = x_axis,
        **AXIS_FORMATS
    )
    yaxis = LinearAxis(
        #ticker     = SingleIntervalTicker(interval=20),
        axis_label = y_axis,
        **AXIS_FORMATS
    )   

    plot.add_layout(xaxis, 'below')
    plot.add_layout(yaxis, 'left')

    text_source = ColumnDataSource({'section': ['%s' % sections[0]]})
    text        = Text(
                    x=2, y=35, text='section',
                    text_font_size='150pt',
                    text_color='#000000'
                    )
    plot.add_glyph(text_source, text)

    # Add the circle
    renderer_source = sources['_%s' % sections[0]]
    if x_axis != 'z_pos':
        circle_glyph    = Circle(
                            x=x_axis, y=y_axis,
                            line_color='#6b0000',
                            line_width=0.5, line_alpha=0.5
                            )
    else:
        circle_glyph    = Line(
                            x=x_axis, y=y_axis,
                            line_color='#6b0000',
                            line_width=0.5, line_alpha=0.5
                            )

    circle_renderer = plot.add_glyph(renderer_source, circle_glyph)



    code = """
        var section = slider.get('value'),
            sources = %s,
            new_source_data = sources[section].get('data');
        renderer_source.set('data', new_source_data);
        text_source.set('data', {'section': [String(section)]});
    """ % js_source_array

    callback = CustomJS(args=sources, code=code)
    slider   = Slider(
                start=sections[0], end=sections[-1],
                value=0, step=1, title="Section",
                callback=callback
                )
    callback.args["renderer_source"] = renderer_source
    callback.args["text_source"] = text_source
    callback.args["slider"] = slider

    return layout([[plot], [slider]], sizing_mode='scale_width')

