# helper functions used to plot results

import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler  # when using resampler, exported html's are not interactive, i.e., low res
import os
import re
from datetime import datetime

import input_data_helper as idh  # needed for add_generation_and_demand_trace

TIMEZONE_DEFAULT = 'Europe/Berlin'

# --- plot formatting --------------------------------------------------------------------------------------------------
SUBPLOT_H_SPACING_REL = 0.25  # 0.25  # 0.2  # 0.12  # 0.03, was 0.04
SUBPLOT_V_SPACING_REL = 0.1  # 0.2  # 0.35  # 0.3  # 0.35  # 0.21  # was 0.035
SUBPLOT_LR_MARGIN = 30
SUBPLOT_TOP_MARGIN = 60  # 130  # 120  # 0
SUBPLOT_TOP_MARGIN_WITH_DETAILS = 185
SUBPLOT_BOT_MARGIN = 0
SUBPLOT_PADDING = 0

HEIGHT_PER_ROW = 180  # in px
PLOT_WIDTH = 1350  # 1850  # in px
# PLOT_HEIGHT = HEIGHT_PER_ROW * SUBPLOT_ROWS -> we need to figure this out dynamically for each plot

# Title: aging type
PLOT_TITLE_RE = "<b>%s</b><br>%s"
PLOT_TITLE_Y_POS_REL = 20.0  # 30.0

X_AXIS_TITLE = 'Date'

PLOT_HOVER_TEMPLATE = "<b>%{text}</b><br>%{y:.4f}<br><extra></extra>"
PLOT_TEXT = f"%s"

TITLE_FONT_SIZE = 17
SUBPLOT_TITLE_FONT_SIZE = 16
AXIS_FONT_SIZE = 16
AXIS_TICK_FONT_SIZE = 14

# figure settings
FIGURE_TEMPLATE = "custom_theme"  # "custom_theme" "plotly_white" "plotly" "none"

BG_COLOR = 'rgba(255, 255, 255, 127)'  # '#fff'
MAJOR_GRID_COLOR = '#bbb'
MINOR_GRID_COLOR = '#e8e8e8'  # '#ddd'

TRACE_OPACITY = 0.8
TRACE_LINE_WIDTH = 1.5
MARKER_OPACITY = 0.8  # 75
MARKER_STYLE = dict(size=5, opacity=MARKER_OPACITY, line=None, symbol='circle')
AGE_FILL_COLORS = ['rgba(203,37,38,0.1)',  # q_loss_sei_total
                   'rgba(242,121,13,0.1)',  # q_loss_cyclic_total
                   'rgba(22,180,197)',  # q_loss_cyclic_low_total
                   'rgba(29,113,171, 0.1)']  # q_loss_plating_total

COLOR_BLACK = 'rgb(0,0,0)'
COLOR_BLUE = 'rgb(29,113,171)'
COLOR_CYAN = 'rgb(22,180,197)'
COLOR_ORANGE = 'rgb(242,121,13)'
COLOR_RED = 'rgb(203,37,38)'
COLOR_GRAY = 'rgb(127,127,127)'
COLOR_LIGHT_GRAY = 'rgb(200,200,200)'

COLOR_RED_DARK = 'rgb(196,22,42)'
COLOR_ORANGE_DARK = 'rgb(250,100,0)'
COLOR_YELLOW_DARK = 'rgb(224,180,0)'
COLOR_GREEN_DARK = 'rgb(55,135,45)'
COLOR_BLUE_DARK = 'rgb(31,96,196)'
COLOR_PURPLE_DARK = 'rgb(143,59,184)'

COLOR_PINK_INTENSE = 'rgb(255,79,185)'
COLOR_GREEN_INTENSE_LIGHT = 'rgb(0,242,0)'
COLOR_GREEN_INTENSE_DARK = 'rgb(0,160,0)'
COLOR_PURPLE_INTENSE_LIGHT = 'rgb(218,138,255)'
COLOR_PURPLE_INTENSE_DARK = 'rgb(174,0,255)'

COLOR_RED_LIGHT = 'rgb(255,115,131)'
COLOR_ORANGE_LIGHT = 'rgb(255,179,87)'
COLOR_YELLOW_LIGHT = 'rgb(255,238,0)'
COLOR_GREEN_LIGHT = 'rgb(150,217,141)'
COLOR_BLUE_LIGHT = 'rgb(138,184,255)'
COLOR_PURPLE_LIGHT = 'rgb(202,149,229)'

COLOR_ORANGE_2 = 'rgb(255,152,48)'
COLOR_ORANGE_LIGHT_2 = 'rgb(255,203,125)'
COLOR_RED_LIGHT_2 = 'rgb(255,166,176)'
COLOR_GREEN_LIGHT_2 = 'rgb(115,191,105)'
COLOR_BLUE_LIGHT_2 = 'rgb(87,148,242)'
COLOR_YELLOW = 'rgb(250,222,42)'
COLOR_PINK = 'rgb(227,119,194)'
COLOR_BROWN = 'rgb(127,51,0)'

COLOR_TRANSPARENT = 'rgba(0,0,0,0)'

COLOR_V_CELL = COLOR_BLUE_DARK
# COLOR_OCV_CELL = COLOR_BLUE_LIGHT
COLOR_I_CELL = COLOR_ORANGE_2
COLOR_P_GRID = COLOR_PINK
COLOR_P_CELL = COLOR_RED_DARK
COLOR_T_CELL = COLOR_GREEN_LIGHT_2
COLOR_SOC_CELL = COLOR_YELLOW
# COLOR_DQ_CELL = COLOR_PURPLE_LIGHT
# COLOR_DE_CELL = COLOR_PURPLE_DARK
# COLOR_EIS = COLOR_PINK
COLOR_SOH_CAP = COLOR_PURPLE_DARK
# COLOR_COULOMB_EFFICIENCY = COLOR_ORANGE_LIGHT_2
# COLOR_ENERGY_EFFICIENCY = COLOR_RED_LIGHT_2
COLOR_EMISSIONS = COLOR_BROWN
COLOR_PRICE = COLOR_BLUE_LIGHT
COLOR_FREQUENCY = COLOR_GRAY

FILL_ALPHA = 0.65
COLOR_BIOMASS = 'rgba(69,171,69,%.2f)' % FILL_ALPHA
COLOR_HYDRO = 'rgba(56,134,187,%.2f)' % FILL_ALPHA
COLOR_WIND_OFFSHORE = 'rgba(160,121,197,%.2f)' % FILL_ALPHA
COLOR_WIND_ONSHORE = 'rgba(51,197,212,%.2f)' % FILL_ALPHA
COLOR_PV = 'rgba(254,142,43,%.2f)' % FILL_ALPHA
COLOR_RESIDUAL = COLOR_RED_DARK  # line  # 'rgb(139,139,139,%.2f)' % FILL_ALPHA
COLOR_LOAD = COLOR_BLACK  # line
COLOR_LOAD_PROFILE = COLOR_BLUE
# COLOR_V2G = 'rgba(255,115,131,%.2f)' % FILL_ALPHA  # COLOR_RED_LIGHT with alpha
# COLOR_G2V = 'rgba(150,217,141,%.2f)' % FILL_ALPHA  # COLOR_GREEN_LIGHT with alpha
COLOR_G2V = 'rgba(237,125,49,%.2f)' % FILL_ALPHA  # orange with alpha
COLOR_V2G = 'rgba(112,173,71,%.2f)' % FILL_ALPHA  # green with alpha

# create custom theme from default plotly theme
pio.templates["custom_theme"] = pio.templates["plotly"]
pio.templates["custom_theme"]['layout']['paper_bgcolor'] = BG_COLOR
pio.templates["custom_theme"]['layout']['plot_bgcolor'] = BG_COLOR
pio.templates["custom_theme"]['layout']['hoverlabel']['namelength'] = -1
pio.templates['custom_theme']['layout']['title']['font']['size'] = TITLE_FONT_SIZE
pio.templates['custom_theme']['layout']['title']['font']['color'] = 'black'
pio.templates['custom_theme']['layout']['xaxis']['gridcolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['gridcolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['zerolinecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['xaxis']['linecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['linecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['zerolinecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['xaxis']['title']['standoff'] = 8
pio.templates['custom_theme']['layout']['yaxis']['title']['standoff'] = 8
pio.templates['custom_theme']['layout']['xaxis']['title']['font']['size'] = AXIS_FONT_SIZE
pio.templates['custom_theme']['layout']['yaxis']['title']['font']['size'] = AXIS_FONT_SIZE
pio.templates['custom_theme']['layout']['xaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
pio.templates['custom_theme']['layout']['yaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
pio.templates['custom_theme']['layout']['annotationdefaults']['font']['size'] = SUBPLOT_TITLE_FONT_SIZE


# generate (one) empty figure template
def generate_base_figure(n_rows, n_cols, plot_title, subplot_titles, y_axis_titles_row_arr, use_resampler=False,
                         plot_title_details=None, x_lim_arr=None, y_lim_arr=None, x_tick_side="top", abs_height=None,
                         x_axis_title=None):
    subplot_h_spacing = (SUBPLOT_H_SPACING_REL / n_cols)
    subplot_v_spacing = (SUBPLOT_V_SPACING_REL / max(n_rows, 5))  # avoid too small vertical spacing
    if abs_height is None:
        plot_height = HEIGHT_PER_ROW * n_rows + SUBPLOT_TOP_MARGIN_WITH_DETAILS
    else:
        plot_height = abs_height + SUBPLOT_TOP_MARGIN_WITH_DETAILS
    plot_title_y_pos = 1.0 - PLOT_TITLE_Y_POS_REL / plot_height
    if use_resampler:
        fig = FigureResampler(make_subplots(
            rows=n_rows, cols=n_cols, shared_xaxes="all",
            horizontal_spacing=subplot_h_spacing, vertical_spacing=subplot_v_spacing, subplot_titles=subplot_titles))
    else:
        fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes="all", horizontal_spacing=subplot_h_spacing,
                            vertical_spacing=subplot_v_spacing, subplot_titles=subplot_titles)

    fig.update_xaxes(ticks="outside",
                     tickcolor=MAJOR_GRID_COLOR,
                     ticklen=10,
                     minor=dict(griddash='dot',
                                gridcolor=MINOR_GRID_COLOR,
                                ticklen=4),
                     showticklabels=False
                     )

    for i_col in range(n_cols):
        fig.update_xaxes(showticklabels=True, side=x_tick_side, row=1, col=(i_col + 1))
        fig.update_xaxes(showticklabels=True, row=n_rows, col=(i_col + 1))
    fig.update_yaxes(ticks="outside",
                     tickcolor=MAJOR_GRID_COLOR,
                     ticklen=10,
                     showticklabels=True,
                     minor=dict(griddash='dot',
                                gridcolor=MINOR_GRID_COLOR,
                                ticklen=4),
                     )
    if x_lim_arr is not None:
        for i_row in range(n_rows):
            x_lim = x_lim_arr[i_row]
            for i_col in range(n_cols):
                fig.update_xaxes(range=x_lim, row=(i_row + 1), col=(i_col + 1))
    if y_lim_arr is not None:
        for i_row in range(n_rows):
            y_lim = y_lim_arr[i_row]
            for i_col in range(n_cols):
                fig.update_yaxes(range=y_lim, row=(i_row + 1), col=(i_col + 1))
    if x_axis_title is None:
        x_axis_title = X_AXIS_TITLE
    for i_col in range(n_cols):
        fig.update_xaxes(title_text=x_axis_title, row=n_rows, col=(i_col + 1))

    for i_row in range(n_rows):
        if type(y_axis_titles_row_arr) is str:
            title_text = y_axis_titles_row_arr
        else:
            title_text = y_axis_titles_row_arr[i_row]
        fig.update_yaxes(title_text=title_text, row=(i_row + 1), col=1)

    if plot_title_details is None:
        subplot_top_margin = SUBPLOT_TOP_MARGIN
        plot_title_use = PLOT_TITLE_RE % (plot_title, "")
    else:
        subplot_top_margin = SUBPLOT_TOP_MARGIN_WITH_DETAILS
        plot_title_use = PLOT_TITLE_RE % (plot_title, plot_title_details)

    fig.update_layout(title={'text': plot_title_use,
                             'y': plot_title_y_pos,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template=FIGURE_TEMPLATE,
                      autosize=True,
                      height=plot_height,
                      width=PLOT_WIDTH,
                      legend=dict(x=0, y=0),
                      margin=dict(l=SUBPLOT_LR_MARGIN, r=SUBPLOT_LR_MARGIN,
                                  t=subplot_top_margin, b=SUBPLOT_BOT_MARGIN,
                                  pad=SUBPLOT_PADDING)
                      )
    # fig.update_layout(hovermode='x unified')
    # fig.update_layout(hovermode='x')
    fig.update_xaxes(showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig


# add trace/line to figure
def add_result_trace(fig, i_row, i_col, x, y, color, show_marker, show_line, text_data=None, timezone=TIMEZONE_DEFAULT,
                     use_hover=True, use_resampler=False, horizontal_lines=False, line_width=TRACE_LINE_WIDTH,
                     line_dash="solid", show_legend=False, name=None, zorder=None, legend_name=None):
    if ((type(x) is type(pd.Index([], dtype=float)))
            or (type(x) is np.float64) or (type(x) is np.float32) or (type(x) is float) or (type(x) is int)):
        x_plot = pd.to_datetime(x, unit="s", origin='unix', utc=True).tz_convert(timezone)
    else:
        x_plot = x
    # data_range = range(x_plot.shape[0])
    # text_data = pd.Series("", index=data_range)
    # for i in data_range:
    #     text_data.loc[i] = PLOT_TEXT % x_plot[i].strftime('%Y-%m-%d  %H:%M:%S')
    # text_data = x_plot.strftime('%Y-%m-%d  %H:%M:%S')

    this_marker_style = MARKER_STYLE.copy()
    this_marker_style["color"] = color

    mode = None
    if show_marker and show_line:
        mode = "markers+lines"
    elif show_marker:
        mode = "markers"
    elif show_line:
        mode = "lines"
    line_shape = None
    if horizontal_lines:
        line_shape = 'hv'  # https://plotly.com/python/line-charts/#interpolation-with-line-plots
    hover_template = None
    if use_hover:
        hover_template = PLOT_HOVER_TEMPLATE
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=show_legend, name=name, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=line_width, shape=line_shape, dash=line_dash,),
                       zorder=zorder,  # only in Plotly 5.21.0+
                       legend=legend_name,
                       ),
            hf_x=x_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_plot, y=y, showlegend=show_legend, name=name, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=line_width, shape=line_shape, dash=line_dash),
                       zorder=zorder,  # only in Plotly 5.21.0+
                       legend=legend_name,
                       ),
            row=(i_row + 1), col=(i_col + 1))


# add electricity generation and demand data lines/filled areas to the result plot
def add_generation_and_demand_trace(fig, i_row, i_col, gen_dem_df, show_marker, show_line, text_data=None,
                                    timezone=TIMEZONE_DEFAULT, use_hover=True, use_resampler=False):
    # general settings
    mode = None
    if show_marker and show_line:
        mode = "markers+lines"
    elif show_marker:
        mode = "markers"
    elif show_line:
        mode = "lines"
    hover_template = None
    if use_hover:
        hover_template = PLOT_HOVER_TEMPLATE

    # x data (time)
    x = gen_dem_df.index
    if ((type(x) is type(pd.Index([], dtype=float)))
            or (type(x) is np.float64) or (type(x) is np.float32) or (type(x) is float) or (type(x) is int)):
        x_plot = pd.to_datetime(x, unit="s", origin='unix', utc=True).tz_convert(timezone)
    else:
        x_plot = x

    # plot definitions
    ren_plot_y_cols = [idh.GEN_BIOMASS, idh.GEN_HYDRO, idh.GEN_WIND_OFFSHORE, idh.GEN_WIND_ONSHORE, idh.GEN_PV]
    ren_plot_colors = [COLOR_BIOMASS, COLOR_HYDRO, COLOR_WIND_OFFSHORE, COLOR_WIND_ONSHORE, COLOR_PV]

    # plot REN (renewable energy generation) stack:
    for i in range(len(ren_plot_y_cols)):
        color = ren_plot_colors[i]
        this_marker_style = MARKER_STYLE.copy()
        this_marker_style["color"] = color
        y = gen_dem_df[ren_plot_y_cols[i]].values
        if use_resampler:
            fig.add_trace(
                go.Scatter(showlegend=False, mode=mode, marker=this_marker_style,
                           text=text_data, hovertemplate=hover_template,
                           line=dict(color=color, width=TRACE_LINE_WIDTH), stackgroup='REN'),
                hf_x=x_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
        else:
            fig.add_trace(
                go.Scatter(x=x_plot, y=y, showlegend=False, mode=mode, marker=this_marker_style,
                           text=text_data, hovertemplate=hover_template,
                           line=dict(color=color, width=TRACE_LINE_WIDTH), stackgroup='REN'),
                row=(i_row + 1), col=(i_col + 1))

    # plot load
    color = COLOR_LOAD
    this_marker_style = MARKER_STYLE.copy()
    this_marker_style["color"] = color
    y = gen_dem_df[idh.DEMAND].values
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=TRACE_LINE_WIDTH)),
            hf_x=x_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_plot, y=y, showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=TRACE_LINE_WIDTH)),
            row=(i_row + 1), col=(i_col + 1))

    # plot residual load
    color = COLOR_RESIDUAL
    this_marker_style = MARKER_STYLE.copy()
    this_marker_style["color"] = color
    y = gen_dem_df[idh.RESIDUAL_LOAD].values
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=TRACE_LINE_WIDTH)),
            hf_x=x_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_plot, y=y, showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=TRACE_LINE_WIDTH)),
            row=(i_row + 1), col=(i_col + 1))


# add solar (PV) and local (household) load profile traces/lines to the result plot
def add_pv_and_load_profile_trace(fig, i_row, i_col, pv_df, load_profile_df, grid_df, show_marker, show_line,
                                  text_data=None, timezone=TIMEZONE_DEFAULT, use_hover=True, use_resampler=False):
    # general settings
    mode = None
    if show_marker and show_line:
        mode = "markers+lines"
    elif show_marker:
        mode = "markers"
    elif show_line:
        mode = "lines"
    hover_template = None
    if use_hover:
        hover_template = PLOT_HOVER_TEMPLATE

    # x data (time)
    x_pv = pv_df.index
    if ((type(x_pv) is type(pd.Index([], dtype=float)))
            or (type(x_pv) is np.float64) or (type(x_pv) is np.float32)
            or (type(x_pv) is float) or (type(x_pv) is int)
            or (type(x_pv[0]) is np.float64) or (type(x_pv[0]) is np.float32)
            or (type(x_pv[0]) is float) or (type(x_pv[0]) is int)):
        x_pv_plot = pd.to_datetime(x_pv, unit="s", origin='unix', utc=True).tz_convert(timezone)
    else:
        x_pv_plot = x_pv

    x_load_profile = load_profile_df.index
    if ((type(x_load_profile) is type(pd.Index([], dtype=float)))
            or (type(x_load_profile) is np.float64) or (type(x_load_profile) is np.float32)
            or (type(x_load_profile) is float) or (type(x_load_profile) is int)
            or (type(x_load_profile[0]) is np.float64) or (type(x_load_profile[0]) is np.float32)
            or (type(x_load_profile[0]) is float) or (type(x_load_profile[0]) is int)):
        x_load_profile_plot = pd.to_datetime(x_load_profile, unit="s", origin='unix', utc=True).tz_convert(timezone)
    else:
        x_load_profile_plot = x_load_profile

    x_grid = grid_df.index
    if ((type(x_grid) is type(pd.Index([], dtype=float)))
            or (type(x_grid) is np.float64) or (type(x_grid) is np.float32)
            or (type(x_grid) is float) or (type(x_grid) is int)
            or (type(x_grid[0]) is np.float64) or (type(x_grid[0]) is np.float32)
            or (type(x_grid[0]) is float) or (type(x_grid[0]) is int)):
        x_grid_plot = pd.to_datetime(x_grid, unit="s", origin='unix', utc=True).tz_convert(timezone)
    else:
        x_grid_plot = x_grid

    # plot load profile (transparently) -> load_profile_df (stack: G2V, no line)
    y = load_profile_df.copy()
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode="lines", fillcolor=COLOR_TRANSPARENT,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=COLOR_TRANSPARENT, width=TRACE_LINE_WIDTH), stackgroup='G2V'),
            hf_x=x_load_profile_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_load_profile_plot, y=y, showlegend=False, mode="lines", fillcolor=COLOR_TRANSPARENT,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=COLOR_TRANSPARENT, width=TRACE_LINE_WIDTH), stackgroup='G2V'),
            row=(i_row + 1), col=(i_col + 1))

    # plot G2V -> grid_df[grid_df > 0] (stack: G2V, no line, filltonexty in red)
    y = grid_df.copy()
    y[y < 0] = 0.0
    color = COLOR_G2V
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode="lines",
                       text=text_data, hovertemplate=hover_template, fillcolor=color,  # fill='tonexty',
                       line=dict(color=COLOR_TRANSPARENT, width=TRACE_LINE_WIDTH), stackgroup='G2V'),
            hf_x=x_grid_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_grid_plot, y=y, showlegend=False, mode="lines",
                       text=text_data, hovertemplate=hover_template, fillcolor=color,  # fill='tonexty',
                       line=dict(color=COLOR_TRANSPARENT, width=TRACE_LINE_WIDTH), stackgroup='G2V'),
            row=(i_row + 1), col=(i_col + 1))

    # plot solar -> pv_df (stack: none, yellow line)
    y = pv_df.copy()
    color = COLOR_PV
    this_marker_style = MARKER_STYLE.copy()
    this_marker_style["color"] = color
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template, fillcolor=COLOR_TRANSPARENT,
                       line=dict(color=color, width=TRACE_LINE_WIDTH), stackgroup='V2G'),
            hf_x=x_pv_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_pv_plot, y=y, showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template, fillcolor=COLOR_TRANSPARENT,
                       line=dict(color=color, width=TRACE_LINE_WIDTH), stackgroup='V2G'),
            row=(i_row + 1), col=(i_col + 1))

    # plot V2G -> grid_df[grid_df < 0] (stack: PV2, no line, filltonexty in green)
    y = grid_df.copy()
    y[y > 0] = 0.0
    y = -y
    color = COLOR_V2G
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode="lines",
                       text=text_data, hovertemplate=hover_template, fillcolor=color,  # fill='tonexty',
                       line=dict(color=COLOR_TRANSPARENT, width=TRACE_LINE_WIDTH), stackgroup='V2G'),
            hf_x=x_grid_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_grid_plot, y=y, showlegend=False, mode="lines",
                       text=text_data, hovertemplate=hover_template, fillcolor=color,  # fill='tonexty',
                       line=dict(color=COLOR_TRANSPARENT, width=TRACE_LINE_WIDTH), stackgroup='V2G'),
            row=(i_row + 1), col=(i_col + 1))

    # plot load profile again, so it is on top -> load_profile_df (no stack, blue line)
    y = load_profile_df.copy()
    color = COLOR_LOAD_PROFILE
    this_marker_style = MARKER_STYLE.copy()
    this_marker_style["color"] = color
    if use_resampler:
        fig.add_trace(
            go.Scatter(showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=TRACE_LINE_WIDTH)),
            hf_x=x_load_profile_plot, hf_y=y, row=(i_row + 1), col=(i_col + 1))
    else:
        fig.add_trace(
            go.Scatter(x=x_load_profile_plot, y=y, showlegend=False, mode=mode, marker=this_marker_style,
                       text=text_data, hovertemplate=hover_template,
                       line=dict(color=color, width=TRACE_LINE_WIDTH)),
            row=(i_row + 1), col=(i_col + 1))


# export a result plot to html/image and/or open it in the browser
def export_figure(fig, export_html, export_image_format, export_path, export_filename_base, open_in_browser,
                  use_resampler=False, image_export_engine='kaleido', append_date=True):
    if append_date:
        run_timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        export_filename_base = export_filename_base + "_" + run_timestring

    if export_html or ((export_image_format is not None) and (export_image_format != "")):
        if not os.path.exists(export_path):
            os.mkdir(export_path)

        if export_html:
            filename = export_path + export_filename_base + ".html"
            print("Saving result plot as html ...\n    %s" % filename)
            if os.path.isfile(filename):  # file already exists
                file_exists = True
            else:
                file_exists = False
            fig.write_html(filename, auto_open=open_in_browser)
            if file_exists:
                os.utime(filename)  # update "last modified date" -> can be pretty confusing otherwise

        if (export_image_format is not None) and (export_image_format != ""):
            filename = export_path + export_filename_base + "." + export_image_format
            print("Saving result plot as image (.%s) ...\n    %s" % (export_image_format, filename))
            if os.path.isfile(filename):  # file already exists
                file_exists = True
            else:
                file_exists = False
            plot_scale_factor = 1.0
            if (export_image_format == "jpg") or (export_image_format == "png"):
                plot_scale_factor = 3.0
            fig.write_image(filename, format=export_image_format, engine=image_export_engine,
                            width=fig['layout']['width'], height=fig['layout']['height'], scale=plot_scale_factor)
            if file_exists:
                os.utime(filename)  # update "last modified date" -> can be pretty confusing otherwise

    if open_in_browser and (not export_html):
        print("Opening result plot in browser ...")
        if use_resampler:
            fig.show_dash(mode='inline')
        else:
            fig.show(validate=False)
