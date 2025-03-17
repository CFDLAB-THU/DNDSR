from cProfile import label
import dash
from dash import dcc, html, Input, Output, State, Dash
from matplotlib.pyplot import xlabel
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import os, argparse, re


right_label_style = {
    "font-weight": "bold",
    "vertical-align": "bottom",
    "align": "center",
    "display": "flex",
    "padding-bottom": "1em",
    "padding-top": "1em",
}


def get_app_layout():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id="line-plot", style={"width": "100%", "height": "100vh"}
                    ),
                ],
                style={
                    "width": "75%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Div(
                        style={
                            "width": "100%",
                            "height": "10vh",
                            "display": "block",
                            "vertical-align": "top",
                            "horizontal-align": "right",
                        },
                    ),
                    html.Label(
                        "Select a file:",
                        style=right_label_style,
                    ),
                    dcc.Dropdown(id="file-dropdown", placeholder="Select a log file"),
                    html.Label(
                        "Select a column for X-Axis:",
                        style=right_label_style,
                    ),
                    dcc.Dropdown(
                        id="column-x-dropdown",
                        placeholder="Select a column",
                        value="iterAll",
                    ),
                    html.Label(
                        "Select a column for Y-Axis:",
                        style=right_label_style,
                    ),
                    dcc.Dropdown(
                        id="column-dropdown",
                        placeholder="Select a column",
                        value="res0",
                    ),
                    dcc.Checklist(
                        ["Log Y"],
                        ["Log Y"],
                        id="checklist_0",
                        style={
                            "display": "flex",
                            "padding-top": "1em",
                            "padding-bottom": "1em",
                        },
                    ),
                    html.Label(
                        "Select a file [1]:",
                        style=right_label_style,
                    ),
                    dcc.Dropdown(id="file-dropdown-1", placeholder="Select a log file"),
                ],
                style={
                    "width": "20%",
                    "height": "100%",
                    "display": "inline-block",
                    "vertical-align": "center",
                    "margin-left": "5px",
                    "margin": "10px",
                },
            ),
            dcc.Interval(
                id="update-interval", interval=2000, n_intervals=0
            ),  # Refresh every 2 seconds
        ]
    )


register_list = []


def register_update_dropdown_options(app: Dash, args: argparse.Namespace):
    # Callback to update dropdown options
    @app.callback(
        Output("column-dropdown", "options"),
        [
            Input("file-dropdown", "value"),
            # Input("update-interval", "n_intervals"), // no need always update
        ],
    )
    def update_dropdown_options(file_dropdown_value):
        try:
            df = pd.read_csv(file_dropdown_value)
            options = [{"label": col, "value": col} for col in df.columns]
            print(f"New Column Options:\n {df.columns}")
            return options
        except Exception as e:
            return []  # Handle error, e.g., file not found or empty file


register_list.append(register_update_dropdown_options)


def register_update_dropdown_x_options(app: Dash, args: argparse.Namespace):
    @app.callback(
        Output("column-x-dropdown", "options"),
        [
            Input("column-dropdown", "options"),
        ],
    )
    def update_dropdown_x_options(column_dropdown_value):
        return column_dropdown_value


register_list.append(register_update_dropdown_x_options)


def register_update_file_dropdown_options(app: Dash, args: argparse.Namespace):
    # Callback to update dropdown options
    @app.callback(
        Output("file-dropdown", "options"),
        Input("update-interval", "n_intervals"),
    )
    def update_dropdown_options(update_n_intervals):
        try:
            names = os.listdir(args.prefix)
            names = filter(lambda x: re.match(r".*\.log", x), names)
            fileDirs = {}

            for name in names:
                namefull = os.path.join(args.prefix, name)
                stat = os.stat(namefull)
                fileDirs[namefull] = stat.st_mtime  # sort with mtime or ctime
                # print(stat)

            fileDirsSorted = sorted(
                fileDirs.items(), key=lambda x: x[1], reverse=True
            )  # latest first
            # print(fileDirsSorted)
            # print(update_n_intervals)

            options = [
                {"label": os.path.split(fileDir[0])[1], "value": fileDir[0]}
                for fileDir in fileDirsSorted
            ]
            return options
        except Exception as e:
            return []  # Handle error, e.g., file not found or empty file


register_list.append(register_update_file_dropdown_options)


def register_update_file_dropdown_1_options(app: Dash, args: argparse.Namespace):
    @app.callback(
        Output("file-dropdown-1", "options"),
        [
            Input("file-dropdown", "options"),
        ],
    )
    def update_file_dropdown_1_options(opts):
        return opts


register_list.append(register_update_file_dropdown_1_options)


def register_update_graph(app: Dash, args: argparse.Namespace):
    # Callback to update graph
    @app.callback(
        Output("line-plot", "figure"),
        [
            Input("file-dropdown", "value"),
            Input("file-dropdown-1", "value"),
            Input("column-dropdown", "value"),
            Input("column-x-dropdown", "value"),
            Input("update-interval", "n_intervals"),
            Input("checklist_0", "value"),
        ],
        State("line-plot", "relayoutData"),
    )
    def update_graph(
        selected_file,
        selected_file_1,
        selected_column,
        selected_column_x,
        _,
        checklist_0,
        relayout_data,
    ):
        fig = go.Figure()
        try:
            if not selected_file:
                return px.line(title="Main log not selected")
            df = pd.read_csv(selected_file)
            # fig = px.line(
            #     title=f"Plot of {selected_column}",
            #     log_y=("Log Y" in checklist_0),
            # )
            if selected_column is None:
                selected_column = "res0"
            fig.update_layout(
                title=f"Plot of {selected_column}",
                margin=dict(l=50, r=100, t=50, b=50),
            )
            if selected_column and selected_column in df.columns:
                x = None
                if "iterAll" in df.columns:
                    x = "iterAll"
                if selected_column_x and selected_column_x in df.columns:
                    x = selected_column_x
                fig.add_scatter(
                    x=df[x],
                    y=df[selected_column],
                    name=selected_file,
                    mode="lines",
                )
            df1 = None
            if selected_file_1:
                try:
                    df1 = pd.read_csv(selected_file_1)
                except:
                    df1 = None
            if df1 is not None and selected_column and selected_column in df1.columns:
                x = None
                if "iterAll" in df1.columns:
                    x = "iterAll"
                if selected_column_x and selected_column_x in df1.columns:
                    x = selected_column_x
                fig.add_scatter(
                    x=df1[x],
                    y=df1[selected_column],
                    name=selected_file_1,
                    mode="lines",
                )
                fig.update_legends(
                    dict(
                        x=0.5,  # Position the legend inside the plot
                        y=0.95,
                        traceorder="normal",
                        orientation="v",  # Vertical layout
                        xanchor="center",
                        bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background
                        bordercolor="black",  # Border color
                        borderwidth=1,
                    )
                )

            if (
                relayout_data
                and "yaxis.range[0]" in relayout_data
                and "yaxis.range[1]" in relayout_data
            ):
                range_value = abs(
                    relayout_data["yaxis.range[0]"] - relayout_data["yaxis.range[1]"]
                ) / (
                    abs(relayout_data["yaxis.range[0]"])
                    + abs(relayout_data["yaxis.range[1]"])
                    + 1e-200
                )
            else:
                vmax = df[selected_column].max()
                vmin = df[selected_column].min()
                range_value = (vmax - vmin) / abs(abs(vmax) + abs(vmin) + 1e-200)

            exponent = math.floor(math.log10(range_value)) if range_value > 0 else 0

            digits = max(0, 1 - exponent)
            # digits = 1
            fig.update_yaxes(
                tickformat=f".{digits:d}e",
                type="log" if ("Log Y" in checklist_0) else "linear",
            )

            fig.update_traces(hovertemplate="x: %{x}<br>y: %{y:.5e}")
            # Restore axes ranges if available
            if (
                relayout_data
                and "xaxis.range[0]" in relayout_data
                and "xaxis.range[1]" in relayout_data
            ):
                fig.update_xaxes(
                    range=[
                        relayout_data["xaxis.range[0]"],
                        relayout_data["xaxis.range[1]"],
                    ]
                )

            if (
                relayout_data
                and "yaxis.range[0]" in relayout_data
                and "yaxis.range[1]" in relayout_data
            ):
                fig.update_yaxes(
                    range=[
                        relayout_data["yaxis.range[0]"],
                        relayout_data["yaxis.range[1]"],
                    ]
                )

            return fig
        except Exception as e:
            print(e)
            return px.line(title="Error loading data")


register_list.append(register_update_graph)
