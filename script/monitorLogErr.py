import dash
from dash import dcc, html, Input, Output, State, Dash
import pandas as pd
import plotly.express as px
import math

app = Dash(__name__)

# Path to the file being monitored
file_path = "../data/out/NACA0012/NACA0012-AOA15_.log"


# Path to the CSV file
csv_file_path = file_path

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(id="line-plot", style={"width": "100%", "height": "100vh"}),
            ],
            style={"width": "75%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                html.Label("Select a column:", style={"font-weight": "bold"}),
                dcc.Dropdown(id="column-dropdown", placeholder="Select a column", value="res0"),
            ],
            style={
                "width": "20%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-left": "5px",
            },
        ),
        dcc.Interval(
            id="update-interval", interval=2000, n_intervals=0
        ),  # Refresh every 2 seconds
    ]
)


# Callback to update dropdown options
@app.callback(
    Output("column-dropdown", "options"),
    Input("update-interval", "n_intervals"),
)
def update_dropdown_options(_):
    try:
        df = pd.read_csv(csv_file_path)
        options = [{"label": col, "value": col} for col in df.columns]
        return options
    except Exception as e:
        return []  # Handle error, e.g., file not found or empty file


# Callback to update graph
@app.callback(
    Output("line-plot", "figure"),
    [Input("column-dropdown", "value"), Input("update-interval", "n_intervals")],
    State("line-plot", "relayoutData"),
)
def update_graph(selected_column, _, relayout_data):
    try:
        df = pd.read_csv(csv_file_path)
        if selected_column and selected_column in df.columns:
            fig = px.line(
                df, y=selected_column, title=f"Plot of {selected_column}", log_y=True
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
                vmax= df[selected_column].max()
                vmin = df[selected_column].min()
                range_value = (vmax - vmin) / abs(abs(vmax) + abs(vmin) + 1e-200)

            exponent = math.floor(math.log10(range_value)) if range_value > 0 else 0

            digits = max(0, 1 - exponent)
            # digits = 1
            fig.update_yaxes(tickformat=f".{digits:d}e")

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
        else:
            fig = px.line(title="Select a column to plot")
        return fig
    except Exception as e:
        return px.line(title="Error loading data")


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
