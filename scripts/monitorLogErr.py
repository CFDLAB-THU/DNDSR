import dash
from dash import Dash
import argparse


import logErrMonitor.monitorApp as monitorApp

# Run the app
if __name__ == "__main__":
    app = Dash(__name__)

    parser = argparse.ArgumentParser(
        description="view residual history for a steady computation"
    )
    parser.add_argument("-p", "--prefix", default="data/out/", type=str)
    parser.add_argument("--port", default=8050, type=int)

    args = parser.parse_args()
    print(args)

    #######

    # Initialize Dash app
    app = dash.Dash(__name__)

    # Layout
    app.layout = monitorApp.get_app_layout()
    for register in monitorApp.register_list:
        register(app, args)

    app.run(
        debug=True,
        port=args.port,
    )
