import dash
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table

import dash.dash_table as dt
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, lognorm, expon, weibull_min, gamma, uniform
import math
import base64
import io
from datetime import datetime
import os
from plotly.subplots import make_subplots

# Get the directory where your script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create the full path to the image
image_path = os.path.join(script_dir, 'Minitab_Image1.jpg')

# Function to encode the image
def encode_image(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f'data:image/jpeg;base64,{encoded_string}'
    except FileNotFoundError:
        print(f"Image file not found at: {image_path}")
        return None

# Encode the Minitab logo
MINITAB_LOGO = encode_image(image_path)

# Sample data
data = {
    "Exclude?": [""] * 60,  # Changed from "Include?" to "Exclude?" with blank default
    "Shot #": range(1, 61),
    "Distance": [
        88.5, 81.5, 85, 84, 79, 85, 84, 84, 83, 84.5,
        80, 78, 82, 82, 82, 72, 77, 86, 79, 82,
        72, 79, 79.5, 75, 74, 93, 82, 78, 83.5, 86,
        82, 89, 86, 84, 83.5, 85, 86, 84, 83, 83.5,
        79, 81, 85, 85, 86, 70, 76, 76, 79, 82,
        82, 80, 84, 84, 84, 87, 82, 81, 79, 89
    ]
}
df = pd.DataFrame(data)

# Calculate default number of bins using Sturges' rule
def calculate_default_bins(data_length):
    return min(max(5, int(1 + math.log2(data_length))), 30)

default_bins = calculate_default_bins(len(df))

# Create the Dash app
app = Dash(__name__)
app._dev_tools.ui = False
app._dev_tools.hot_reload = False
app._dev_tools.props_check = False
app._dev_tools.serve_dev_bundles = False

# Layout
app.layout = html.Div(style={"display": "flex", "height": "100vh"}, children=[
    # Sidebar
    html.Div(style={
        "width": "250px",
        "background-color": "#f5f5f5",
        "padding": "20px",
        "border-right": "1px solid #ddd"
    }, children=[
        # Minitab Logo
        html.Img(
            src=MINITAB_LOGO,
            style={
                "width": "100%",
                "margin-bottom": "20px"
            }
        ),
        
        # File Upload
        dcc.Upload(
            id='upload-data',
            children=html.Button(
                "Import Data",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "background-color": "#6baed6",
                    "color": "white",
                    "border": "none",
                    "border-radius": "4px",
                    "cursor": "pointer",
                    "font-family": "Segoe UI"
                }
            ),
            multiple=False
        ),
        
        # Save Data Button (remove the format dropdown)
        html.Div(style={"margin-top": "10px"}, children=[
            html.Button(
                "Save Data",
                id="save-button",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "background-color": "#6baed6",
                    "color": "white",
                    "border": "none",
                    "border-radius": "4px",
                    "cursor": "pointer",
                    "font-family": "Segoe UI"
                }
            ),
        ]),
        
        # Specification Settings
        html.Div(style={"margin-top": "20px"}, children=[
            html.Label("Specification Type:", 
                      style={"font-family": "Segoe UI", "font-weight": "bold"}),
            dcc.Dropdown(
                id="spec-type",
                options=[
                    {"label": "Target is Better", "value": "target"},
                    {"label": "Higher is Better", "value": "higher"},
                    {"label": "Lower is Better", "value": "lower"}
                ],
                value="target",
                style={"margin-top": "5px"}
            ),
            
            # Specification inputs
            html.Div(
                id="spec-inputs",
                style={"margin-top": "10px", "display": "flex", "gap": "10px"},
                children=[
                    html.Div([
                        html.Label("LSL"),
                        dcc.Input(
                            id="lsl-input",
                            type="number",
                            style={"width": "60px"}
                        )
                    ]),
                    html.Div([
                        html.Label("Target"),
                        dcc.Input(
                            id="target-input",
                            type="number",
                            style={"width": "60px"}
                        )
                    ]),
                    html.Div([
                        html.Label("USL"),
                        dcc.Input(
                            id="usl-input",
                            type="number",
                            style={"width": "60px"}
                        )
                    ])
                ]
            )
        ]),
        
        # Error message div
        html.Div(id='upload-error', 
                 style={"color": "red", "margin-top": "5px", "font-size": "12px"}),
        
        # Download component for saving data
        dcc.Download(id="download-data"),
        
        # Column Selection Dropdown (hidden by default)
        html.Div(
            id='column-select-container',
            style={'display': 'none', 'margin-top': "20px"},
            children=[
                html.Label("Select Column to Plot:",
                          style={"font-family": "Segoe UI", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='column-select',
                    style={"margin-top": "5px"}
                )
            ]
        ),
        
        # Add Radio Buttons for Capability Distribution Selection
        html.Div(style={"margin-top": "20px"}, children=[
            html.H4("Capability Distribution", 
                    style={"font-family": "Segoe UI", "font-size": "18px"}),
            dcc.RadioItems(
                id="capability-distribution",
                options=[
                    {"label": "Normal", "value": "normal"},
                    {"label": "LogNormal", "value": "lognormal"},
                    {"label": "Exponential", "value": "exponential"},
                    {"label": "Weibull", "value": "weibull"},
                    {"label": "Gamma", "value": "gamma"},
                    {"label": "Uniform", "value": "uniform"}
                ],
                value="normal",  # Default to normal distribution
                style={"font-family": "Segoe UI"}
            )
        ]),

        # Existing Distribution Options for Histogram remain unchanged
        html.H4("Distribution Options", 
                style={"font-family": "Segoe UI", "font-size": "18px", "margin-top": "20px"}),
        dcc.Checklist(
            id="distribution-checklist",
            options=[
                {"label": "Normal", "value": "normal"},
                {"label": "LogNormal", "value": "lognormal"},
                {"label": "Exponential", "value": "exponential"},
                {"label": "Weibull", "value": "weibull"},
                {"label": "Gamma", "value": "gamma"},
                {"label": "Uniform", "value": "uniform"}
            ],
            value=["normal"],
            style={"font-family": "Segoe UI"}
        ),
        html.Div(style={"margin-top": "20px"}, children=[
            html.Label("Number of Bins:", 
                      style={"font-family": "Segoe UI", "font-weight": "bold"}),
            html.Div(style={"margin-top": "10px", "padding": "0 20px"},
                children=[
                    dcc.Slider(
                        id="nbins-slider",
                        min=5,
                        max=30,
                        step=1,
                        value=default_bins,
                        marks={i: str(i) for i in range(5, 31, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ]
            ),
        ]),
        # Store for selected bin
        dcc.Store(id="selected-bin", data=None),
        # Store for current data
        dcc.Store(id="current-data"),
        
        # Add Local Data Filter button after the existing column-select dropdown
        html.Button(
            "Local Data Filter",
            id="local-filter-button",
            style={
                "width": "100%",
                "padding": "10px",
                "background-color": "#6baed6",
                "color": "white",
                "border": "none",
                "border-radius": "4px",
                "cursor": "pointer",
                "font-family": "Segoe UI",
                "margin-top": "10px"
            }
        ),
        
        # Add confirmation dialog for the filter button
        dcc.ConfirmDialog(
            id='filter-dialog',
            message='Future capability to filter data by other variables in the data table.'
        ),
    ]),

    # Main content
    html.Div(style={"flex": "1", "padding": "20px"}, children=[
        # Top section with plots and tables
        html.Div(style={"display": "flex", "gap": "20px", "margin-bottom": "20px"}, children=[
            # Left side - Plots (60%)
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label="Process Analysis", 
                        style={'backgroundColor': '#6baed6', 'color': 'white'},
                        selected_style={'backgroundColor': 'white', 'color': 'black'},
                        children=[
                            # Histogram
                            html.Div([
                                html.H4("Histogram with Fitted Distributions", 
                                       style={"font-family": "Segoe UI", "font-size": "20px"}),
                                dcc.Graph(
                                    id="histogram",
                                    config={"displayModeBar": False},
                                    style={"height": "400px"}
                                )
                            ]),
                            
                            # Box Plot
                            html.Div([
                                html.H4("Box Plot with Outliers", 
                                       style={"font-family": "Segoe UI", "font-size": "20px", "margin-top": "20px"}),
                                dcc.Graph(
                                    id="box-plot",
                                    config={"displayModeBar": False},
                                    style={"height": "200px"}
                                )
                            ], style={
                                "border": "1px solid #ddd",
                                "padding": "15px",
                                "margin-top": "20px"
                            }),

                            # I-MR Chart
                            html.Div([
                                html.H4("Individuals and Moving Range Chart", 
                                       style={"font-family": "Segoe UI", "font-size": "20px", "margin-top": "20px"}),
                                dcc.Graph(
                                    id="imr-plot",
                                    config={"displayModeBar": False},
                                    style={"height": "400px"}
                                )
                            ], style={
                                "border": "1px solid #ddd",
                                "padding": "15px",
                                "margin-top": "20px"
                            })
                        ]),
                    dcc.Tab(label="Process Capability",
                        style={'backgroundColor': '#6baed6', 'color': 'white'},
                        selected_style={'backgroundColor': 'white', 'color': 'black'},
                        children=[
                            # First container for Metrics and Plot (side by side)
                            html.Div([
                                # Left side - Capability Metrics Table
                                html.Div([
                                    html.H4("Process Capability Metrics", 
                                           style={"font-family": "Segoe UI", "font-size": "28px"}),
                                    html.Div(id="capability-metrics-table",
                                           style={"font-family": "Segoe UI", "font-size": "14px", "margin-top": "20px"})
                                ], style={
                                    "padding": "15px",
                                    "flex": "1"
                                }),
                                
                                # Right side - Capability Plot
                                html.Div([
                                    html.H4("Capability Analysis Plot", 
                                           style={"font-family": "Segoe UI", "font-size": "28px"}),
                                    dcc.Graph(
                                        id="capability-plot",
                                        config={"displayModeBar": False},
                                        style={"height": "400px"}
                                    )
                                ], style={
                                    "padding": "15px",
                                    "flex": "1"
                                })
                            ], style={
                                "display": "flex",
                                "gap": "20px"
                            }),
                            
                            # Separate container for Interpretation section
                            html.Div([
                                html.H4("Interpretation and Recommended Actions",
                                        style={"font-family": "Segoe UI", 
                                               "margin-top": "30px", 
                                               "margin-bottom": "20px",
                                               "font-size": "24px"}),
                                
                                # Container for the two columns
                                html.Div(style={
                                    "display": "flex",
                                    "justify-content": "space-between",
                                    "gap": "20px"
                                }, children=[
                                    # Left column - Interpretations
                                    html.Div(style={
                                        "flex": "1",
                                        "padding": "20px",
                                        "background-color": "#f8f9fa",
                                        "border-radius": "5px",
                                        "border": "1px solid #dee2e6"
                                    }, children=[
                                        html.H5("Interpretations",
                                                style={"font-family": "Segoe UI", "margin-bottom": "15px"}),
                                        html.P("Shown here will be an interpretation of the statistical aspects of the data including its stability, spread, etc. This will provide the information needed to focus on in the Recommended Actions section.",
                                               style={"font-family": "Segoe UI", "color": "#666"})
                                    ]),
                                    
                                    # Right column - Recommended Actions
                                    html.Div(style={
                                        "flex": "1",
                                        "padding": "20px",
                                        "background-color": "#f8f9fa",
                                        "border-radius": "5px",
                                        "border": "1px solid #dee2e6"
                                    }, children=[
                                        html.H5("Recommended Actions",
                                                style={"font-family": "Segoe UI", "margin-bottom": "15px"}),
                                        html.P("Shown here will be recommended actions needed to be taken to achieve the required level of Process Capability. This will include actions such as reduce exceptional variation, recenter the process, reduce variability. It will also include the best methods and tools to achieve these outcomes.",
                                               style={"font-family": "Segoe UI", "color": "#666"})
                                    ])
                                ])
                            ])
                        ])
                ], style={"font-family": "Segoe UI", "font-size": "20px"})
            ], style={"flex": "60%", "border": "1px solid #ddd", "padding": "15px"}),

            # Right side - Tables and Data Grid (40%)
            html.Div(style={"flex": "40%"}, children=[
                # Tables side by side
                html.Div(style={
                    "display": "flex", 
                    "gap": "20px", 
                    "margin-bottom": "20px",
                    "border": "1px solid #ddd",
                    "padding": "15px"
                }, children=[
                    # Descriptive Statistics
                    html.Div([
                        html.H4("Descriptive Statistics", 
                               style={"font-family": "Segoe UI", "font-size": "20px"}),
                        html.Table(id="stats-table", 
                                 style={"font-family": "Segoe UI", "font-size": "14px"}),
                    ], style={"flex": "50%"}),

                    # Data Summary
                    html.Div([
                        html.H4("Data Summary", 
                               style={"font-family": "Segoe UI", "font-size": "20px"}),
                        html.Table(id="data-summary-table", 
                                 style={"font-family": "Segoe UI", "font-size": "14px"}),
                    ], style={"flex": "50%", "border-left": "1px solid #ddd", "padding-left": "20px"}),
                ]),

                # Data Grid
                html.Div(style={
                    "border": "1px solid #ddd",
                    "padding": "15px"
                }, children=[
                    html.H4("Data Grid", 
                           style={"font-family": "Segoe UI", "font-size": "20px"}),
                    dt.DataTable(
                        id="data-table",
                        columns=[
                            {"name": "Exclude?", "id": "Exclude?", "editable": True},
                            {"name": "Shot #", "id": "Shot #", "editable": False},
                            {"name": "Distance", "id": "Distance", "editable": True},
                        ],
                        data=df.to_dict("records"),
                        editable=True,
                        row_selectable="multi",
                        selected_rows=[],
                        sort_action='native',
                        sort_mode='single',
                        style_table={"overflowX": "auto", "height": "400px"},
                        style_cell={"textAlign": "left", "font-family": "Segoe UI", "font-size": "14px"},
                        style_header={"fontWeight": "bold", "font-family": "Segoe UI", "font-size": "16px"},
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{Exclude?} = 'Y'"},
                                "backgroundColor": "#FFECEC",
                                "color": "black",
                            },
                        ],
                    )
                ])
            ])
        ]),
    ])
])

# Callback for file upload
@app.callback(
    [Output('data-table', 'data'),
     Output('data-table', 'columns'),
     Output('column-select-container', 'style'),
     Output('column-select', 'options'),
     Output('column-select', 'value'),
     Output('upload-error', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate

    try:
        # Decode the file contents
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Read the file into a dataframe
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return no_update, no_update, no_update, no_update, no_update, "Please upload a .csv or .xlsx file"

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return no_update, no_update, no_update, no_update, no_update, "No numeric columns found in the file"

        # Create new dataframe with only Exclude? column and numeric columns
        new_df = pd.DataFrame()
        new_df['Exclude?'] = [''] * len(df)  # Default to blank (included)
        for col in numeric_cols:
            new_df[col] = df[col]

        # Create columns for datatable
        columns = [
            {"name": "Exclude?", "id": "Exclude?", "editable": True}
        ] + [
            {"name": i, "id": i, "editable": True} for i in new_df.columns if i != "Exclude?"
        ]

        # Create dropdown options for numeric columns
        dropdown_options = [{'label': col, 'value': col} for col in numeric_cols]
        
        # Show the column selector
        column_select_style = {
            'display': 'block',
            'margin-top': '20px'
        }

        return (new_df.to_dict('records'), columns, column_select_style, 
                dropdown_options, numeric_cols[0], "")

    except Exception as e:
        return no_update, no_update, no_update, no_update, no_update, f"Error processing file: {str(e)}"

# Callback for saving data
@app.callback(
    Output("download-data", "data"),
    Input("save-button", "n_clicks"),
    State("data-table", "data"),
    prevent_initial_call=True,
)
def save_data(n_clicks, table_data):
    if n_clicks is None:
        raise PreventUpdate
    
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "data.csv", index=False)

# Callback to manage specification input visibility
@app.callback(
    [Output("lsl-input", "style"),
     Output("target-input", "style"),
     Output("usl-input", "style")],
    Input("spec-type", "value")
)
def update_spec_visibility(spec_type):
    base_style = {"width": "60px"}
    hidden_style = {"width": "60px", "display": "none"}
    
    if spec_type == "target":
        return base_style, base_style, base_style
    elif spec_type == "higher":
        return base_style, hidden_style, hidden_style
    elif spec_type == "lower":
        return hidden_style, hidden_style, base_style
    else:
        return hidden_style, hidden_style, hidden_style

# Main callback for histogram, box plot, and tables
@app.callback(
    [Output("histogram", "figure"),
     Output("box-plot", "figure"),
     Output("imr-plot", "figure"),
     Output("stats-table", "children"),
     Output("data-summary-table", "children")],
    [Input("distribution-checklist", "value"),
     Input("data-table", "data"),
     Input("nbins-slider", "value"),
     Input("selected-bin", "data"),
     Input("data-table", "selected_rows"),
     Input("histogram", "relayoutData"),
     Input("column-select", "value"),
     Input("spec-type", "value"),
     Input("lsl-input", "value"),
     Input("target-input", "value"),
     Input("usl-input", "value")]
)
def update_plots(selected_distributions, table_data, nbins, selected_bin, 
                selected_rows, relayout_data, selected_column, spec_type,
                lsl, target, usl):
    if not selected_column:
        selected_column = "Distance"  # Default column
        
    if nbins is None:
        nbins = default_bins

    # Convert selected column to numeric and filter by Exclude? = 'Y'
    try:
        filtered_data = [
            float(row[selected_column]) for row in table_data if row["Exclude?"] != "Y"
        ]
    except (ValueError, KeyError):
        return go.Figure(), go.Figure(), go.Figure(), [], []

    if not filtered_data:
        return go.Figure(), go.Figure(), go.Figure(), [], []

    # Calculate statistics
    mean = np.mean(filtered_data)
    if mean <= 0:  # Add check for non-positive mean
        raise ValueError("Cannot fit lognormal distribution to non-positive data")
    std_dev = np.std(filtered_data)
    
    # Calculate axis ranges
    data_min = min(filtered_data)
    data_max = max(filtered_data)
    data_range = data_max - data_min
    
    # Set minimum with 10% padding below min
    plot_min = min(data_min - (data_range * 0.10), mean - 4 * std_dev)
    # Set maximum with enough room for highest value plus padding
    plot_max = max(data_max + (data_range * 0.05), 
                  mean + 4 * std_dev)
    
    # Use relayout data if available, otherwise use calculated ranges
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        x_min = float(relayout_data['xaxis.range[0]'])
        x_max = float(relayout_data['xaxis.range[1]'])
    else:
        x_min = plot_min
        x_max = plot_max
    
    # Calculate statistics for tables and box plot
    ci_margin = 1.96 * (std_dev / np.sqrt(len(filtered_data)))
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin
    
    p10 = np.percentile(filtered_data, 10)
    p25 = np.percentile(filtered_data, 25)
    p50 = np.median(filtered_data)
    p75 = np.percentile(filtered_data, 75)
    p90 = np.percentile(filtered_data, 90)
    
    moving_range = np.abs(np.diff(filtered_data))
    avg_moving_range = np.mean(moving_range)
    sigma_local = avg_moving_range / 1.128
    
    # Calculate outliers for box plot
    iqr = p75 - p25
    outlier_threshold_low = p25 - 1.5 * iqr
    outlier_threshold_high = p75 + 1.5 * iqr
    outliers = [x for x in filtered_data if x < outlier_threshold_low or x > outlier_threshold_high]
    non_outliers = [x for x in filtered_data if outlier_threshold_low <= x <= outlier_threshold_high]
    outlier_count = len(outliers)

    # Create histogram
    hist_fig = go.Figure()
    
    # Calculate bin information consistently - use only non-excluded data
    filtered_data = [round(float(row[selected_column]), 3) for row in table_data if row["Exclude?"] != "Y"]
    data_min = min(filtered_data)
    data_max = max(filtered_data)
    bin_width = (data_max - data_min) / nbins
    
    # Create histogram with explicit bins
    hist_fig.add_trace(go.Histogram(
        x=filtered_data,
        nbinsx=nbins,
        name="Data",
        marker=dict(
            color=[
                "#6baed6" if selected_bin is not None and 
                abs(x - float(selected_bin)) <= bin_width/2 + 1e-10
                else "lightblue"
                for x in filtered_data
            ],
            line=dict(
                color="#444444",
                width=1.0
            )
        ),
        opacity=0.75,
        autobinx=False,
        xbins=dict(
            start=data_min,
            end=data_max,
            size=bin_width
        )
    ))

    # Add distribution curves
    x_vals = np.linspace(x_min, x_max, 500)
    for dist in selected_distributions:
        if dist == "normal":
            params = norm.fit(filtered_data)
            pdf = norm.pdf(x_vals, *params)
        elif dist == "lognormal":
            # Safely calculate lognormal parameters
            try:
                log_sigma_global = np.sqrt(np.log1p((std_dev/mean)**2))
                log_mu = np.log(mean) - 0.5 * log_sigma_global**2
                pdf = lognorm.pdf(x_vals, s=log_sigma_global, scale=np.exp(log_mu))
            except (ValueError, ZeroDivisionError):
                print("Error: Cannot fit lognormal distribution to the data")
                pdf = None
        elif dist == "exponential":
            params = expon.fit(filtered_data)
            pdf = expon.pdf(x_vals, *params)
        elif dist == "weibull":
            params = weibull_min.fit(filtered_data)
            pdf = weibull_min.pdf(x_vals, *params)
        elif dist == "gamma":
            params = gamma.fit(filtered_data)
            pdf = gamma.pdf(x_vals, *params)
        elif dist == "uniform":
            params = uniform.fit(filtered_data)
            pdf = uniform.pdf(x_vals, *params)

        if pdf is not None:
            hist_fig.add_trace(go.Scatter(
                x=x_vals,
                y=pdf * len(filtered_data) * np.diff(x_vals)[0],
                mode="lines",
                name=dist.capitalize(),
                yaxis="y2"
            ))

    # Add specification lines to histogram
    if spec_type == "target":
        if lsl is not None:
            hist_fig.add_vline(x=lsl, line=dict(color="darkred", width=1.5), 
                             annotation=dict(text="LSL", x=lsl, xanchor="right",
                                           font=dict(color="darkred")))
        if target is not None:
            hist_fig.add_vline(x=target, line=dict(color="darkgrey", width=0.75),
                             annotation=dict(text="Target", x=target, xanchor="right",
                                           font=dict(color="darkgrey")))
        if usl is not None:
            hist_fig.add_vline(x=usl, line=dict(color="darkred", width=1.5),
                             annotation=dict(text="USL", x=usl, xanchor="left",
                                           font=dict(color="darkred")))
    elif spec_type == "higher" and lsl is not None:
        hist_fig.add_vline(x=lsl, line=dict(color="darkred", width=1.5),
                          annotation=dict(text="LSL", x=lsl, xanchor="right",
                                        font=dict(color="darkred")))
    elif spec_type == "lower" and usl is not None:
        hist_fig.add_vline(x=usl, line=dict(color="darkred", width=1.5),
                          annotation=dict(text="USL", x=usl, xanchor="left",
                                        font=dict(color="darkred")))

    hist_fig.update_layout(
        title=f"Histogram with Fitted Distributions - {selected_column}",
        xaxis_title=selected_column,
        xaxis=dict(range=[plot_min, plot_max]),
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="",
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        bargap=0
    )

    # Create box plot
    box_fig = go.Figure()
    
    # Add box plot
    box_fig.add_trace(go.Box(
        x=non_outliers,
        name="",
        boxpoints=False,
        marker_color="lightblue",
        line=dict(
            color="#444444",
            width=1.0
        ),
        fillcolor="lightblue"
    ))
    
    # Add outliers as scatter points
    if outliers:
        box_fig.add_trace(go.Scatter(
            x=outliers,
            y=[0] * len(outliers),
            mode='markers',
            name='Outliers',
            marker=dict(
                color='red',
                symbol='circle',
                size=8
            )
        ))

    # Add specification lines to box plot
    if spec_type == "target":
        if lsl is not None:
            box_fig.add_vline(x=lsl, line_dash="dash", line_color="red", name="LSL")
        if target is not None:
            box_fig.add_vline(x=target, line_dash="dash", line_color="green", name="Target")
        if usl is not None:
            box_fig.add_vline(x=usl, line_dash="dash", line_color="red", name="USL")
    elif spec_type == "higher":
        if lsl is not None:
            box_fig.add_vline(x=lsl, line_dash="dash", line_color="red", name="LSL")
    elif spec_type == "lower":
        if usl is not None:
            box_fig.add_vline(x=usl, line_dash="dash", line_color="red", name="USL")

    box_fig.update_layout(
        title=f"Box Plot - {selected_column}",
        xaxis_title=selected_column,
        xaxis=dict(range=[plot_min, plot_max]),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        height=200,
        margin=dict(t=30, b=30),
        plot_bgcolor='white'
    )

    # Create I-MR Chart
    imr_fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Individuals (I) Chart', 'Moving Range (MR) Chart'),
                           vertical_spacing=0.15)
    
    # Get data in time order (Shot #)
    ordered_data = [(row['Shot #'], float(row[selected_column])) 
                   for row in table_data if row["Exclude?"] != "Y"]
    ordered_data.sort(key=lambda x: x[0])
    
    x_vals = [x[0] for x in ordered_data]
    y_vals = [x[1] for x in ordered_data]
    
    # Calculate control limits for Individuals
    mean_i = np.mean(y_vals)
    moving_range = np.abs(np.diff(y_vals))
    mean_mr = np.mean(moving_range)
    
    ucl_i = mean_i + 2.66 * mean_mr
    lcl_i = mean_i - 2.66 * mean_mr
    
    # Calculate control limits for Moving Range
    ucl_mr = 3.267 * mean_mr
    lcl_mr = 0
    
    # Create point colors based on control limits
    i_colors = ['darkred' if y > ucl_i or y < lcl_i else '#4682B4' for y in y_vals]
    mr_colors = ['darkred' if mr > ucl_mr else '#4682B4' for mr in moving_range]
    
    # Add Individuals connecting line first
    imr_fig.add_trace(
        go.Scatter(x=x_vals, y=y_vals, mode='lines',
                  line=dict(color='#4682B4', width=0.75),
                  showlegend=False),
        row=1, col=1
    )
    
    # Add Individuals points with filled markers for out-of-control points
    for x, y, color in zip(x_vals, y_vals, i_colors):
        symbol = 'circle' if color == 'darkred' else 'circle-open'
        imr_fig.add_trace(
            go.Scatter(x=[x], y=[y], mode='markers',
                      marker=dict(
                          symbol=symbol,
                          size=8,
                          color=color,
                          line=dict(width=1, color=color)
                      ),
                      showlegend=False),
            row=1, col=1
        )
    
    # Add Moving Range connecting line first
    imr_fig.add_trace(
        go.Scatter(x=x_vals[1:], y=moving_range, mode='lines',
                  line=dict(color='#4682B4', width=0.75),
                  showlegend=False),
        row=2, col=1
    )
    
    # Add Moving Range points with filled markers for out-of-control points
    for x, mr, color in zip(x_vals[1:], moving_range, mr_colors):
        symbol = 'circle' if color == 'darkred' else 'circle-open'
        imr_fig.add_trace(
            go.Scatter(x=[x], y=[mr], mode='markers',
                      marker=dict(
                          symbol=symbol,
                          size=8,
                          color=color,
                          line=dict(width=1, color=color)
                      ),
                      showlegend=False),
            row=2, col=1
        )
    
    # Add control limits and center lines
    # Individuals
    imr_fig.add_hline(y=mean_i, line=dict(color="gray", dash="dash", width=1), row=1, col=1)
    imr_fig.add_hline(y=ucl_i, line=dict(color="red", width=1), row=1, col=1,
                      annotation=dict(text="UNPL", x=1, xanchor="left",
                                    font=dict(color="red")))
    imr_fig.add_hline(y=lcl_i, line=dict(color="red", width=1), row=1, col=1,
                      annotation=dict(text="LNPL", x=1, xanchor="left",
                                    font=dict(color="red")))
    
    # Moving Range
    imr_fig.add_hline(y=mean_mr, line=dict(color="gray", dash="dash", width=1), row=2, col=1)
    imr_fig.add_hline(y=ucl_mr, line=dict(color="red", width=1), row=2, col=1,
                      annotation=dict(text="UNPL", x=1, xanchor="left",
                                    font=dict(color="red")))
    
    # Update layout
    imr_fig.update_layout(
        height=500,
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50),
        clickmode='event+select'  # Enable clicking
    )
    
    # Update axes
    imr_fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False,
        title_text="Sample",
        row=2, col=1
    )
    
    imr_fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False,
        row=1, col=1
    )
    
    imr_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False,
        title_text="Distance",
        row=1, col=1
    )
    
    imr_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False,
        title_text="Moving Range",
        row=2, col=1
    )

    # Create tables
    stats_table = html.Table([
        html.Tr([html.Th("Metric"), html.Th("Value")]),
        html.Tr([html.Td("Mean"), html.Td(f"{mean:.2f}")]),
        html.Tr([html.Td("95% CI"), html.Td(f"{ci_lower:.2f} to {ci_upper:.2f}")]),
        html.Tr([html.Td("Standard Deviation"), html.Td(f"{std_dev:.2f}")]),
        html.Tr([html.Td("Ave Moving Range"), html.Td(f"{avg_moving_range:.2f}")]),
        html.Tr([html.Td("Sigma (Local SD)"), html.Td(f"{sigma_local:.2f}")]),
        html.Tr([html.Td("10th Percentile"), html.Td(f"{p10:.0f}")]),
        html.Tr([html.Td("25th Percentile"), html.Td(f"{p25:.0f}")]),
        html.Tr([html.Td("Median"), html.Td(f"{p50:.0f}")]),
        html.Tr([html.Td("75th Percentile"), html.Td(f"{p75:.0f}")]),
        html.Tr([html.Td("90th Percentile"), html.Td(f"{p90:.0f}")]),
        html.Tr([html.Td("# of Outliers (>1.5 IQR)"), html.Td(f"{outlier_count}")]),
    ], style={"width": "100%"})

    data_summary_table = html.Table([
        html.Tr([html.Th("Metric"), html.Th("Count")]),
        html.Tr([html.Td("Number of Data Points"), html.Td(len(table_data))]),
        html.Tr([html.Td("Number of Excluded Points"), 
                html.Td(sum(1 for row in table_data if row["Exclude?"] == "Y"))]),
        html.Tr([html.Td("Number of Selected Points"), 
                html.Td(len([i for i in selected_rows if table_data[i]["Exclude?"] != "Y"]) if selected_rows else 0)]),
    ], style={"width": "100%"})

    return hist_fig, box_fig, imr_fig, stats_table, data_summary_table

# Handle histogram clicks
@app.callback(
    [Output("data-table", "selected_rows"),
     Output("selected-bin", "data")],
    Input("histogram", "clickData"),
    [State("data-table", "data"),
     State("nbins-slider", "value"),
     State("column-select", "value")]
)
def update_selected_rows(click_data, table_data, nbins, selected_column):
    if not click_data:
        return [], None
    
    if not selected_column:
        selected_column = "Distance"
    
    try:
        # Get the clicked point
        clicked_x = click_data['points'][0]['x']
        
        # Calculate bins using same method as histogram
        filtered_data = [round(float(row[selected_column]), 3) for row in table_data if row["Exclude?"] != "Y"]
        data_min = min(filtered_data)
        data_max = max(filtered_data)
        bin_width = (data_max - data_min) / nbins
        
        # Calculate bin index with tolerance
        TOLERANCE = 1e-10  # Small tolerance for floating-point comparison
        bin_index = int((clicked_x - data_min) / bin_width)
        bin_start = data_min + (bin_index * bin_width)
        bin_end = bin_start + bin_width
        
        # Find all rows that fall within this bin, using tolerance
        selected_rows = [
            i for i, row in enumerate(table_data)
            if abs(float(row[selected_column]) - clicked_x) <= bin_width/2 + TOLERANCE
        ]
        
        return selected_rows, clicked_x
        
    except Exception as e:
        print(f"Error processing click data: {e}")
        return [], None

# Add this helper function at the top of your file with other functions
def create_summary_table(table_data, selected_rows):
    """Create a summary table with basic statistics"""
    return html.Table([
        html.Tr([html.Th("Metric"), html.Th("Count", style={"textAlign": "right"})]),
        html.Tr([html.Td("Total Data Points"), 
                html.Td(str(len(table_data)), style={"textAlign": "right"})]),
        html.Tr([html.Td("Selected Points"), 
                html.Td(str(len(selected_rows)), style={"textAlign": "right"})]),
    ], style={"width": "100%"})

# Update the callback to handle the bin width calculation correctly
@app.callback(
    [Output("data-table", "selected_rows", allow_duplicate=True),
     Output("data-summary-table", "children", allow_duplicate=True)],
    [Input("histogram", "clickData"),
     Input("box-plot", "clickData"),
     Input("imr-plot", "clickData")],
    [State("data-table", "data"),
     State("column-select", "value"),
     State("nbins-slider", "value")],
    prevent_initial_call=True
)
def update_selection_from_all_plots(hist_click, box_click, imr_click, table_data, selected_column, nbins):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [], create_summary_table(table_data, [])
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not selected_column:
        selected_column = "Distance"
    
    try:
        selected_indices = []
        
        if trigger_id == "histogram" and hist_click:
            # Get clicked bin center
            bin_x = hist_click['points'][0]['x']
            
            # Calculate bin width using only non-excluded data
            filtered_data = [float(row[selected_column]) for row in table_data if row["Exclude?"] != "Y"]
            data_range = max(filtered_data) - min(filtered_data)
            bin_width = data_range / nbins
            
            # Find matching rows
            selected_indices = [
                i for i, row in enumerate(table_data)
                if row["Exclude?"] != "Y" and
                abs(float(row[selected_column]) - bin_x) <= bin_width/2
            ]
            
        elif trigger_id == "box-plot" and box_click:
            point_value = box_click['points'][0]['x']
            selected_indices = [
                i for i, row in enumerate(table_data)
                if float(row[selected_column]) == point_value
            ]
            
        elif trigger_id == "imr-plot" and imr_click:
            point = imr_click['points'][0]
            y_value = point['y']
            subplot = point['curveNumber']
            
            if subplot <= 1:  # Individuals chart
                selected_indices = [
                    i for i, row in enumerate(table_data)
                    if float(row[selected_column]) == y_value
                ]
            else:  # Moving Range chart
                moving_ranges = [
                    abs(float(table_data[i+1][selected_column]) - float(table_data[i][selected_column]))
                    for i in range(len(table_data)-1)
                ]
                try:
                    mr_index = moving_ranges.index(y_value)
                    selected_indices = [mr_index, mr_index + 1]
                except ValueError:
                    selected_indices = []
        
        # Create detailed summary table
        if selected_indices:
            summary_table = html.Table([
                html.Tr([html.Th("Metric"), html.Th("Count", style={"textAlign": "right"})]),
                html.Tr([html.Td("Total Data Points"), 
                        html.Td(str(len(table_data)), style={"textAlign": "right"})]),
                html.Tr([html.Td("Selected Points"), 
                        html.Td(str(len(selected_indices)), style={"textAlign": "right"})]),
                html.Tr([html.Td("Selected Values"), 
                        html.Td(", ".join([f"{float(table_data[i][selected_column]):.1f}" 
                                         for i in selected_indices]), 
                               style={"textAlign": "right"})]),
            ], style={"width": "100%"})
        else:
            summary_table = create_summary_table(table_data, [])
        
        return selected_indices, summary_table
        
    except Exception as e:
        print(f"Error processing plot click: {e}")
        return [], create_summary_table(table_data, [])

# Add this function if not already present
def calculate_capability_metrics(data, lsl, usl, target, sigma_local, sigma_global):
    """Calculate process capability metrics"""
    mean = np.mean(data)
    
    try:
        # Calculate Cp and Pp (if both LSL and USL exist)
        if lsl is not None and usl is not None:
            cp = (usl - lsl) / (6 * sigma_local) if sigma_local > 0 else None
            pp = (usl - lsl) / (6 * sigma_global) if sigma_global > 0 else None
        else:
            cp = pp = None
        
        # Calculate Cpk and Ppk
        if lsl is not None:
            cpu_local = (mean - lsl) / (3 * sigma_local) if sigma_local > 0 else None
            cpu_global = (mean - lsl) / (3 * sigma_global) if sigma_global > 0 else None
        else:
            cpu_local = cpu_global = None
            
        if usl is not None:
            cpl_local = (usl - mean) / (3 * sigma_local) if sigma_local > 0 else None
            cpl_global = (usl - mean) / (3 * sigma_global) if sigma_global > 0 else None
        else:
            cpl_local = cpl_global = None
        
        # Get the minimum of upper and lower capability indices
        cpk = min(cpu_local, cpl_local) if cpu_local and cpl_local else None
        ppk = min(cpu_global, cpl_global) if cpu_global and cpl_global else None
        
        return {
            'Cp': cp,
            'Pp': pp,
            'Cpk': cpk,
            'Ppk': ppk
        }
    except Exception as e:
        print(f"Error calculating capability metrics: {e}")
        return {'Cp': None, 'Pp': None, 'Cpk': None, 'Ppk': None}

# Add this callback if not already present
@app.callback(
    Output("capability-metrics-table", "children"),
    [Input("data-table", "data"),
     Input("column-select", "value"),
     Input("lsl-input", "value"),
     Input("usl-input", "value"),
     Input("target-input", "value"),
     Input("capability-distribution", "value")]
)
def update_capability_metrics(table_data, selected_column, lsl, usl, target, selected_dist):
    if not selected_column:
        selected_column = "Distance"
    
    try:
        # Get filtered data
        filtered_data = [float(row[selected_column]) 
                        for row in table_data 
                        if row["Exclude?"] != "Y"]
        
        if not filtered_data:
            return "No data available for capability analysis"
        
        # Calculate basic statistics
        mean = np.mean(filtered_data)
        if mean <= 0:  # Add check for non-positive mean
            raise ValueError("Cannot fit lognormal distribution to non-positive data")
            
        sigma_global = np.std(filtered_data, ddof=1)
        
        # Calculate sigma_local from average moving range
        moving_ranges = [abs(filtered_data[i+1] - filtered_data[i]) 
                        for i in range(len(filtered_data)-1)]
        sigma_local = np.mean(moving_ranges) / 1.128 if moving_ranges else None
        
        # Calculate traditional capability metrics
        metrics = calculate_capability_metrics(
            filtered_data, lsl, usl, target, sigma_local, sigma_global
        )
        
        # Calculate OOS percentages based on selected distribution
        if lsl is not None or usl is not None:
            # Calculate sigma values
            moving_ranges = [abs(filtered_data[i+1] - filtered_data[i]) 
                           for i in range(len(filtered_data)-1)]
            sigma_local = np.mean(moving_ranges) / 1.128 if moving_ranges else None
            sigma_global = np.std(filtered_data, ddof=1)
            
            # Calculate actual OOS percentage first (this was missing)
            actual_oos = len([x for x in filtered_data 
                            if (lsl is not None and x < lsl) or 
                               (usl is not None and x > usl)]) / len(filtered_data) * 100
            
            if selected_dist == "normal":
                # Normal distribution calculations
                local_oos_lower = norm.cdf(lsl, mean, sigma_local) if lsl is not None else 0
                local_oos_upper = 1 - norm.cdf(usl, mean, sigma_local) if usl is not None else 0
                local_oos_total = (local_oos_lower + local_oos_upper) * 100
                
                global_oos_lower = norm.cdf(lsl, mean, sigma_global) if lsl is not None else 0
                global_oos_upper = 1 - norm.cdf(usl, mean, sigma_global) if usl is not None else 0
                global_oos_total = (global_oos_lower + global_oos_upper) * 100
                
            elif selected_dist == "lognormal":
                # For lognormal, we need to convert our arithmetic sigma to log-space sigma
                log_sigma_local = np.sqrt(np.log(1 + (sigma_local/mean)**2))
                log_sigma_global = np.sqrt(np.log(1 + (sigma_global/mean)**2))
                
                # Calculate log-space mu (location parameter)
                log_mu = np.log(mean) - 0.5 * log_sigma_global**2
                
                # Calculate OOS using local variation
                local_oos_lower = lognorm.cdf(lsl, s=log_sigma_local, scale=np.exp(log_mu)) if lsl is not None else 0
                local_oos_upper = 1 - lognorm.cdf(usl, s=log_sigma_local, scale=np.exp(log_mu)) if usl is not None else 0
                local_oos_total = (local_oos_lower + local_oos_upper) * 100
                
                # Calculate OOS using global variation
                global_oos_lower = lognorm.cdf(lsl, s=log_sigma_global, scale=np.exp(log_mu)) if lsl is not None else 0
                global_oos_upper = 1 - lognorm.cdf(usl, s=log_sigma_global, scale=np.exp(log_mu)) if usl is not None else 0
                global_oos_total = (global_oos_lower + global_oos_upper) * 100
                
            elif selected_dist == "exponential":
                # Calculate rate parameters using respective sigma values
                rate_local = 1/sigma_local if sigma_local else 0
                rate_global = 1/sigma_global if sigma_global else 0
                
                # Calculate OOS percentages
                local_oos_lower = expon.cdf(lsl, loc=0, scale=1/rate_local) if lsl is not None else 0
                local_oos_upper = 1 - expon.cdf(usl, loc=0, scale=1/rate_local) if usl is not None else 0
                local_oos_total = (local_oos_lower + local_oos_upper) * 100
                
                global_oos_lower = expon.cdf(lsl, loc=0, scale=1/rate_global) if lsl is not None else 0
                global_oos_upper = 1 - expon.cdf(usl, loc=0, scale=1/rate_global) if usl is not None else 0
                global_oos_total = (global_oos_lower + global_oos_upper) * 100
                
            elif selected_dist == "weibull":
                # Fit base parameters
                shape, loc, scale = weibull_min.fit(filtered_data)
                
                # Calculate local and global scale parameters
                scale_local = sigma_local / gamma(1 + 1/shape)
                scale_global = sigma_global / gamma(1 + 1/shape)
                
                # Calculate OOS percentages
                local_oos_lower = weibull_min.cdf(lsl, shape, loc=0, scale=scale_local) if lsl is not None else 0
                local_oos_upper = 1 - weibull_min.cdf(usl, shape, loc=0, scale=scale_local) if usl is not None else 0
                local_oos_total = (local_oos_lower + local_oos_upper) * 100
                
                global_oos_lower = weibull_min.cdf(lsl, shape, loc=0, scale=scale_global) if lsl is not None else 0
                global_oos_upper = 1 - weibull_min.cdf(usl, shape, loc=0, scale=scale_global) if usl is not None else 0
                global_oos_total = (global_oos_lower + global_oos_upper) * 100
                
            elif selected_dist == "gamma":
                # Fit base parameters
                shape, loc, scale = gamma.fit(filtered_data)
                
                # Calculate local and global scale parameters
                scale_local = sigma_local**2 / mean
                scale_global = sigma_global**2 / mean
                shape_local = mean / scale_local
                shape_global = mean / scale_global
                
                # Calculate OOS percentages
                local_oos_lower = gamma.cdf(lsl, shape_local, loc=0, scale=scale_local) if lsl is not None else 0
                local_oos_upper = 1 - gamma.cdf(usl, shape_local, loc=0, scale=scale_local) if usl is not None else 0
                local_oos_total = (local_oos_lower + local_oos_upper) * 100
                
                global_oos_lower = gamma.cdf(lsl, shape_global, loc=0, scale=scale_global) if lsl is not None else 0
                global_oos_upper = 1 - gamma.cdf(usl, shape_global, loc=0, scale=scale_global) if usl is not None else 0
                global_oos_total = (global_oos_lower + global_oos_upper) * 100
                
            elif selected_dist == "uniform":
                # Calculate ranges based on respective sigma values
                local_range = np.sqrt(12) * sigma_local
                global_range = np.sqrt(12) * sigma_global
                
                # Calculate OOS percentages using uniform distribution properties
                local_oos_total = sum([
                    1 if lsl is not None and lsl > mean - local_range/2 else 0,
                    1 if usl is not None and usl < mean + local_range/2 else 0
                ]) * 100
                
                global_oos_total = sum([
                    1 if lsl is not None and lsl > mean - global_range/2 else 0,
                    1 if usl is not None and usl < mean + global_range/2 else 0
                ]) * 100
        else:
            # Initialize all OOS variables when no specification limits are provided
            local_oos_total = None
            global_oos_total = None
            actual_oos = None
        
        # Create the table with improved styling
        return html.Table([
            # Traditional Metrics Section
            html.Thead([
                html.Tr([
                    html.Th("Traditional Capability Metrics", 
                           style={"text-align": "left", "padding": "8px", "background-color": "#f5f5f5", "colspan": "2"})
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Cp - Process Capability (Within)", style={"padding": "8px"}),
                    html.Td(f"{metrics['Cp']:.2f}" if metrics['Cp'] is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ]),
                html.Tr([
                    html.Td("Cpk - Process Capability Index (Within)", style={"padding": "8px"}),
                    html.Td(f"{metrics['Cpk']:.2f}" if metrics['Cpk'] is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ]),
                html.Tr([
                    html.Td("Pp - Process Performance (Overall)", style={"padding": "8px"}),
                    html.Td(f"{metrics['Pp']:.2f}" if metrics['Pp'] is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ]),
                html.Tr([
                    html.Td("Ppk - Process Performance Index (Overall)", style={"padding": "8px"}),
                    html.Td(f"{metrics['Ppk']:.2f}" if metrics['Ppk'] is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ])
            ]),
            
            # Out of Specification Section
            html.Thead([
                html.Tr([
                    html.Th(f"Expected OOS % ({selected_dist.capitalize()} Distribution)", 
                           style={"text-align": "left", "padding": "8px", "background-color": "#f5f5f5", "colspan": "2"})
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Expected % OOS (Local Variation)", style={"padding": "8px"}),
                    html.Td(f"{local_oos_total:.2f}%" if local_oos_total is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ]),
                html.Tr([
                    html.Td("Expected % OOS (Global Variation)", style={"padding": "8px"}),
                    html.Td(f"{global_oos_total:.2f}%" if global_oos_total is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ]),
                html.Tr([
                    html.Td("Actual % Out of Specification", style={"padding": "8px"}),
                    html.Td(f"{actual_oos:.2f}%" if actual_oos is not None else "N/A",
                           style={"text-align": "right", "padding": "8px"})
                ])
            ])
        ], style={"width": "100%", "border-collapse": "collapse", "border": "1px solid #ddd"})
        
    except Exception as e:
        print(f"Error updating capability metrics: {e}")
        return "Error calculating capability metrics"

# Add this callback if not already present
@app.callback(
    Output("capability-plot", "figure"),
    [Input("data-table", "data"),
     Input("column-select", "value"),
     Input("lsl-input", "value"),
     Input("usl-input", "value"),
     Input("target-input", "value"),
     Input("capability-distribution", "value")]
)
def update_capability_plot(table_data, selected_column, lsl, usl, target, selected_dist):
    if not selected_column:
        selected_column = "Distance"
    
    try:
        # Get filtered data
        filtered_data = [float(row[selected_column]) 
                        for row in table_data 
                        if row["Exclude?"] != "Y"]
        
        if not filtered_data:
            return {}
        
        # Calculate statistics
        mean = np.mean(filtered_data)
        std = np.std(filtered_data, ddof=1)
        
        # Create x range
        x = np.linspace(min(filtered_data), max(filtered_data), 200)
        
        # Calculate y values based on selected distribution
        if selected_dist == "normal":
            y = norm.pdf(x, mean, std)
        elif selected_dist == "lognormal":
            shape, loc, scale = lognorm.fit(filtered_data)
            y = lognorm.pdf(x, shape, loc=loc, scale=scale)
        elif selected_dist == "exponential":
            loc, scale = expon.fit(filtered_data)
            y = expon.pdf(x, loc=loc, scale=scale)
        elif selected_dist == "weibull":
            shape, loc, scale = weibull_min.fit(filtered_data)
            y = weibull_min.pdf(x, shape, loc=loc, scale=scale)
        elif selected_dist == "gamma":
            shape, loc, scale = gamma.fit(filtered_data)
            y = gamma.pdf(x, shape, loc=loc, scale=scale)
        elif selected_dist == "uniform":
            loc, scale = uniform.fit(filtered_data)
            y = uniform.pdf(x, loc=loc, scale=scale)
        
        # Create the figure
        fig = go.Figure()
        
        # Add shaded areas for specification limits
        if lsl is not None and usl is not None:
            x_within = x[(x >= lsl) & (x <= usl)]
            y_within = y[(x >= lsl) & (x <= usl)]
            fig.add_trace(go.Scatter(
                x=x_within,
                y=y_within,
                fill='tozeroy',
                fillcolor='rgba(173, 216, 230, 0.5)',
                line=dict(width=0),
                showlegend=False
            ))
        
        # Add distribution curve
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='navy', width=2),
            name=f'{selected_dist.capitalize()} Distribution'
        ))
        
        # Add specification lines
        if lsl is not None:
            fig.add_vline(x=lsl, line_color='darkred', line_dash='solid', line_width=2,
                         annotation_text="LSL")
        if usl is not None:
            fig.add_vline(x=usl, line_color='darkred', line_dash='solid', line_width=2,
                         annotation_text="USL")
        if target is not None:
            fig.add_vline(x=target, line_color='darkgrey', line_dash='solid', line_width=2,
                         annotation_text="Target")
        
        # Update layout
        fig.update_layout(
            title=f"'{selected_column}' - Capability Analysis ({selected_dist.capitalize()})",
            xaxis_title="Value",
            yaxis_title="Probability Density",
            showlegend=True,
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating capability plot: {e}")
        return {}

# Add this callback if not already present
@app.callback(
    Output('filter-dialog', 'displayed'),
    Input('local-filter-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_filter_dialog(n_clicks):
    if n_clicks:
        return True
    return False

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)