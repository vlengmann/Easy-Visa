"""
Easy Visa Prediction Dashboard
Interactive Dash app for predicting visa certification status
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Load the trained model and feature columns
model = joblib.load("model/model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
feature_importances = np.load("model/feature_importances.npy")

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': feature_importances
}).sort_values('importance', ascending=True)  # Ascending for horizontal bar chart

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Easy Visa Prediction"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Easy Visa Certification Predictor", className="text-center mb-4 mt-4"),
            html.P(
                "Enter employee and job details below to predict visa certification status.",
                className="text-center text-muted mb-4"
            )
        ])
    ]),

    dbc.Row([
        # Left column - Input form
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Employee & Job Information")),
                dbc.CardBody([
                    # Education
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Education Level"),
                            dcc.Dropdown(
                                id='education',
                                options=[
                                    {'label': 'High School', 'value': 1},
                                    {'label': "Bachelor's Degree", 'value': 2},
                                    {'label': "Master's Degree", 'value': 3},
                                    {'label': 'Doctorate', 'value': 4}
                                ],
                                value=2,
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Continent"),
                            dcc.Dropdown(
                                id='continent',
                                options=[
                                    {'label': 'Asia', 'value': 'Asia'},
                                    {'label': 'Europe', 'value': 'Europe'},
                                    {'label': 'Africa', 'value': 'Africa'},
                                    {'label': 'North America', 'value': 'North America'},
                                    {'label': 'South America', 'value': 'South America'},
                                    {'label': 'Oceania', 'value': 'Oceania'}
                                ],
                                value='Asia',
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),

                    # Job Experience & Training
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Has Job Experience?"),
                            dcc.Dropdown(
                                id='has_job_experience',
                                options=[
                                    {'label': 'Yes', 'value': 1},
                                    {'label': 'No', 'value': 0}
                                ],
                                value=1,
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Requires Job Training?"),
                            dcc.Dropdown(
                                id='requires_job_training',
                                options=[
                                    {'label': 'Yes', 'value': 1},
                                    {'label': 'No', 'value': 0}
                                ],
                                value=0,
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),

                    # Company details
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of Employees"),
                            dbc.Input(
                                id='no_of_employees',
                                type='number',
                                value=1000,
                                min=1
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Year of Establishment"),
                            dbc.Input(
                                id='yr_of_estab',
                                type='number',
                                value=2000,
                                min=1800,
                                max=2025
                            )
                        ], width=6)
                    ], className="mb-3"),

                    # Region and Wage
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Region of Employment"),
                            dcc.Dropdown(
                                id='region_of_employment',
                                options=[
                                    {'label': 'Northeast', 'value': 'Northeast'},
                                    {'label': 'South', 'value': 'South'},
                                    {'label': 'Midwest', 'value': 'Midwest'},
                                    {'label': 'West', 'value': 'West'},
                                    {'label': 'Island', 'value': 'Island'}
                                ],
                                value='Northeast',
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Full Time Position?"),
                            dcc.Dropdown(
                                id='full_time_position',
                                options=[
                                    {'label': 'Yes', 'value': 1},
                                    {'label': 'No', 'value': 0}
                                ],
                                value=1,
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),

                    # Wage details
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Prevailing Wage"),
                            dbc.Input(
                                id='prevailing_wage',
                                type='number',
                                value=75000,
                                min=0
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Unit of Wage"),
                            dcc.Dropdown(
                                id='unit_of_wage',
                                options=[
                                    {'label': 'Hourly', 'value': 'Hour'},
                                    {'label': 'Weekly', 'value': 'Week'},
                                    {'label': 'Monthly', 'value': 'Month'},
                                    {'label': 'Yearly', 'value': 'Year'}
                                ],
                                value='Year',
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),

                    # Predict button
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Predict Visa Status",
                                id='predict-button',
                                color="primary",
                                size="lg",
                                className="w-100"
                            )
                        ])
                    ], className="mt-4")
                ])
            ])
        ], width=8),

        # Right column - Prediction result
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Prediction Result")),
                dbc.CardBody([
                    html.Div(id='prediction-output')
                ])
            ], className="mb-3"),

            # Model info card
            dbc.Card([
                dbc.CardHeader(html.H5("Model Information")),
                dbc.CardBody([
                    html.P([html.Strong("Model: "), "XGBoost Classifier"]),
                    html.P([html.Strong("Accuracy: "), "74.5%"]),
                    html.P([html.Strong("F1 Score: "), "82.2%"]),
                    html.Hr(),
                    html.P([html.Strong("Top Features:")], className="mb-1"),
                    html.Ul([
                        html.Li("Education Level (33%)"),
                        html.Li("Job Experience (18%)"),
                        html.Li("Continent (7%)"),
                    ])
                ])
            ])
        ], width=4)
    ]),

    # Feature Importance Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Feature Importance Analysis")),
                dbc.CardBody([
                    dcc.Graph(
                        id='feature-importance-chart',
                        figure=go.Figure(
                            data=[
                                go.Bar(
                                    x=importance_df['importance'],
                                    y=importance_df['feature'],
                                    orientation='h',
                                    marker=dict(
                                        color=importance_df['importance'],
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title="Importance")
                                    ),
                                    text=importance_df['importance'].round(4),
                                    textposition='auto',
                                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                                )
                            ],
                            layout=go.Layout(
                                title='Model Feature Importances',
                                xaxis=dict(title='Relative Importance'),
                                yaxis=dict(title='Features'),
                                height=600,
                                margin=dict(l=200, r=50, t=80, b=50),
                                hovermode='closest'
                            )
                        )
                    ),
                    html.P([
                        html.Strong("Interpretation: "),
                        "Higher values indicate features that have more influence on the model's predictions. ",
                        "Education level and job experience are the most important factors in visa certification decisions."
                    ], className="mt-3 text-muted")
                ])
            ])
        ], width=12)
    ], className="mt-4")
], fluid=True, className="py-4")


# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('education', 'value'),
    State('continent', 'value'),
    State('has_job_experience', 'value'),
    State('requires_job_training', 'value'),
    State('no_of_employees', 'value'),
    State('yr_of_estab', 'value'),
    State('region_of_employment', 'value'),
    State('full_time_position', 'value'),
    State('prevailing_wage', 'value'),
    State('unit_of_wage', 'value'),
    prevent_initial_call=True
)
def predict_visa_status(n_clicks, education, continent, has_exp, req_training,
                       no_employees, yr_estab, region, full_time, wage, wage_unit):
    try:
        # Create input dataframe with zeros for all features
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Set the basic features
        input_data['education_of_employee'] = education
        input_data['has_job_experience'] = has_exp
        input_data['requires_job_training'] = req_training
        input_data['no_of_employees'] = no_employees
        input_data['yr_of_estab'] = yr_estab
        input_data['prevailing_wage'] = wage
        input_data['full_time_position'] = full_time

        # Set one-hot encoded features for continent
        continent_col = f'continent_{continent}'
        if continent_col in input_data.columns:
            input_data[continent_col] = 1

        # Set one-hot encoded features for region
        region_col = f'region_of_employment_{region}'
        if region_col in input_data.columns:
            input_data[region_col] = 1

        # Set one-hot encoded features for wage unit
        wage_col = f'unit_of_wage_{wage_unit}'
        if wage_col in input_data.columns:
            input_data[wage_col] = 1

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Format result
        if prediction == 1:
            result_text = "CERTIFIED"
            result_color = "success"
            icon = "✓"
            prob = prediction_proba[1] * 100
        else:
            result_text = "DENIED"
            result_color = "danger"
            icon = "✗"
            prob = prediction_proba[0] * 100

        return dbc.Alert([
            html.H3([f"{icon} {result_text}"], className="alert-heading text-center"),
            html.Hr(),
            html.P(f"Confidence: {prob:.1f}%", className="text-center mb-0"),
            html.P([
                html.Small(f"Probability of Certification: {prediction_proba[1]*100:.1f}%"),
                html.Br(),
                html.Small(f"Probability of Denial: {prediction_proba[0]*100:.1f}%")
            ], className="text-center mt-2 mb-0")
        ], color=result_color, className="text-center")

    except Exception as e:
        return dbc.Alert([
            html.H5("Error", className="alert-heading"),
            html.P(f"An error occurred: {str(e)}")
        ], color="warning")


# Expose the Flask server for deployment
server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=8050)
