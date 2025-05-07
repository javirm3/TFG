import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import sympy as sp
from scipy.optimize import fsolve



def phi_X0_placeholder(s0, R, R_I, I0, c):
    X0 = s0 * R - R_I * c + I0
    if X0 <= 0:
        return 0
    elif X0 <= 1:
        return X0**2
    else:
        return 2*np.sqrt(X0-3/4)

def phi_I_X0_placeholder(g,R,I_I):
    X0_I = g*R+I_I
    if X0_I <= 0:
        return 0
    elif X0_I <= 1:
        return X0_I**2
    else:
        return 2*np.sqrt(X0_I-3/4)
    
def phi_prime_X0_placeholder(s0, R, R_I, I0, c):
    X0 = s0 * R - R_I * c + I0
    if X0 <= 0:
        return 0
    elif X0 <= 1:
        return 2*X0
    else:
        return 1/np.sqrt(X0)

def phi_I_prime_X0_placeholder(g,R,I_I):
    X0_I = g*R+I_I
    if X0_I <= 0:
        return 0
    elif X0_I <= 1:
        return 2*X0_I
    else:
        return 1/np.sqrt(X0_I)

def phi_double_prime_X0_placeholder(s0, R, R_I, I0, c):
    X0 = s0 * R - R_I * c + I0
    if X0 <= 0:
        return 0
    elif X0 <= 1:
        return 2
    return -1/(2*X0*np.sqrt(X0))

def phi_triple_prime_X0_placeholder(s0, R, R_I, I0, c):
    X0 = s0 * R - R_I * c + I0 
    if X0 <= 0:
        return 0
    elif X0 <= 1:
        return 0
    else:
        return 3/(4*(X0**2)*np.sqrt(X0))

def solve_parameters(s0_val, c_val, g_val, I_I_val, guess=(1.0, 1.0, 0.5)):
    def equations(vars):
        R_val, R_I_val, I0_val = vars
        # Calcula los tres residuales
        eq1 = s0_val * phi_prime_X0_placeholder(s0_val, R_val, R_I_val, I0_val, c_val) - 1
        eq2 =       R_val -   phi_X0_placeholder       (s0_val, R_val, R_I_val, I0_val, c_val)
        eq3 =      R_I_val -   phi_I_X0_placeholder    (g_val,  R_val,        I_I_val)
        return [eq1, eq2, eq3]

    sol = fsolve(equations, guess)
    return sol  # array([R_val, R_I_val, I0_val])


def potential_function(X1, X2, params):
    # Extraer parámetros y valores de phi
    phi_p = params["phi_p_val"]
    phi_pp = params["phi_pp_val"]
    phi_ppp = params["phi_ppp_val"]
    phi_Ip = params["phi_Ip_val"]
    s0    = params["s0_val"]
    I0    = params["I0_val"]
    c     = params["c_val"]
    g     = params["g_val"]
    R     = params["R_val"]
    I_L    = params["IL_val"]
    I_C    = params["IC_val"]
    I_R    = params["IR_val"]
    s_L    = params["sL_val"]
    s_C    = params["sC_val"]
    s_R    = params["sR_val"]
    
    # Función potencial simplificada
    return (1/24)*(X1**4*s0**3*(2*c*g*phi_Ip*phi_pp**2 - c*g*phi_Ip*phi_ppp - 2*phi_pp**2*s0) + 2*X1**2*(-I_C*c*g*phi_Ip*phi_pp*s0 - 2*I_C*phi_pp*s0**2 - I_L*c*g*phi_Ip*phi_pp*s0 - 2*I_L*phi_pp*s0**2 + 2*I_R*c*g*phi_Ip*phi_pp*s0 - 2*I_R*phi_pp*s0**2 - R*c*g*phi_Ip*phi_pp*s0*s_C - R*c*g*phi_Ip*phi_pp*s0*s_L + 2*R*c*g*phi_Ip*phi_pp*s0*s_R - 2*R*phi_pp*s0**2*s_C - 2*R*phi_pp*s0**2*s_L - 2*R*phi_pp*s0**2*s_R + 6*X2**2*c*g*phi_Ip*phi_pp**2*s0**3 - 3*X2**2*c*g*phi_Ip*phi_ppp*s0**3 - 6*X2**2*phi_pp**2*s0**4 - 6*X2*c*g*phi_Ip*phi_pp*s0**2 - 3*c*g*phi_Ip*phi_p*s_C - 3*c*g*phi_Ip*phi_p*s_L) + 9*X2**4*s0**3*(2*c*g*phi_Ip*phi_pp**2 - c*g*phi_Ip*phi_ppp - 2*phi_pp**2*s0) - 3*X2**2*(I_C*c*g*phi_Ip*phi_pp*s0 + I_C*phi_pp*s0**2 + I_L*c*g*phi_Ip*phi_pp*s0 + I_L*phi_pp*s0**2 + 7*I_R*c*g*phi_Ip*phi_pp*s0 + I_R*phi_pp*s0**2 + R*c*g*phi_Ip*phi_pp*s0*s_C + R*c*g*phi_Ip*phi_pp*s0*s_L + 7*R*c*g*phi_Ip*phi_pp*s0*s_R + R*phi_pp*s0**2*s_C + R*phi_pp*s0**2*s_L + R*phi_pp*s0**2*s_R + 2*c*g*phi_Ip*phi_p*s_C + 2*c*g*phi_Ip*phi_p*s_L + 8*c*g*phi_Ip*phi_p*s_R) + 12*c*g*phi_Ip*(X1*(I_C*X2*phi_pp*s0 + I_C*phi_p - I_L*X2*phi_pp*s0 - I_L*phi_p + R*X2*phi_pp*s0*s_C - R*X2*phi_pp*s0*s_L + R*phi_p*s_C - R*phi_p*s_L + X2*phi_p*s_C - X2*phi_p*s_L) + X2**3*phi_pp*s0**2 - X2*phi_p*(I_C + I_L - 2*I_R + R*s_C + R*s_L - 2*R*s_R)))/(c*g*phi_Ip)

def create_interactive_3d_plot(parameters):
    # Calcular los valores de phi usando las funciones placeholder
    phi_p_val = phi_prime_X0_placeholder(
        parameters["s0_val"], parameters["R_val"], 
        parameters["R_I_val"], parameters["I0_val"], parameters["c_val"]
    )
    phi_pp_val = phi_double_prime_X0_placeholder(
        parameters["s0_val"], parameters["R_val"], 
        parameters["R_I_val"], parameters["I0_val"], parameters["c_val"]
    )
    phi_ppp_val = phi_triple_prime_X0_placeholder(
        parameters["s0_val"], parameters["R_val"], 
        parameters["R_I_val"], parameters["I0_val"], parameters["c_val"]
    )
    phi_Ip_val = phi_I_prime_X0_placeholder(
        parameters["g_val"], parameters["R_val"], 
        parameters["I_I_val"]
    )

    
    
    # Actualizar parámetros con los valores calculados de phi
    params = parameters.copy()
    params["phi_p_val"] = phi_p_val
    params["phi_pp_val"] = phi_pp_val
    params["phi_ppp_val"] = phi_ppp_val
    params["phi_Ip_val"] = phi_Ip_val
    
    # Crear malla de puntos
    x_vals = np.linspace(-15, 15, 200)
    y_vals = np.linspace(-15, 15, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Calcular Z en cada punto de la malla
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = potential_function(X[i, j], Y[i, j], params)
    
    # Limitar valores extremos para mejor visualización
    low_threshold = -20.0
    high_threshold = 20.0
    Z = np.clip(Z, low_threshold, high_threshold)
    
    # Crear figura interactiva con Plotly
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    fig.update_layout(
        title="Potential ψ(X₁,X₂)",
        scene=dict(
            xaxis_title="X₁",
            yaxis_title="X₂",
            zaxis_title="ψ(X₁,X₂)"
        )
    )
    return fig, (phi_p_val, phi_pp_val, phi_ppp_val)

# Configuración de la aplicación Dash
app = dash.Dash(__name__)
server = app.server

# Estilos CSS personalizados
slider_style = {
    'margin': '8px auto',
    'padding': '3px 5px',
    'border-radius': '5px',
    'background-color': '#f8f9fa',
    'box-shadow': '0 1px 2px rgba(0,0,0,0.1)',
    'width': '90%',  # Make sliders narrower
    'height': '25px',  # Make sliders shorter
}

label_style = {
    'textAlign': 'left',  # Alinea el texto a la izquierda
    'display': 'block',
    'margin-bottom': '1px',
    'font-weight': 'bold',
    'color': '#2c3e50',
    'font-size': '12px',
}

group_style = {
    'border': '1px solid #e0e0e0',
    'border-radius': '8px',
    'padding': '5px',
    'margin-bottom': '10px',
    'background-color': '#ffffff',
    'box-shadow': '0 1px 3px rgba(0,0,0,0.05)',
}

group_title_style = {
    'textAlign': 'center',
    'margin-bottom': '5px',
    'padding-bottom': '3px',
    'border-bottom': '1px solid #e0e0e0',
    'color': '#3498db',
    'font-size': '14px',
}

app.layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'width': '100%', 'fontFamily': 'Arial, sans-serif'},
    children=[
        # Left column: grouped and styled sliders
        html.Div(
            style={'width': '20%', 'padding': '10px', 'textAlign': 'center', 'background-color': '#f5f7fa', 'border-radius': '15px', 'margin': '10px'},
            children=[
                html.H2("Parameters", style={'textAlign': 'center', 'color': '#2c3e50', 'margin-bottom': '15px', 'font-size': '18px'}),
                
                # Main parameters group
                html.Div(style=group_style, children=[
                    html.Div("Main Parameters", style=group_title_style),
                    html.Div([
                        html.Label("s₀", style=label_style),
                        dcc.Slider(
                            id='s0_val', min=0.1, max=2.0, step=0.01, value=1,
                            marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                            className='custom-slider'
                        )
                    ], style=slider_style),

                    # html.Div([
                    #     html.Label("R", style=label_style),
                    #     dcc.Slider(
                    #         id='R_val', min=0.1, max=2.0, step=0.01, value=1.0,
                    #         marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                    #         tooltip={"placement": "bottom", "always_visible": True},
                    #         updatemode="drag"
                    #     )
                    # ], style=slider_style),
                    
                #     html.Div([
                #         html.Label("I₀", style=label_style),
                #         dcc.Slider(
                #             id='I0_val', min=0.1, max=2.0, step=0.01, value=0.5,
                #             marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                #             tooltip={"placement": "bottom", "always_visible": True},
                #             updatemode="drag"
                #         )
                #     ], style=slider_style),
                ]),
                
                # Currents group
                html.Div(style=group_style, children=[
                    html.Div("Currents", style=group_title_style),
                    html.Div([
                        html.Label("Iₗ", style=label_style),
                        dcc.Slider(
                            id='IL_val', min=-2.0, max=2.0, step=0.01, value=0,
                            marks={i: f'{i}' for i in [0.0, 0.5, 1.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                    
                    html.Div([
                        html.Label("Iₖ", style=label_style),
                        dcc.Slider(
                            id='IC_val', min=-2.0, max=2.0, step=0.01, value=0,
                            marks={i: f'{i}' for i in [0.0, 0.5, 1.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                    
                    html.Div([
                        html.Label("Iᵣ", style=label_style),
                        dcc.Slider(
                            id='IR_val', min=-2.0, max=2.0, step=0.01, value=0,
                            marks={i: f'{i}' for i in [0.0, 0.5, 1.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),

                    html.Div([
                        html.Label("I_I", style=label_style),
                        dcc.Slider(
                            id='I_I_val', min=0.0, max=1.0, step=0.01, value=0.25,
                            marks={i: f'{i}' for i in [0.0, 0.5, 1.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                ]),
                
                # s parameters group
                html.Div(style=group_style, children=[
                    html.Div("s Parameters", style=group_title_style),
                    html.Div([
                        html.Label("sₗ", style=label_style),
                        dcc.Slider(
                            id='sL_val', min=0, max=2.0, step=0.01, value=0,
                            marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                    
                    html.Div([
                        html.Label("sₖ", style=label_style),
                        dcc.Slider(
                            id='sC_val', min=0, max=2.0, step=0.01, value=0,
                            marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                    
                    html.Div([
                        html.Label("sᵣ", style=label_style),
                        dcc.Slider(
                            id='sR_val', min=0, max=2.0, step=0.01, value=0,
                            marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                ]),
                
                # Other parameters group
                html.Div(style=group_style, children=[
                    html.Div("Other Parameters", style=group_title_style),
                    html.Div([
                        html.Label("c", style=label_style),
                        dcc.Slider(
                            id='c_val', min=0.1, max=2.0, step=0.01, value=1.0,
                            marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                    
                    html.Div([
                        html.Label("g", style=label_style),
                        dcc.Slider(
                            id='g_val', min=0.1, max=2.0, step=0.01, value=1.0,
                            marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"
                        )
                    ], style=slider_style),
                    
                    # html.Div([
                    #     html.Label("R_I", style=label_style),
                    #     dcc.Slider(
                    #         id='R_I_val', min=0.1, max=2.0, step=0.01, value=1.0,
                    #         marks={i: f'{i}' for i in [0.1, 1.0, 2.0]},
                    #         tooltip={"placement": "bottom", "always_visible": True},
                    #         updatemode="drag"
                    #     )
                    # ], style=slider_style),
                   
                ]),
                html.Div(
                style={'width': '100%', 'padding': '0px'},
                children=[
                    html.Div(id='phi_values', style={
                        'padding': '10px', 
                        'fontSize': '12px', 
                        'textAlign': 'center',
                        'background-color': '#f8f9fa',
                        'border-radius': '8px',
                        'margin-top': '10px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'font-family': 'monospace'
                    }),
                    html.Div(id='sols_values', style={
                        'padding': '10px', 
                        'fontSize': '14px', 
                        'textAlign': 'center',
                        'background-color': '#f8f9fa',
                        'border-radius': '8px',
                        'margin-top': '10px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'font-family': 'monospace'
                    }),
            ]
        )
            ]
        ),
        # Right column: 3D plot and phi values
        html.Div(
            style={'width': '80%', 'padding': '10px'},
            children=[
                dcc.Graph(id='potential_plot', style={'height': '90vh'}),
                # html.Div(id='phi_values', style={
                #     'padding': '10px', 
                #     'fontSize': '16px', 
                #     'textAlign': 'center',
                #     'background-color': '#f8f9fa',
                #     'border-radius': '8px',
                #     'margin-top': '10px',
                #     'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                #     'font-family': 'monospace'
                # }),
                # html.Div(id='sols_values', style={
                #     'padding': '10px', 
                #     'fontSize': '16px', 
                #     'textAlign': 'center',
                #     'background-color': '#f8f9fa',
                #     'border-radius': '8px',
                #     'margin-top': '10px',
                #     'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                #     'font-family': 'monospace'
                # }),
            ]
        )
    ]
)

# Callback para actualizar el gráfico y los valores de phi según los sliders
@app.callback(
    [Output('potential_plot', 'figure'),
     Output('phi_values', 'children'),
     Output('sols_values', 'children')],
    [Input('s0_val', 'value'),
     Input('IL_val', 'value'),
     Input('IC_val', 'value'),
     Input('IR_val', 'value'),
     Input('I_I_val', 'value'),
     Input('sL_val', 'value'),
     Input('sC_val', 'value'),
     Input('sR_val', 'value'),
    #  Input('R_val', 'value'),
    # Input('R_I_val', 'value'),
    #  Input('I0_val', 'value'),
     Input('c_val', 'value'),
     Input('g_val', 'value')], 
     
)
def update_plot(s0_val, IL_val, IC_val, IR_val, I_I_val, sL_val, sC_val, sR_val, c_val, g_val):
    R_val, R_I_val, I0_val = solve_parameters(
        s0_val, c_val, g_val, I_I_val,
        guess=(0.5, 0.5, 0.5)  # ajusta si quieres
    )

    params = {
        "s0_val": s0_val,
        "IL_val": IL_val,
        "IC_val": IC_val,
        "IR_val": IR_val,
        "I_I_val": I_I_val,
        "sL_val": sL_val,
        "sC_val": sC_val,
        "sR_val": sR_val,
        "R_val": R_val,
        "I0_val": I0_val,
        "c_val": c_val,
        "g_val": g_val,
        "R_I_val": R_I_val,
    }
    fig, (phi_p, phi_pp, phi_ppp) = create_interactive_3d_plot(params)
    phi_text = f"φ'(X₀) = {phi_p:.2f}    φ''(X₀) = {phi_pp:.2f}    φ'''(X₀) = {phi_ppp:.2f}"
    sols_text = f"R = {R_val:.2f}    R_I = {R_I_val:.2f}    I₀ = {I0_val:.2f}"
    return fig, phi_text, sols_text

if __name__ == '__main__':
    app.run_server(debug=True)
