import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import sympy as sp
import plotly.graph_objs as go

# ——————————————————————————————————————————————
# 1) Definición de funciones base (drift, simulador, extracción r123)
# ——————————————————————————————————————————————
# (Asume que X1, X2, subs_dict2, IL_val, IC_val, IR_val, F1, F2, V_expr ya están definidos en tu entorno)

# Lambdify del campo de deriva
F1_num = sp.lambdify((X1, X2, I_L, I_C, I_R), F1.subs(subs_dict2), 'numpy')
F2_num = sp.lambdify((X1, X2, I_L, I_C, I_R), F2.subs(subs_dict2)/2, 'numpy')

def drift(X, I_L, I_C, I_R):
    x, y = X
    return -np.array([F1_num(x, y, I_L, I_C, I_R),
                      F2_num(x, y, I_L, I_C, I_R)/2])

# Funciones de input
def U_t(t, onset=0.500, offset=1.5):
    return np.where((t < onset) | (t > onset + offset),
                    -1, 2 / offset * (t - onset) - 1)

def stim_t(t, onset=0.20, offset=1.8, amplitude=0.4):
    return np.where((t < onset) | (t > onset + offset),
                    0, amplitude)

# Simulación de un solo trial
def simulate_path(x0, dt, Tmax, noise_amp=1):
    N = int(Tmax / dt)
    X = np.zeros((N+1, 2))
    X[0] = x0
    for i in range(N):
        x, y = X[i]
        dW = np.random.randn(3) * np.sqrt(dt)
        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2*dW[2]) / 6
        U = U_t(i*dt)
        S = stim_t(i*dt)
        X[i+1] = X[i] + drift(X[i], I_L=U+S, I_C=U, I_R=U) * dt + noise_amp * np.array([dB1, dB2])
    # Determinar ganador
    r1 = X[:,0] + X[:,1]
    r2 = -X[:,0] + X[:,1]
    r3 = -2 * X[:,1]
    final = np.argmax([r1[-1], r2[-1], r3[-1]])
    win = ['r1','r2','r3'][final]
    return X, r1, r2, r3, win

# Extracción r123
def extract_r123(traj):
    X1_vals, X2_vals = traj[:,0], traj[:,1]
    return X1_vals + X2_vals, -X1_vals + X2_vals, -2 * X2_vals

# ——————————————————————————————————————————————
# 2) Configuración inicial
# ——————————————————————————————————————————————
dt = 1e-3
Tmax = 3
x0 = np.array([0,0])

# ——————————————————————————————————————————————
# 3) DASH app
# ——————————————————————————————————————————————
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Explorador de Trayectorias Estocásticas"),
    html.Label("Número de trials:"),
    dcc.Slider(id='n-trajs-slider', min=10, max=1000, step=10, value=100,
               marks={i: str(i) for i in range(0,1001,200)}),
    dcc.Graph(id='avg-graph')
])

@app.callback(
    Output('avg-graph', 'figure'),
    [Input('n-trajs-slider', 'value')]
)
def update_graph(n_trajs):
    # Simular n_trajs trials y agrupar
    r1_trials, r2_trials, r3_trials = [], [], []
    for _ in range(n_trajs):
        traj, r1, r2, r3, win = simulate_path(x0, dt, Tmax)
        if win == 'r1':
            r1_trials.append(r1)
        elif win == 'r2':
            r2_trials.append(r2)
        else:
            r3_trials.append(r3)

    # Alinear longitud mínima
    min_len = min(len(r) for group in [r1_trials, r2_trials, r3_trials] for r in group)
    t = np.linspace(0, Tmax*1000, min_len)

    # Función para trazar promedio ± std
    def make_trace(data, name, color):
        arr = np.array([r[:min_len] for r in data])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        return [go.Scatter(x=t, y=mean, mode='lines', name=name, line=dict(color=color)),
                go.Scatter(x=t, y=mean+std, mode='lines', showlegend=False, line=dict(color=color, dash='dash')),
                go.Scatter(x=t, y=mean-std, mode='lines', showlegend=False, line=dict(color=color, dash='dash'))]

    traces = []
    if r1_trials: traces += make_trace(r1_trials, 'r_L', 'red')
    if r2_trials: traces += make_trace(r2_trials, 'r_C', 'green')
    if r3_trials: traces += make_trace(r3_trials, 'r_R', 'blue')

    # Inputs
    U_vals = U_t(np.linspace(0, Tmax, min_len))
    S_vals = stim_t(np.linspace(0, Tmax, min_len))
    traces += [
        go.Scatter(x=t, y=U_vals, mode='lines', name='U(t)', line=dict(color='black', dash='dot')),
        go.Scatter(x=t, y=S_vals, mode='lines', name='stim(t)', line=dict(color='orange', dash='dot'))
    ]

    layout = go.Layout(
        xaxis=dict(title='Time (ms)', showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(title='r_i', showgrid=True, gridcolor='lightgrey'),
        title=f'Trayectorias promedio (n_trajs={n_trajs})',
        legend=dict(orientation='h', x=0, y=1.1)
    )

    return go.Figure(data=traces, layout=layout)

if __name__ == '__main__':
    app.run_server(debug=True)
