import math
import numpy as np
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget  
# =============================================================================
# DEFINICIÓN DE LAS FUNCIONES DEL POTENCIAL
# =============================================================================

def phi_p_fun(I0, R, s0, c, R_I):
    if I0 + R * s0 - R_I * c < 0:
        return 0
    elif I0 + R * s0 - R_I * c <= 1:
        return 2 * I0 + 2 * R * s0 - 2 * R_I * c
    else:
        return 1 / math.sqrt(I0 + R * s0 - R_I * c)

def phi_pp_fun(I0, R, s0, c, R_I):
    if I0 + R * s0 - R_I * c < 0:
        return 0
    elif I0 + R * s0 - R_I * c <= 1:
        return 2
    else:
        return -(1 / 2) / (I0 + R * s0 - R_I * c) ** (3 / 2)

def phi_ppp_fun(I0, R, s0, c, R_I):
    if I0 + R * s0 - R_I * c <= 1:
        return 0
    else:
        return (3 / 4) / (I0 + R * s0 - R_I * c) ** (5 / 2)

def U(X1, X2, phi_p, phi_pp, phi_ppp, phi_Ip, s0, I0, c, g, R,
      I_L, I_C, I_R, s_L, s_C, s_R, R_I):
    # Se recalculan las derivadas en función de los parámetros.
    phi_p = phi_p_fun(I0, R, s0, c, R_I)
    phi_pp = phi_pp_fun(I0, R, s0, c, R_I)
    phi_ppp = phi_ppp_fun(I0, R, s0, c, R_I)
    return -(1/24) * (
        X1**4 * s0**3 * (-2 * c * g * phi_Ip * phi_pp**2 + c * g * phi_Ip * phi_ppp + 2 * phi_pp**2 * s0) +
        2 * X1**2 * (
            I_C * c * g * phi_Ip * phi_pp * s0 + 2 * I_C * phi_pp * s0**2 +
            I_L * c * g * phi_Ip * phi_pp * s0 + 2 * I_L * phi_pp * s0**2 -
            2 * I_R * c * g * phi_Ip * phi_pp * s0 + 2 * I_R * phi_pp * s0**2 +
            R * c * g * phi_Ip * phi_pp * s0 * s_C + R * c * g * phi_Ip * phi_pp * s0 * s_L -
            2 * R * c * g * phi_Ip * phi_pp * s0 * s_R + 2 * R * phi_pp * s0**2 * s_C +
            2 * R * phi_pp * s0**2 * s_L + 2 * R * phi_pp * s0**2 * s_R -
            6 * X2**2 * c * g * phi_Ip * phi_pp**2 * s0**3 + 3 * X2**2 * c * g * phi_Ip * phi_ppp * s0**3 +
            6 * X2**2 * phi_pp**2 * s0**4 + 6 * X2 * c * g * phi_Ip * phi_pp * s0**2 +
            3 * c * g * phi_Ip * phi_p * s_C + 3 * c * g * phi_Ip * phi_p * s_L
        ) +
        9 * X2**4 * s0**3 * (-2 * c * g * phi_Ip * phi_pp**2 + c * g * phi_Ip * phi_ppp + 2 * phi_pp**2 * s0) +
        3 * X2**2 * (
            I_C * c * g * phi_Ip * phi_pp * s0 + I_C * phi_pp * s0**2 +
            I_L * c * g * phi_Ip * phi_pp * s0 + I_L * phi_pp * s0**2 +
            7 * I_R * c * g * phi_Ip * phi_pp * s0 + I_R * phi_pp * s0**2 +
            R * c * g * phi_Ip * phi_pp * s0 * s_C + R * c * g * phi_Ip * phi_pp * s0 * s_L +
            7 * R * c * g * phi_Ip * phi_pp * s0 * s_R + R * phi_pp * s0**2 * s_C +
            R * phi_pp * s0**2 * s_L + R * phi_pp * s0**2 * s_R +
            2 * c * g * phi_Ip * phi_p * s_C + 2 * c * g * phi_Ip * phi_p * s_L + 8 * c * g * phi_Ip * phi_p * s_R
        ) +
        12 * c * g * phi_Ip * (
            -X1 * (I_C * X2 * phi_pp * s0 + I_C * phi_p - I_L * X2 * phi_pp * s0 - I_L * phi_p +
                   R * X2 * phi_pp * s0 * s_C - R * X2 * phi_pp * s0 * s_L + R * phi_p * s_C - R * phi_p * s_L +
                   X2 * phi_p * s_C - X2 * phi_p * s_L) -
            X2**3 * phi_pp * s0**2 + X2 * phi_p * (I_C + I_L - 2 * I_R + R * s_C + R * s_L - 2 * R * s_R)
        )
    ) / (c * g * phi_Ip)

# =============================================================================
# INTERFAZ SHINY PARA PYTHON CON PLOTLY
# =============================================================================

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider("s0_val", "s₀", 0.1, 5, 1, step=0.1),
        ui.input_slider("I0_val", "I₀", 0.1, 5, 1, step=0.1),
        ui.input_slider("c_val", "c", 0.1, 5, 1, step=0.1),
        ui.input_slider("g_val", "g", 0.1, 5, 1, step=0.1),
        ui.input_slider("R_val", "R", 0.1, 5, 1, step=0.05),
        ui.input_slider("IL_val", "Iₗ", 0, 10, 0.5, step=0.1),
        ui.input_slider("IC_val", "Iₖ", 0, 10, 0.5, step=0.1),
        ui.input_slider("IR_val", "Iᵣ", 0, 10, 0.5, step=0.1),
        ui.input_slider("sL_val", "sₗ", 0.1, 5, 1, step=0.1),
        ui.input_slider("sC_val", "sₖ", 0.1, 5, 1, step=0.1),
        ui.input_slider("sR_val", "sᵣ", 0.1, 5, 1, step=0.1),
        ui.input_slider("R_I_val", "R_I", 0.1, 5, 1, step=0.05),
        ui.input_slider("phi_Ip_val", "φ_Ip", 0.1, 5, 1, step=0.1)
    ),
    output_widget("plot"),  
)

def server(input, output, session):
    @reactive.Calc
    def params():
        return {
            "s0_val": input.s0_val(),
            "I0_val": input.I0_val(),
            "c_val": input.c_val(),
            "g_val": input.g_val(),
            "R_val": input.R_val(),
            "IL_val": input.IL_val(),
            "IC_val": input.IC_val(),
            "IR_val": input.IR_val(),
            "sL_val": input.sL_val(),
            "sC_val": input.sC_val(),
            "sR_val": input.sR_val(),
            "R_I_val": input.R_I_val(),
            "phi_Ip_val": input.phi_Ip_val()
        }
    
    @render_widget
    def pot_plot():
        p = params()
        # Crear malla para X₁ y X₂
        x_vals = np.linspace(-3, 3, 200)
        y_vals = np.linspace(-3, 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluar la función potencial U en cada punto
        U_vectorized = np.vectorize(lambda x1, x2: U(
            x1, x2, None, None, None, p["phi_Ip_val"],
            p["s0_val"], p["I0_val"], p["c_val"], p["g_val"], p["R_val"],
            p["IL_val"], p["IC_val"], p["IR_val"],
            p["sL_val"], p["sC_val"], p["sR_val"], p["R_I_val"]
        ))
        Z = U_vectorized(X, Y)
        Z = np.clip(Z, -50, 100)
        
        # Crear el gráfico de superficie con Plotly
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
        
        caption = (f"Parámetros: s₀={p['s0_val']}, I₀={p['I0_val']}, c={p['c_val']}, g={p['g_val']}, R={p['R_val']}, "
                   f"Iₗ={p['IL_val']}, Iₖ={p['IC_val']}, Iᵣ={p['IR_val']}, sₗ={p['sL_val']}, sₖ={p['sC_val']}, "
                   f"sᵣ={p['sR_val']}, R_I={p['R_I_val']}, φ_Ip={p['phi_Ip_val']}")
        
        fig.update_layout(
            title="Potencial U",
            scene=dict(
                xaxis_title="X₁",
                yaxis_title="X₂",
                zaxis_title="U(X₁, X₂)"
            ),
            annotations=[
                dict(
                    text=caption,
                    xref="paper", yref="paper",
                    x=0, y=-0.1, showarrow=False
                )
            ]
        )
        # Convertir la figura a un widget de Plotly
        return fig.to_widget()


app = App(app_ui, server)

if __name__ == '__main__':
    app.run()
