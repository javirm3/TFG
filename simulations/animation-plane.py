import sympy as sp

# Definimos las variables simbólicas
X1, X2 = sp.symbols('X1 X2')
tau, phi_p, phi_pp, phi_ppp, s0, c,g, phi_Ip = sp.symbols('tau phi_p phi_pp phi_ppp s0 c g phi_Ip')
I_L, I_C, I_R, s_L, s_C, s_R, R = sp.symbols('I_L I_C I_R s_L s_C s_R R')

# Definimos los componentes del campo vectorial (parte determinista, sin ruido)
F1 = -(
    (phi_p/2)*(I_L - I_C + X1*(s_L+s_C) + (R+X2)*(s_L-s_C)) +
    (s0*phi_pp/2)*(
         R*(s_L+s_C) + I_L + I_C + (2*(s0-c*g*phi_Ip))/(3*c*g*phi_Ip)*(I_L+I_C+I_R + R*(s_L+s_C+s_R))
    )*X1 +
    (s0*phi_pp/2)*(R*(s_L-s_C) + I_L - I_C)*X2 +
    s0**2 * phi_pp * X1*X2 +
    ((s0-c*g*phi_Ip)/(3*c*g*phi_Ip))* s0**3 * (phi_pp**2) * X1*(X1**2+3*X2**2) +
    (phi_ppp*s0**3/6)* X1*(X1**2+3*X2**2)
)

F2 = -3*(
    (phi_p/6)*(X1*(s_L-s_C) + X2*(s_L+s_C+4*s_R) + I_L + I_C - 2*I_R + R*(s_L+s_C-2*s_R)) +
    (s0*phi_pp/6)*(
         R*(s_L+s_C+4*s_R) + (I_L+I_C+4*I_R) + ((s0-c*g*phi_Ip)/(2*c*g*phi_Ip))*(I_L+I_C+I_R + R*(s_L+s_C+s_R))
    )*X2 +
    (s0*phi_pp/6)*(R*(s_L-s_C) + I_L - I_C)*X1 +
    (s0**2 * phi_pp/6)*(X1**2-3*X2**2) +
    ((s0-c*g*phi_Ip)/(3*c*g*phi_Ip))* s0**3 * (phi_pp**2) * X2*(X1**2+3*X2**2) +
    (phi_ppp*s0**3/6)* X2*(X1**2+3*X2**2)
)


curl = sp.simplify(sp.diff(F2, X1) - sp.diff(F1, X2))
print("Curl del campo:", curl)

U = sp.integrate(F1, X1)
U += sp.integrate(F2 - sp.diff(U, X2), X2)


# Simplificamos el potencial obtenido
U_simpl = sp.simplify(U)

# Mostramos el potencial (forma simbólica y en LaTeX)
print("Potencial U:")
print("\nForma LaTeX:")
print(sp.latex(U_simpl))
U_simpl

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import sympy as sp

x = sp.symbols('x', real=True)
s0, R, R_I, I0, c, I_I = sp.symbols('s0 R R_I I0 c I_I', real=True)

phi = sp.Piecewise(
    (0, x < 0),
    (x**2, sp.And(x >= 0, x <= 1)),
    (2*sp.sqrt(x) - sp.Rational(3, 4), x > 1)
)

phi_prime = sp.diff(phi, x)
phi_double_prime = sp.diff(phi_prime, x)
phi_triple_prime = sp.diff(phi_double_prime, x)



s0_val = 1
IL_val, IC_val, IR_val = 1,1,1
sL_val, sC_val, sR_val = 0,0,0
c_val = 1        
g_val = 1  
I_I_val = 1/5

X0 = s0*R - c*R_I + I0
X0_I = g*R+I_I


phi_X0 = sp.simplify(phi.subs(x, X0))
phi_prime_X0 = sp.simplify(phi_prime.subs(x, X0))
phi_double_prime_X0 = sp.simplify(phi_double_prime.subs(x, X0))
phi_triple_prime_X0 = sp.simplify(phi_triple_prime.subs(x, X0))

phi_X0 = sp.simplify(phi.subs(x, X0))
phi_prime_X0 = sp.simplify(phi_prime.subs(x, X0))
phi_I_prime = sp.simplify(phi_prime.subs(x, X0_I))
phi_I = sp.simplify(phi.subs(x, X0_I))

eq1 = s0*phi_prime_X0-1
eq2 = R - phi_X0 
eq3 = R_I - phi_I
eq1 = eq1.subs({s0:s0_val, c:c_val, g:g_val, I_I:I_I_val})
eq2 = eq2.subs({s0:s0_val, c:c_val, g:g_val, I_I:I_I_val})
eq3 = eq3.subs({s0:s0_val, c:c_val, g:g_val, I_I:I_I_val})
F_so = sp.lambdify(
    (R, R_I, I0),
    (eq1, eq2, eq3),
    'numpy'
)

def fun(vars):
    R_val, R_I_val, I0_val = vars
    return F_so(R_val, R_I_val, I0_val)

x0 = [1/4, 1/4, 1/2]
R_val, R_I_val, I0_val = fsolve(fun, x0)

print("Valores de R, R_I, I0:")
print(f"R = {R_val}, R_I = {R_I_val}, I0 = {I0_val}")

# Definición de X₀ (aunque en el potencial no se reemplace directamente, se utiliza para interpretar φ, etc.)
phi_p_val = sp.N(phi_prime_X0.subs({s0: s0_val, R: R_val, R_I: R_I_val,I0: I0_val, c: c_val}))
phi_pp_val = sp.N(phi_double_prime_X0.subs({s0: s0_val, R: R_val, R_I: R_I_val, I0: I0_val, c: c_val}))
phi_ppp_val = sp.N(phi_triple_prime_X0.subs({s0: s0_val, R: R_val, R_I: R_I_val, I0: I0_val, c: c_val}))
phi_Ip_val = sp.N(phi_I_prime.subs({s0: s0_val, R: R_val, R_I: R_I_val, I0: I0_val, c: c_val, g: g_val, I_I: I_I_val}))

# Diccionario de sustitución: reemplazamos φ, φ', φ''' por sus versiones evaluadas en X₀ (phi1, phi2, phi3)
subs_dict = {
    sp.symbols('phi_p'): phi_p_val, 
    sp.symbols('phi_pp'): phi_pp_val,
    sp.symbols('phi_ppp'): phi_ppp_val,
    sp.symbols('phi_Ip'): phi_Ip_val,
    sp.symbols('s0'): s0_val,
    sp.symbols('I0'): I0_val,
    sp.symbols('c'): c_val,
    sp.symbols('g'): g_val,
    sp.symbols('R'): R_val,
    sp.symbols('I_L'): IL_val,
    sp.symbols('I_C'): IC_val,
    sp.symbols('I_R'): IR_val,
    sp.symbols('s_L'): sL_val,
    sp.symbols('s_C'): sC_val,
    sp.symbols('s_R'): sR_val,
    sp.symbols('R_I'): R_I_val,
}

# Sustituir en la expresión del potencial y forzar la evaluación numérica
pot_expr_num = sp.N(U_simpl.subs(subs_dict))

# Crear la función numérica a partir de la expresión evaluada
potencial_num = sp.lambdify((sp.symbols('X1'), sp.symbols('X2')), pot_expr_num, 'numpy')


from scipy.optimize import fsolve
import pandas as pd
# ------------------------------------------
# Cálculo del gradiente (derivadas parciales)
# ------------------------------------------
F1 = sp.diff(U_simpl, X1)  # dU/dX1
F2 = sp.diff(U_simpl, X2)  # dU/dX2

# Sustituir los parámetros en las derivadas
F1_num_expr = sp.N(F1.subs(subs_dict))
F2_num_expr = sp.N(F2.subs(subs_dict))

# Convertir el gradiente en una función numérica
grad_func = sp.lambdify((X1, X2), [F1_num_expr, F2_num_expr], 'numpy')

# ------------------------------------------
# Resolver numéricamente grad_U = 0 usando fsolve
# ------------------------------------------
def grad_wrapper(x):
    return np.array(grad_func(x[0], x[1])).astype(float)

# Elegimos varios puntos iniciales (se esperan tres wells)
x_guess = np.linspace(-5, 5, 10)
y_guess = np.linspace(-5, 5, 10)
initial_guesses = np.array(np.meshgrid(x_guess, y_guess)).T.reshape(-1, 2)

critical_points_fsolve = []
for guess in initial_guesses:
    sol = fsolve(grad_wrapper, guess)
    sol_rounded = np.round(sol, decimals=6)  # redondeamos para evitar duplicados
    if not any(np.allclose(sol_rounded, cp) for cp in critical_points_fsolve):
        critical_points_fsolve.append(sol_rounded)

print("Critical points found with fsolve:")
for cp in critical_points_fsolve:
    print(cp)
print("Done with fsolve.")


Hessian = sp.hessian(U_simpl, (X1, X2))

print("\nHessian and classification of critical points (using sympy.solve results):")
for cp in critical_points_fsolve:
# Sustituir el punto crítico y los parámetros en la Hessiana
    H_cp = Hessian.subs({X1: cp[0], X2: cp[1]}).subs(subs_dict)
    eigenvals = H_cp.eigenvals()
    U_val = U_simpl.subs({X1: cp[0], X2: cp[1]}).subs(subs_dict)
    print("Critical point:", cp)
    print("Potential value:", U_val)
    print("Eigenvalues of Hessian:")
    sp.pprint(eigenvals)
    # Se considera mínimo si todos los autovalores son positivos
    is_min = all(val > 0 for val in eigenvals.keys())
    print("Is minimum:", is_min)
    print("-"*40)
# Create a dataframe with critical points and their classification

# Lists to store the classifications
point_coords = []
potential_vals = []
eigenvalue1 = []
eigenvalue2 = []
point_types = []

# Process each critical point
for cp in critical_points_fsolve:
    # Calculate Hessian eigenvalues at this point
    H_cp = Hessian.subs({X1: cp[0], X2: cp[1]}).subs(subs_dict)
    eigenvals_dict = H_cp.eigenvals()
    eigenvals_list = list(eigenvals_dict.keys())
    
    # Store potential value
    U_val = float(U_simpl.subs({X1: cp[0], X2: cp[1]}).subs(subs_dict))
    
    # Classify the critical point based on eigenvalues
    if all(val > 0 for val in eigenvals_list):
        point_type = "Minimum"
    elif all(val < 0 for val in eigenvals_list):
        point_type = "Maximum"
    else:
        point_type = "Saddle Point"
    
    # Append to lists
    point_coords.append(f"({cp[0]:.4f}, {cp[1]:.4f})")
    potential_vals.append(U_val)
    eigenvalue1.append(float(eigenvals_list[0]))
    if len(eigenvals_list) > 1:
        # If there are two eigenvalues, append the second one
        eigenvalue2.append(float(eigenvals_list[1]))
    else:
        # If only one eigenvalue, append None or 0
        eigenvalue2.append(float(eigenvals_list[0]))
    point_types.append(point_type)

# Create dataframe
critical_points = pd.DataFrame({
    'Critical Point': point_coords,
    'Potential Value': potential_vals,
    'Eigenvalue 1': eigenvalue1,
    'Eigenvalue 2': eigenvalue2,
    'Type': point_types
})


import numpy as np
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(10, 8))

# Crear una malla para X1 y X2
x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_vals, y_vals)

def find_critical_points(current_subs_dict):
    # Recalcular el potencial con los valores actuales
    pot_expr_num = sp.N(U_simpl.subs(current_subs_dict))
    
    # Calcular el gradiente usando sympy
    grad_U = [sp.diff(pot_expr_num, X1), sp.diff(pot_expr_num, X2)]
    
    # Resolver el sistema usando sympy
    critical_points_sym = sp.solve(grad_U, (X1, X2), dict=True)
    
    # Clasificar los puntos críticos
    classified_points = []
    for point in critical_points_sym:
        # Verificar si el punto es real
        if point[X1].is_real and point[X2].is_real:
            # Convertir el punto a coordenadas numéricas
            cp = np.array([float(sp.re(point[X1])), float(sp.re(point[X2]))])
            
            # Verificar si el punto está dentro de los límites razonables
            if abs(cp[0]) <= 10 and abs(cp[1]) <= 10:
                # Calcular y clasificar usando la Hessiana
                H_cp = Hessian.subs({X1: cp[0], X2: cp[1]}).subs(current_subs_dict)
                eigenvals = H_cp.eigenvals()
                eigenvals_list = list(eigenvals.keys())
                
                if all(val > 0 for val in eigenvals_list):
                    point_type = "Minimum"
                elif all(val < 0 for val in eigenvals_list):
                    point_type = "Maximum"
                else:
                    point_type = "Saddle Point"
                    
                classified_points.append((cp, point_type))
    
    return classified_points

# Función para actualizar el frame
def update(frame):
    ax.clear()
    
    # Calcular el valor de I basado en el frame
    I_val = -2 + (frame / 50) * 4  # Va de -2 a 2 en 100 frames
    
    # Actualizar los valores de las corrientes
    current_subs_dict = subs_dict.copy()
    current_subs_dict.update({
        sp.symbols('I_L'): I_val,
        sp.symbols('I_C'): I_val,
        sp.symbols('I_R'): I_val
    })
    
    # Recalcular el potencial con los nuevos valores
    pot_expr_num = sp.N(U_simpl.subs(current_subs_dict))
    potencial_num = sp.lambdify((sp.symbols('X1'), sp.symbols('X2')), pot_expr_num, 'numpy')
    
    # Calcular Z y limitar su rango
    Z = potencial_num(X, Y)
    Z = np.clip(Z, -50, 100)
    
    # Crear el contour plot
    contour = ax.contourf(X, Y, Z, 30, cmap='viridis')
    ax.contour(X, Y, Z, 15, colors='white', linewidths=0.5, alpha=0.7)
    
    # Encontrar y marcar los puntos críticos
    critical_points = find_critical_points(current_subs_dict)
    
    # Diccionario de colores para los diferentes tipos de puntos
    colors = {'Maximum': 'red', 'Minimum': 'blue', 'Saddle Point': 'green'}
    
    # Plotear los puntos críticos
    for cp, point_type in critical_points:
        ax.scatter(cp[0], cp[1], c=colors[point_type], s=100, marker='o', label=point_type)
    
    # Añadir etiquetas y título
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    text_I = f"$I_L=I_R=I_C={I_val:.2f}$"
    ax.text(0.05, -0.1, text_I, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.set_title('Potential Landscape $\\psi(X_1, X_2)$')
    
    # Mantener los límites constantes
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    
    # Añadir leyenda única para los tipos de puntos
    handles = [plt.scatter([], [], c=color, label=label) for label, color in colors.items()]
    ax.legend(handles=handles)
    
    return contour,

# Crear la animación
anim = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# Guardar la animación
anim.save('potential_landscape_animation.gif', writer='pillow', fps=10)
plt.close()