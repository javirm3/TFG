using BifurcationKit, Plots
using Accessors: @optic
using LinearAlgebra

"""
Sistema dinámico que representa las ecuaciones del modelo.
"""
function F(u, p)
    x1, x2 = u
    I_L = p.I_L
    I_C = p.I_C
    I_R = p.I_R

    # Primera ecuación del sistema
    expr1 = 0.5*I_C - 0.5*I_L - 2*x1*x2 -
            4/3*x1*(x1^2 + 3*x2^2) -
            x1*(5/3*I_C + 5/3*I_L + 2/3*I_R) -
            x2*(-I_C + I_L)

    # Segunda ecuación del sistema
    expr2 = -0.5*I_C - 0.5*I_L + I_R -
            x1^2 - x1*(-I_C + I_L) +
            3*x2^2 - 4*x2*(x1^2 + 3*x2^2) -
            x2*(1.5*I_C + 1.5*I_L + 4.5*I_R)

    return [expr1, expr2]
end

"""
Jacobiano del sistema para mejorar la convergencia.
"""
function J(u, p)
    x1, x2 = u
    I_L = p.I_L
    I_C = p.I_C
    I_R = p.I_R
    
    # Derivadas parciales
    ∂f1_∂x1 = -2*x2 - 4*(x1^2 + 3*x2^2) - 4*x1^2 - 
              (5/3*I_C + 5/3*I_L + 2/3*I_R)
    
    ∂f1_∂x2 = -2*x1 - 12*x1*x2 - (-I_C + I_L)
    
    ∂f2_∂x1 = -2*x1 - (-I_C + I_L) - 8*x2*x1
    
    ∂f2_∂x2 = 6*x2 - 4*(x1^2 + 3*x2^2) - 12*x2^2 - 
              (1.5*I_C + 1.5*I_L + 4.5*I_R)
    
    return [
        ∂f1_∂x1  ∂f1_∂x2
        ∂f2_∂x1  ∂f2_∂x2
    ]
end

# Función para registrar la solución que maneja correctamente los kwargs
function record_solution(x, p; kwargs...)
    return (x₁ = x[1], x₂ = x[2])
end

# Función para calcular una rama de bifurcación con parámetros específicos
function calcular_rama(valor_inicial, condicion_inicial; max_pasos=1000, tolerancia=1e-6)
    # Configuración inicial del problema
    params = (I_L=valor_inicial, I_C=valor_inicial, I_R=valor_inicial)
    
    # Definición del problema de bifurcación
    prob = BifurcationProblem(
        F,                                  # Función del sistema
        condicion_inicial,                  # Condición inicial
        params,                             # Parámetros
        (@optic _.I_L);                     # Parámetro de bifurcación
        J = J,                              # Jacobiano del sistema
        record_from_solution = record_solution
    )
    
    # Opciones para la continuación numérica
    opts = ContinuationPar(
        ds = 0.002,                         # Tamaño de paso inicial muy pequeño
        dsmin = 1e-6,                       # Tamaño de paso mínimo muy pequeño
        dsmax = 0.01,                       # Tamaño de paso máximo pequeño
        p_min = -2.0,                       # Valor mínimo del parámetro
        p_max = 2.0,                        # Valor máximo del parámetro
        max_steps = max_pasos,              # Número máximo de pasos
        detect_bifurcation = 3,             # Nivel de detección de bifurcaciones
        nev = 2,                            # Número de valores propios a calcular
        newton_options = NewtonPar(         # Opciones para el método de Newton
            tol = tolerancia,               # Tolerancia personalizable
            max_iterations = 50             # Más iteraciones para mejor convergencia
        ),
        save_eigenvectors = true            # Guardar vectores propios para bifurcaciones
    )
    
    # Cálculo de la rama de soluciones (sin bothside para evitar problemas)
    try
        return continuation(prob, PALC(), opts)
    catch e
        println("Error en cálculo de rama: $e")
        # Intentar con un método alternativo si PALC falla
        return continuation(prob, Natural(), opts)
    end
end

# Calcular múltiples ramas con diferentes condiciones iniciales
println("Calculando ramas...")

# Conjunto de condiciones iniciales para explorar el espacio de soluciones
condiciones_iniciales = [
    (0.0, [0.1, 0.0]),    # Rama 1: cerca del origen, x1 positivo
    (0.0, [-0.1, 0.0]),   # Rama 2: cerca del origen, x1 negativo
    (0.0, [0.0, 0.1]),    # Rama 3: cerca del origen, x2 positivo
    (0.0, [0.0, -0.1]),   # Rama 4: cerca del origen, x2 negativo
    (0.5, [0.5, 0.0]),    # Rama 5: lejos del origen en x1
    (0.5, [0.0, 0.5]),    # Rama 6: lejos del origen en x2
    (0.5, [0.5, 0.5]),    # Rama 7: lejos del origen en ambas direcciones
    (1.0, [1.0, 0.0]),    # Rama 8: más lejos en x1
    (1.0, [0.0, 1.0]),    # Rama 9: más lejos en x2
    (-0.5, [0.5, 0.0]),   # Rama 10: parámetro negativo
]

# Calcular todas las ramas
all_branches = []
for (i, (param, cond)) in enumerate(condiciones_iniciales)
    println("Calculando rama $i con parámetro=$param, condición=$cond")
    try
        # Intentar con diferentes tolerancias si es necesario
        branch = calcular_rama(param, cond, max_pasos=800, tolerancia=1e-6)
        push!(all_branches, branch)
    catch e
        println("Error en rama $i: $e")
        try
            # Segundo intento con tolerancia más relajada
            branch = calcular_rama(param, cond, max_pasos=500, tolerancia=1e-5)
            push!(all_branches, branch)
        catch e2
            println("Segundo intento fallido para rama $i: $e2")
        end
    end
end

# Filtrar ramas vacías o problemáticas
all_branches = filter(br -> length(br) > 5, all_branches)

# Crear el gráfico de bifurcación con todas las ramas
println("Creando gráfico...")
colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :brown, :black, :darkgreen]
p = plot(size=(1000, 800), grid=true, dpi=300)

for (i, branch) in enumerate(all_branches)
    color = colors[mod1(i, length(colors))]
    
    # Extraer datos de la rama
    βs = branch.param
    x1s = [get_solx(branch,i)[1] for i in 1:length(branch)]
    x2s = [get_solx(branch,i)[2] for i in 1:length(branch)]
    
    # Graficar componente x1
    plot!(p, βs, x1s, 
        label="x₁ rama $i", 
        color=color,
        linewidth=2
    )
    
    # Graficar componente x2
    plot!(p, βs, x2s, 
        label="x₂ rama $i", 
        color=color,
        linestyle=:dash,
        linewidth=2
    )
    
    # Añadir puntos de bifurcación
    if length(branch.specialpoint) > 0
        bif_points = [(sp.param, sp.x[1]) for sp in branch.specialpoint if sp.type == :bp]
        if !isempty(bif_points)
            scatter!(p, [p[1] for p in bif_points], 
                    [p[2] for p in bif_points],
                    label="BP rama $i",
                    marker=:star,
                    markersize=8,
                    color=color)
        end
    end
end

# Configuración final del gráfico
xlabel!(p, "β")
ylabel!(p, "x")
title!(p, "Diagrama de Bifurcación Completo")

# Guardar el gráfico con alta resolución
savefig(p, "bifurcation_diagram_complete.png")

# Imprimir información sobre puntos de bifurcación
println("\nInformación de bifurcaciones:")
for (i, branch) in enumerate(all_branches)
    println("\nRama $i:")
    for (j, sp) in enumerate(branch.specialpoint)
        println("  Punto $j: β = $(sp.param), tipo = $(sp.type)")
    end
end
