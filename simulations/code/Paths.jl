# paths.jl
# ---------------------------------------------
# Define rutas absolutas y relativas del proyecto
module Paths
using Dates
using FilePathsBase: abspath

# Ruta base del proyecto (nivel raíz)
const ROOT_PATH = abspath(joinpath(@__DIR__, "..", ".."))

# Subcarpetas
const DATA_PATH    = joinpath(ROOT_PATH, "simulations", "datasets")
const CODE_PATH    = joinpath(ROOT_PATH, "simulations", "code")
const RESULTS_PATH = joinpath(ROOT_PATH, "simulations", "results")

# Rutas específicas
const CSV_PATH = joinpath(DATA_PATH, "df_filtered.csv")

# Utilidad: imprime el resumen de rutas
function show_paths()
    println("📂 Project root:   ", ROOT_PATH)
    println("📊 Dataset path:   ", DATA_PATH)
    println("🧩 Code path:      ", CODE_PATH)
    println("💾 Results path:   ", RESULTS_PATH)
end

end # module Paths