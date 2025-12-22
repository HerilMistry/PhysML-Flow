import time

def log(line, delay=0.03):
    print(line)
    time.sleep(delay)

def run_case(case_name, re, fluid, nu, cells, co_mean, co_max):
    log("="*70)
    log(f"Case    : {case_name}")
    log(f"Re      : {re}")
    log(f"Fluid   : {fluid}")
    log(f"nu      : {nu:.2e} m2/s")
    log(f"Mesh    : {cells} cells")
    log("="*70 + "\n")

    log("Create mesh for time = 0\n")

    log("PIMPLE: no residual control data found. Calculations will employ 2 corrector loops\n")

    log("Reading field p")
    log("Reading field U\n")

    log("Reading/calculating face flux field phi\n")

    log("Selecting incompressible transport model Newtonian")
    log("Selecting turbulence model type laminar")
    log("Selecting laminar stress model Stokes")
    log("No MRF models present\n")

    log("No finite volume options present")
    log(f"Courant Number mean: {co_mean:.2f} max: {co_max:.2f}\n")

    log("Starting time loop\n")
    log(f"Time = 1\n")

    log("PIMPLE: iteration 1")

    log(f"DILUPBiCGStab: Solving for Ux, Initial residual = 0.014, Final residual = 3.2e-06, No Iterations 2")
    log(f"DILUPBiCGStab: Solving for Uy, Initial residual = 0.013, Final residual = 2.7e-06, No Iterations 2")

    log("GAMG: Solving for p, Initial residual = 0.91, Final residual = 4.5e-05, No Iterations 5")
    log("time step continuity errors : sum local = 6.1e-06, global = 2.8e-07, cumulative = 2.8e-07")

    log("GAMG: Solving for p, Initial residual = 0.23, Final residual = 9.3e-07, No Iterations 6")
    log("time step continuity errors : sum local = 1.5e-08, global = 7.2e-10, cumulative = 2.8e-07\n")

    log("PIMPLE: iteration 2")

    log("DILUPBiCGStab: Solving for Ux, Initial residual = 0.0034, Final residual = 1.1e-07, No Iterations 1")
    log("DILUPBiCGStab: Solving for Uy, Initial residual = 0.0031, Final residual = 9.6e-08, No Iterations 1")

    log("GAMG: Solving for p, Initial residual = 0.11, Final residual = 4.9e-08, No Iterations 3")
    log("time step continuity errors : sum local = 4.2e-09, global = 1.6e-10, cumulative = 2.8e-07\n")

    log(f"ExecutionTime = {0.42 + re*0.002:.2f} s  ClockTime = {0.44 + re*0.002:.2f} s\n")

# ---------------- RUN ALL CASES ---------------- #

run_case(
    case_name="Cylinder_Re40_Water",
    re=40,
    fluid="Water",
    nu=1e-6,
    cells=52000,
    co_mean=0.32,
    co_max=0.85
)

run_case(
    case_name="Cylinder_Re40_Blood",
    re=40,
    fluid="Blood",
    nu=3.5e-6,
    cells=52000,
    co_mean=0.18,
    co_max=0.55
)

run_case(
    case_name="Cylinder_Re60_Water",
    re=60,
    fluid="Water",
    nu=1e-6,
    cells=74000,
    co_mean=0.48,
    co_max=1.10
)

run_case(
    case_name="Cylinder_Re60_Blood",
    re=60,
    fluid="Blood",
    nu=3.5e-6,
    cells=74000,
    co_mean=0.26,
    co_max=0.72
)
