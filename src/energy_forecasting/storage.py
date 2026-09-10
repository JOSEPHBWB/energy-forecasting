from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class BatterySpec:
    capacity_mwh: float = 4.0
    max_power_mw: float = 1.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc_mwh: float = 2.0
    terminal_soc_mwh: float = 2.0


def optimize_storage_dispatch(price_forecast, spec: BatterySpec = BatterySpec()):
    """Solve a finite-horizon battery arbitrage MILP.

    Binary mode variables prohibit simultaneous charging and discharging.
    The objective maximizes forecast-price arbitrage revenue while respecting
    state-of-charge, power, efficiency, and terminal-state constraints.
    """
    p = np.asarray(price_forecast, dtype=float)
    T = len(p)
    if T == 0:
        raise ValueError("price_forecast must be non-empty")

    # Variable blocks: charge[T], discharge[T], soc[T], mode[T]
    n = 4 * T
    C, D, S, Z = 0, T, 2 * T, 3 * T

    # scipy.optimize.milp minimizes c @ x.
    c = np.zeros(n)
    c[C:C + T] = p
    c[D:D + T] = -p

    lb = np.zeros(n)
    ub = np.empty(n)
    ub[C:C + T] = spec.max_power_mw
    ub[D:D + T] = spec.max_power_mw
    ub[S:S + T] = spec.capacity_mwh
    ub[Z:Z + T] = 1.0
    bounds = Bounds(lb, ub)

    integrality = np.zeros(n, dtype=int)
    integrality[Z:Z + T] = 1

    # Equalities: SOC dynamics plus terminal SOC.
    Aeq = lil_matrix((T + 1, n), dtype=float)
    beq = np.zeros(T + 1)
    for t in range(T):
        Aeq[t, S + t] = 1.0
        Aeq[t, C + t] = -spec.charge_efficiency
        Aeq[t, D + t] = 1.0 / spec.discharge_efficiency
        if t > 0:
            Aeq[t, S + t - 1] = -1.0
        else:
            beq[t] = spec.initial_soc_mwh
    Aeq[T, S + T - 1] = 1.0
    beq[T] = spec.terminal_soc_mwh

    # Mode constraints: charge <= Pmax*z; discharge <= Pmax*(1-z).
    Aub = lil_matrix((2 * T, n), dtype=float)
    bub = np.zeros(2 * T)
    for t in range(T):
        Aub[2 * t, C + t] = 1.0
        Aub[2 * t, Z + t] = -spec.max_power_mw
        bub[2 * t] = 0.0
        Aub[2 * t + 1, D + t] = 1.0
        Aub[2 * t + 1, Z + t] = spec.max_power_mw
        bub[2 * t + 1] = spec.max_power_mw

    constraints = [
        LinearConstraint(Aeq.tocsr(), beq, beq),
        LinearConstraint(Aub.tocsr(), -np.inf, bub),
    ]
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"storage MILP failed: {res.message}")

    x = res.x
    return {
        "charge_mw": x[C:C + T],
        "discharge_mw": x[D:D + T],
        "soc_mwh": x[S:S + T],
        "mode": np.rint(x[Z:Z + T]).astype(int),
        "forecast_objective_revenue": float(-res.fun),
    }


def realized_revenue(dispatch, realized_price) -> float:
    price = np.asarray(realized_price, dtype=float)
    charge = np.asarray(dispatch["charge_mw"], dtype=float)
    discharge = np.asarray(dispatch["discharge_mw"], dtype=float)
    return float(np.sum(price * (discharge - charge)))
