from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import math
import random
import bisect


NumberOrSeries = Union[float, Sequence[float]]


def at(value: NumberOrSeries, t: int) -> float:
    """Return value at period t (1-indexed)."""
    if isinstance(value, (int, float)):
        return float(value)
    if t < 1 or t > len(value):
        raise IndexError(f"t={t} out of range for series with length {len(value)}")
    return float(value[t - 1])


@dataclass(frozen=True)
class PlantParams:
    plant_id: str
    base_type: str  # "legacy" or "candidate"
    K_bar: NumberOrSeries  # design capacity per period
    c_mfg: float  # unit manufacturing cost
    chi_by_regime: Dict[int, float]  # throughput factor in (0,1]
    ramp_kappa: float = 0.0  # candidate only, rho(age)=1-exp(-kappa*age)

    @property
    def is_candidate(self) -> bool:
        return self.base_type.lower() == "candidate"


@dataclass(frozen=True)
class SupplierParams:
    supplier_id: str
    W_bar: NumberOrSeries  # supply capacity per period
    c_mat_base: float  # base material unit cost (supplying legacy base)
    delta_rel: float  # relationship premium for candidate base during early periods


@dataclass(frozen=True)
class MarketParams:
    market_id: str
    d0: NumberOrSeries  # baseline demand scale absent tariff pressure
    cu: float  # unit cost of demand loss (used when potential > realised)
    price: float  # uniform selling price p_k for this market


@dataclass(frozen=True)
class ReconfigParams:
    activation_plant: str  # candidate plant id
    F: float  # fixed investment cost, paid when plant becomes operational (t+1)
    qual_G: float  # qualification / deposit fixed cost during early window after activation
    qual_ell: int  # length of qualification window (periods after activation)
    withdrawal_plant: str  # legacy plant id
    S_salv: float  # salvage credit when withdrawing legacy capacity
    terminal_tail_horizon: int = 8  # continuation-value look-ahead beyond the planning horizon
    terminal_tail_weight: float = 1.0  # weight on continuation value


@dataclass(frozen=True)
class RouteParams:
    """Minimal route representation to satisfy route-based multi-stop tour assumption.

    visits is an ordered list of nodes visited on the tour (excluding the depot plant).
    """
    route_id: str
    visits: List[str]
    dist_km: float
    vehicle_cap: float
    cost_per_km: float


@dataclass
class SupplyChainScenario:
    """Scenario container for the Chapter 3 and 4 model.

    This class only configures the scenario and provides helper methods.
    It does not solve the operational LP and does not run RL training.
    """

    H: int
    gamma: float

    suppliers: Dict[str, SupplierParams]
    plants: Dict[str, PlantParams]
    markets: Dict[str, MarketParams]

    legacy_plant_id: str
    candidate_plant_id: str
    core_market_id: str
    additional_market_id: str

    regimes: List[int]
    P: List[List[float]]
    xi_1: int
    xi_2_forced: Optional[int] = None

    activation_lag: int = 1
    initial_a: Optional[Dict[str, int]] = None
    initial_age: int = 0

    alpha_by_plant: Optional[Dict[str, float]] = None
    c_in: Optional[Dict[str, Dict[str, float]]] = None
    c_out: Optional[Dict[str, Dict[str, float]]] = None
    tau: Optional[Dict[str, Dict[str, Dict[int, float]]]] = None
    tau_time_series: Optional[Dict[str, Dict[str, List[float]]]] = None

    H_rel: int = 10
    reconfig: Optional[ReconfigParams] = None

    pickup_routes: Optional[Dict[str, List[RouteParams]]] = None
    delivery_routes: Optional[Dict[str, List[RouteParams]]] = None

    ramp_full_age: int = 4
    ramp_full_threshold: float = 0.90

    tariff_demand_curve: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.initial_a is None:
            self.initial_a = {self.legacy_plant_id: 1, self.candidate_plant_id: 0}
        if self.alpha_by_plant is None:
            self.alpha_by_plant = {j: 1.0 for j in self.plants}
        if self.c_in is None or self.c_out is None or self.tau is None or self.reconfig is None:
            raise ValueError("c_in, c_out, tau, and reconfig must be provided")

        if self.pickup_routes is None:
            self.pickup_routes = {}
        if self.delivery_routes is None:
            self.delivery_routes = {}

        self.tariff_demand_curve = self._normalise_tariff_demand_curve(self.tariff_demand_curve)

    def _normalise_tariff_demand_curve(self, curve: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if curve is None:
            return {
                "curve_model": "identity",
                "curve_params": {},
                "tariff_grid": [0.0, 5.0],
                "retention_grid": [1.0, 1.0],
            }

        if not isinstance(curve, dict):
            raise TypeError("tariff_demand_curve must be a dict or None")

        model = str(curve.get("curve_model", curve.get("selected_curve_model", "identity")))
        params = curve.get("curve_params", {}) or {}
        tariff_grid = curve.get("tariff_grid")
        retention_grid = curve.get("retention_grid")

        if tariff_grid is not None or retention_grid is not None:
            if tariff_grid is None or retention_grid is None:
                raise ValueError("tariff_demand_curve must provide both tariff_grid and retention_grid when using grid interpolation")
            tariff_grid = [float(x) for x in tariff_grid]
            retention_grid = [float(x) for x in retention_grid]
            if len(tariff_grid) != len(retention_grid):
                raise ValueError("tariff_grid and retention_grid must have the same length")
            if len(tariff_grid) < 2:
                raise ValueError("tariff_grid must contain at least two points")
            for i in range(1, len(tariff_grid)):
                if tariff_grid[i] < tariff_grid[i - 1]:
                    raise ValueError("tariff_grid must be nondecreasing")
            for r in retention_grid:
                if not (0.0 <= r <= 1.0):
                    raise ValueError(f"retention_grid entries must lie in [0,1], got {r}")
        else:
            tariff_grid = None
            retention_grid = None

        return {
            "curve_model": model,
            "curve_params": {str(k): float(v) for k, v in params.items()},
            "tariff_grid": tariff_grid,
            "retention_grid": retention_grid,
        }

    def tariff_demand_retention(self, tariff_rate: float) -> float:
        curve = getattr(self, "tariff_demand_curve", None) or {}
        tau = max(0.0, float(tariff_rate))

        tariff_grid = curve.get("tariff_grid")
        retention_grid = curve.get("retention_grid")
        if tariff_grid is not None and retention_grid is not None:
            xs = tariff_grid
            ys = retention_grid
            if tau <= xs[0]:
                return float(max(0.0, min(1.0, ys[0])))
            if tau >= xs[-1]:
                return float(max(0.0, min(1.0, ys[-1])))
            idx = bisect.bisect_right(xs, tau)
            x0, x1 = xs[idx - 1], xs[idx]
            y0, y1 = ys[idx - 1], ys[idx]
            if x1 <= x0:
                return float(max(0.0, min(1.0, y0)))
            w = (tau - x0) / (x1 - x0)
            y = y0 + w * (y1 - y0)
            return float(max(0.0, min(1.0, y)))

        model = str(curve.get("curve_model", "identity"))
        params = curve.get("curve_params", {}) or {}

        if model == "identity":
            y = 1.0
        elif model == "exponential":
            eta = float(params.get("eta", 0.0))
            y = math.exp(-eta * tau)
        elif model == "floor_exponential":
            floor_ = float(params.get("floor", 0.0))
            eta = float(params.get("eta", 0.0))
            y = floor_ + (1.0 - floor_) * math.exp(-eta * tau)
        elif model == "linear_clipped":
            a = float(params.get("a", 1.0))
            b = float(params.get("b", 0.0))
            y = a + b * tau
        else:
            raise ValueError(f"Unsupported tariff demand curve model: {model}")

        return float(max(0.0, min(1.0, y)))

    def sample_regime_path(self, seed: Optional[int] = None) -> List[int]:
        rng = random.Random(seed)
        path = [self.xi_1]
        if self.H >= 2:
            if self.xi_2_forced is not None:
                path.append(self.xi_2_forced)
            else:
                path.append(self._sample_next_regime(path[-1], rng))
        while len(path) < self.H:
            path.append(self._sample_next_regime(path[-1], rng))
        return path

    def _sample_next_regime(self, xi: int, rng: random.Random) -> int:
        idx = self.regimes.index(xi)
        probs = self.P[idx]
        u = rng.random()
        s = 0.0
        for j, p in enumerate(probs):
            s += p
            if u <= s:
                return self.regimes[j]
        return self.regimes[-1]

    def ramp_factor(self, plant_id: str, age: int) -> float:
        plant = self.plants[plant_id]
        if not plant.is_candidate:
            return 1.0
        return 1.0 - math.exp(-plant.ramp_kappa * max(age, 0))

    def effective_capacity(self, plant_id: str, t: int, xi_t: int, a_jt: int, age: int) -> float:
        plant = self.plants[plant_id]
        K = at(plant.K_bar, t)
        chi = float(plant.chi_by_regime.get(xi_t, 1.0))
        rho = self.ramp_factor(plant_id, age)
        return K * chi * float(a_jt) * rho

    def base_delivered_cost(self, plant_id: str, market_id: str) -> float:
        return float(self.plants[plant_id].c_mfg) + float(self.c_out[plant_id][market_id])

    def tariff_rate(self, plant_id: str, market_id: str, t: int, xi_t: int) -> float:
        series = getattr(self, "tau_time_series", None)
        if series is not None and plant_id in series and market_id in series[plant_id]:
            vals = series[plant_id][market_id]
            if t < 1 or t > len(vals):
                raise IndexError(f"t={t} out of range for tariff series {plant_id}->{market_id} with length {len(vals)}")
            return float(vals[t - 1])
        return float(self.tau[plant_id][market_id][xi_t])

    def tariff_inclusive_delivered_cost(self, plant_id: str, market_id: str, xi_t: int, t: Optional[int] = None) -> float:
        if t is None:
            t = 1
        c = self.base_delivered_cost(plant_id, market_id)
        tr = self.tariff_rate(plant_id, market_id, int(t), xi_t)
        return c * (1.0 + tr)

    def inbound_unit_cost(self, supplier_id: str, plant_id: str) -> float:
        return float(self.c_in[supplier_id][plant_id])

    def relationship_premium_factor(self, candidate_age: int) -> float:
        """Front-loaded decay factor for supplier relationship premium.

        The old implementation imposed a flat premium for the entire H_rel window,
        which made the inbound cost floor too stable. The revised rule keeps the
        same horizon H_rel but lets the premium decay linearly with plant age.
        """
        age = int(candidate_age)
        horizon = max(1, int(self.H_rel))
        if age <= 0:
            return 1.0
        if age > horizon:
            return 0.0
        return max(0.0, (horizon - age + 1) / horizon)

    def material_unit_cost(self, supplier_id: str, plant_id: str, candidate_age: int) -> float:
        sup = self.suppliers[supplier_id]
        if plant_id == self.candidate_plant_id:
            rel_w = self.relationship_premium_factor(candidate_age)
            return float(sup.c_mat_base) + float(sup.delta_rel) * rel_w
        return float(sup.c_mat_base)

    def qualification_weight(self, age: int) -> float:
        """Front-loaded qualification cost weight with average 1 over the window.

        Summing qualification_cost(age) for age=1..qual_ell yields the same total
        burden as the old flat schedule qual_G * qual_ell, but the time profile is
        steeper early and lighter later, which helps period costs show less flatness.
        """
        ell = max(1, int(self.reconfig.qual_ell))
        if age < 1 or age > ell:
            return 0.0
        if ell == 1:
            return 1.0
        start_w = 1.8
        end_w = 0.6
        raw = [start_w + (end_w - start_w) * i / (ell - 1) for i in range(ell)]
        scale = ell / sum(raw)
        return raw[age - 1] * scale

    def qualification_cost(self, a_candidate: int, age: int) -> float:
        if int(a_candidate) != 1:
            return 0.0
        if age < 1 or age > int(self.reconfig.qual_ell):
            return 0.0
        return float(self.reconfig.qual_G) * self.qualification_weight(int(age))

    def approximate_unit_service_cost(self, t: int, plant_id: str, market_id: str, xi_t: int, age: int) -> float:
        """Proxy unit cost used to form a service-share approximation before solving the LP.

        This includes the plant-to-market tariff-inclusive delivered cost plus the cheapest
        upstream material + inbound procurement cost available to that plant.
        """
        candidate_age = age if plant_id == self.candidate_plant_id else self.H_rel + 1
        best_upstream = min(
            self.material_unit_cost(i, plant_id, candidate_age) + self.inbound_unit_cost(i, plant_id)
            for i in self.suppliers
        )
        return best_upstream + self.tariff_inclusive_delivered_cost(plant_id, market_id, xi_t, t=t)

    def service_share_proxy(self, t: int, xi_t: int, a: Dict[str, int], age: int, market_id: str) -> Dict[str, float]:
        """Construct a capacity- and cost-feasible proxy for market service shares."""
        plant_ids = [self.legacy_plant_id, self.candidate_plant_id]
        raw: Dict[str, float] = {}
        for j in plant_ids:
            a_j = int(a.get(j, 0))
            if a_j <= 0:
                raw[j] = 0.0
                continue
            cap_j = self.effective_capacity(
                plant_id=j,
                t=t,
                xi_t=xi_t,
                a_jt=a_j,
                age=(age if j == self.candidate_plant_id else 0),
            )
            age_for_cost = age if j == self.candidate_plant_id else self.H_rel + 1
            unit_cost = self.approximate_unit_service_cost(t=t, plant_id=j, market_id=market_id, xi_t=xi_t, age=age_for_cost)
            raw[j] = float(cap_j) / max(float(unit_cost), 1.0e-8)

        total = sum(raw.values())
        if total <= 1.0e-12:
            return {self.legacy_plant_id: 1.0, self.candidate_plant_id: 0.0}
        return {j: raw[j] / total for j in plant_ids}

    def tariff_pressure_index(
        self,
        t: int,
        xi_t: int,
        a_candidate: Optional[int] = None,
        a: Optional[Dict[str, int]] = None,
        age: int = 0,
    ) -> Dict[str, float]:
        """Return market-level effective tariff pressure for every market.

        Demand-side tariff pressure is computed for both markets using
        the two-plant service-share proxy so that all four plant->market tariff lanes
        can affect realised demand under different strategies.
        """
        if a is None:
            a = {
                self.legacy_plant_id: 1,
                self.candidate_plant_id: int(a_candidate or 0),
            }

        plant_ids = [self.legacy_plant_id, self.candidate_plant_id]
        out: Dict[str, float] = {}
        for k in self.markets:
            weights = self.service_share_proxy(t=t, xi_t=xi_t, a=a, age=age, market_id=k)
            out[k] = sum(weights[j] * self.tariff_rate(j, k, t, xi_t) for j in plant_ids)
        return out

    def potential_demand(
        self,
        t: int,
        xi_t: int,
        a_candidate: Optional[int] = None,
        a: Optional[Dict[str, int]] = None,
        age: int = 0,
    ) -> Dict[str, float]:
        """Return potential demand under the annual tariff-demand retention curve."""
        bar_tau = self.tariff_pressure_index(t=t, xi_t=xi_t, a_candidate=a_candidate, a=a, age=age)

        out: Dict[str, float] = {}
        for k, mk in self.markets.items():
            d0 = at(mk.d0, t)
            out[k] = d0 * self.tariff_demand_retention(bar_tau[k])
        return out

    def realised_demand(self, t: int, xi_t: int, a: Dict[str, int], age: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        pot = self.potential_demand(t=t, xi_t=xi_t, a=a, age=age)

        cap_total = 0.0
        for j in self.plants:
            cap_total += self.effective_capacity(
                plant_id=j,
                t=t,
                xi_t=xi_t,
                a_jt=int(a[j]),
                age=(age if j == self.candidate_plant_id else 0),
            )

        pot_total = sum(pot.values())
        lam = 1.0 if pot_total <= 0 else min(1.0, cap_total / pot_total)

        d = {k: lam * pot[k] for k in pot}
        L = {k: pot[k] - d[k] for k in pot}
        return d, L

    def diagnose(self) -> List[Tuple[str, bool, str]]:
        checks: List[Tuple[str, bool, str]] = []

        def add(name: str, ok: bool, msg: str):
            checks.append((name, ok, msg))

        add("multi_period", self.H >= 2, f"H={self.H} must be >= 2")
        add("activation_lag_one", self.activation_lag == 1, f"activation_lag={self.activation_lag} must be 1")

        n = len(self.regimes)
        ok_shape = len(self.P) == n and all(len(row) == n for row in self.P)
        add("markov_matrix_shape", ok_shape, "P must be n-by-n")
        ok_rows = ok_shape and all(abs(sum(row) - 1.0) < 1e-8 for row in self.P)
        add("markov_rows_sum_to_one", ok_rows, "each row of P must sum to 1")

        try:
            d1 = at(self.markets[self.core_market_id].d0, 1)
            d2 = at(self.markets[self.additional_market_id].d0, 1)
            add("core_market_larger", d1 > d2, f"d0(D1)={d1} must be > d0(D2)={d2}")
        except Exception as e:
            add("core_market_larger", False, f"failed to read d0 values: {e}")

        ok_price = True
        msgs: List[str] = []
        for k, mk in self.markets.items():
            if mk.price <= 0:
                ok_price = False
                msgs.append(f"{k}: price={mk.price}")
        add("market_prices_positive", ok_price, "; ".join(msgs) if msgs else "all prices > 0")

        try:
            probe = [0.0, 0.25, 1.0, 2.0, 3.0]
            vals = [self.tariff_demand_retention(x) for x in probe]
            ok_curve = all(0.0 <= v <= 1.0 for v in vals) and all(vals[i] <= vals[i - 1] + 1.0e-10 for i in range(1, len(vals)))
            add("tariff_demand_curve_valid", ok_curve, f"retention probes={vals}")
        except Exception as e:
            add("tariff_demand_curve_valid", False, f"failed to evaluate tariff demand curve: {e}")

        try:
            c_jo_kc = self.base_delivered_cost(self.legacy_plant_id, self.core_market_id)
            c_jn_kc = self.base_delivered_cost(self.candidate_plant_id, self.core_market_id)
            add(
                "legacy_static_advantage_core",
                c_jo_kc < c_jn_kc,
                f"c_base({self.legacy_plant_id}->{self.core_market_id})={c_jo_kc:.4f}, "
                f"c_base({self.candidate_plant_id}->{self.core_market_id})={c_jn_kc:.4f}",
            )
        except Exception as e:
            add("legacy_static_advantage_core", False, f"failed to compare base delivered cost: {e}")

        ok_inbound = True
        msgs = []
        for i in self.suppliers:
            lhs = self.inbound_unit_cost(i, self.legacy_plant_id)
            rhs = self.inbound_unit_cost(i, self.candidate_plant_id)
            if not (lhs < rhs):
                ok_inbound = False
                msgs.append(f"{i}: c_in({self.legacy_plant_id})={lhs} !< c_in({self.candidate_plant_id})={rhs}")
        add("legacy_inbound_advantage_all_suppliers", ok_inbound, "; ".join(msgs) if msgs else "all suppliers satisfy c_in legacy < candidate")

        ok_rel = True
        msgs = []
        for i in self.suppliers:
            delta = self.suppliers[i].delta_rel
            if delta < 0:
                ok_rel = False
                msgs.append(f"{i}: delta_rel={delta}")
        add("relationship_premium_nonnegative", ok_rel, "; ".join(msgs) if msgs else "all delta_rel >= 0")

        try:
            xi = self.xi_1
            tau_jo_kc = self.tariff_rate(self.legacy_plant_id, self.core_market_id, 1, xi)
            tau_jn_kc = self.tariff_rate(self.candidate_plant_id, self.core_market_id, 1, xi)
            add(
                "initial_candidate_tariff_advantage_core",
                tau_jn_kc < tau_jo_kc,
                f"tau({self.legacy_plant_id}->{self.core_market_id}, xi1)={tau_jo_kc:.4f}, "
                f"tau({self.candidate_plant_id}->{self.core_market_id}, xi1)={tau_jn_kc:.4f}",
            )
        except Exception as e:
            add("initial_candidate_tariff_advantage_core", False, f"failed to compare initial tariff: {e}")

        if len(self.regimes) >= 2:
            try:
                tau1 = self.tariff_rate(self.legacy_plant_id, self.core_market_id, 1, self.xi_1)
                tau2 = self.tariff_rate(self.legacy_plant_id, self.core_market_id, 2, self.xi_2_forced if self.xi_2_forced is not None else self.xi_1)
                add("initial_tariff_escalation_jo_kc", tau2 > tau1, f"tau_t1={tau1:.4f}, tau_t2={tau2:.4f}")
            except Exception as e:
                add("initial_tariff_escalation_jo_kc", False, f"failed to check tau escalation: {e}")

        try:
            cap0 = self.effective_capacity(
                plant_id=self.candidate_plant_id,
                t=1,
                xi_t=self.xi_1,
                a_jt=1,
                age=0,
            )
            add("candidate_capacity_zero_before_opening", abs(cap0) < 1e-12, f"Cap(age=0)={cap0:.6f}")
        except Exception as e:
            add("candidate_capacity_zero_before_opening", False, f"failed to evaluate ramp at age 0: {e}")

        try:
            rho_full = self.ramp_factor(self.candidate_plant_id, self.ramp_full_age)
            add(
                "candidate_ramp_near_full_at_full_age",
                rho_full >= self.ramp_full_threshold,
                f"rho(age={self.ramp_full_age})={rho_full:.4f}, threshold={self.ramp_full_threshold:.4f}",
            )
        except Exception as e:
            add("candidate_ramp_near_full_at_full_age", False, f"failed to evaluate ramp full threshold: {e}")

        add("reconfig_positive_F", self.reconfig.F > 0, f"F={self.reconfig.F}")
        add("reconfig_nonnegative_G", self.reconfig.qual_G >= 0, f"G={self.reconfig.qual_G}")
        add("reconfig_nonnegative_salvage", self.reconfig.S_salv >= 0, f"S_salv={self.reconfig.S_salv}")

        return checks

    def validate_assumptions(self) -> None:
        failed = [(n, m) for n, ok, m in self.diagnose() if not ok]
        if failed:
            detail = "\n".join([f"- {n}: {m}" for n, m in failed])
            raise ValueError(f"Scenario assumption checks failed:\n{detail}")

    def print_diagnose(self) -> None:
        for name, ok, msg in self.diagnose():
            print(f"[{'OK' if ok else 'FAIL'}] {name}: {msg}")
