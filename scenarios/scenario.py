from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import random


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
    eta: float  # demand sensitivity to tariff pressure
    cu: float  # unit cost of demand loss (used when potential > realised)
    price: float  # uniform selling price p_k for this market
    demand_floor: float = 0.0  # minimum fraction of d0 that remains even under high tariff pressure


@dataclass(frozen=True)
class ReconfigParams:
    activation_plant: str  # candidate plant id
    F: float  # fixed investment cost, paid when plant becomes operational (t+1)
    qual_G: float  # qualification / deposit fixed cost during early window after activation
    qual_ell: int  # length of qualification window (periods after activation)
    withdrawal_plant: str  # legacy plant id
    S_salv: float  # salvage credit when withdrawing legacy capacity


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

    # horizon and discounting
    H: int
    gamma: float

    # sets
    suppliers: Dict[str, SupplierParams]
    plants: Dict[str, PlantParams]
    markets: Dict[str, MarketParams]

    # story identifiers
    legacy_plant_id: str
    candidate_plant_id: str
    core_market_id: str
    additional_market_id: str

    # regime process
    regimes: List[int]
    P: List[List[float]]
    xi_1: int
    xi_2_forced: Optional[int] = None  # to enforce early escalation background

    # structural dynamics parameters stored here, enforced later in env
    activation_lag: int = 1
    initial_a: Optional[Dict[str, int]] = None  # a_{j,1}
    initial_age: int = 0  # age_1 for candidate plant

    # material requirement alpha_j
    alpha_by_plant: Optional[Dict[str, float]] = None

    # inbound logistics cost c_in[i][j]
    c_in: Optional[Dict[str, Dict[str, float]]] = None

    # outbound baseline logistics cost c_out[j][k]
    c_out: Optional[Dict[str, Dict[str, float]]] = None

    # outbound tariff tau[j][k][xi] in (0,1)
    tau: Optional[Dict[str, Dict[str, Dict[int, float]]]] = None

    # relationship stabilisation horizon
    H_rel: int = 10

    # structural costs
    reconfig: Optional[ReconfigParams] = None

    # route-based tours
    pickup_routes: Optional[Dict[str, List[RouteParams]]] = None   # supplier -> plant tours, keyed by plant
    delivery_routes: Optional[Dict[str, List[RouteParams]]] = None  # plant -> market tours, keyed by plant

    # ramp-up target used for scenario-level validation (story says roughly 3 periods to normal, i.e., age=4 near full)
    ramp_full_age: int = 4
    ramp_full_threshold: float = 0.90  # rho(ramp_full_age) should be >= this

    def __post_init__(self):
        if self.initial_a is None:
            self.initial_a = {self.legacy_plant_id: 1, self.candidate_plant_id: 0}
        if self.alpha_by_plant is None:
            self.alpha_by_plant = {j: 1.0 for j in self.plants}
        if self.c_in is None or self.c_out is None or self.tau is None or self.reconfig is None:
            raise ValueError("c_in, c_out, tau, and reconfig must be provided")

        # routes default: empty dicts are not allowed because 3.1 assumes route-based tours
        if self.pickup_routes is None:
            self.pickup_routes = {}
        if self.delivery_routes is None:
            self.delivery_routes = {}

    # ---------- regime process ----------
    def sample_regime_path(self, seed: Optional[int] = None) -> List[int]:
        """Sample xi_1..xi_H. If xi_2_forced is set, xi_2 is fixed to that value."""
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

    # ---------- ramp-up and capacity ----------
    def ramp_factor(self, plant_id: str, age: int) -> float:
        """rho_j(age). Legacy rho=1; candidate rho(age)=1-exp(-kappa*age)."""
        plant = self.plants[plant_id]
        if not plant.is_candidate:
            return 1.0
        return 1.0 - math.exp(-plant.ramp_kappa * max(age, 0))

    def effective_capacity(self, plant_id: str, t: int, xi_t: int, a_jt: int, age: int) -> float:
        """Cap_{j,t} = K_bar * chi(xi_t) * a_{j,t} * rho(age)."""
        plant = self.plants[plant_id]
        K = at(plant.K_bar, t)
        chi = float(plant.chi_by_regime.get(xi_t, 1.0))
        rho = self.ramp_factor(plant_id, age)
        return K * chi * float(a_jt) * rho

    # ---------- unit costs ----------
    def base_delivered_cost(self, plant_id: str, market_id: str) -> float:
        """c_{j,k} = c_mfg_j + c_out_{j,k}."""
        return float(self.plants[plant_id].c_mfg) + float(self.c_out[plant_id][market_id])

    def tariff_inclusive_delivered_cost(self, plant_id: str, market_id: str, xi_t: int) -> float:
        """tilde c_{j,k,t} = c_{j,k} * (1 + tau_{j,k}(xi_t))."""
        c = self.base_delivered_cost(plant_id, market_id)
        tr = float(self.tau[plant_id][market_id][xi_t])
        return c * (1.0 + tr)

    def inbound_unit_cost(self, supplier_id: str, plant_id: str) -> float:
        return float(self.c_in[supplier_id][plant_id])

    def material_unit_cost(self, supplier_id: str, plant_id: str, candidate_age: int) -> float:
        """c_mat_{i,j,t}. For candidate base, apply relationship premium for age < H_rel."""
        sup = self.suppliers[supplier_id]
        if plant_id == self.candidate_plant_id and candidate_age < self.H_rel:
            return float(sup.c_mat_base) + float(sup.delta_rel)
        return float(sup.c_mat_base)

    # ---------- demand construction (Chapter 3) ----------
    def tariff_pressure_index(self, xi_t: int, a_candidate: int) -> Dict[str, float]:
        """bar_tau_{k,t} as in Chapter 3."""
        out: Dict[str, float] = {}
        for k in self.markets:
            if a_candidate == 0:
                out[k] = float(self.tau[self.legacy_plant_id][k][xi_t])
            else:
                out[k] = min(
                    float(self.tau[self.legacy_plant_id][k][xi_t]),
                    float(self.tau[self.candidate_plant_id][k][xi_t]),
                )
        return out

    def potential_demand(self, t: int, xi_t: int, a_candidate: int) -> Dict[str, float]:
        """bar d_{k,t} = d0_{k,t} * exp(-eta_k * bar_tau_{k,t})."""
        bar_tau = self.tariff_pressure_index(xi_t, a_candidate)
        out: Dict[str, float] = {}
        for k, mk in self.markets.items():
            d0 = at(mk.d0, t)
            floor = float(getattr(mk, "demand_floor", 0.0))
            floor = max(0.0, min(0.99, floor))  # safety clamp
            out[k] = d0 * (floor + (1.0 - floor) * math.exp(-mk.eta * bar_tau[k]))
        return out

    def realised_demand(
        self, t: int, xi_t: int, a: Dict[str, int], age: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (d, L) where d is realised demand and L is demand loss."""
        a_candidate = int(a[self.candidate_plant_id])
        pot = self.potential_demand(t, xi_t, a_candidate)

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

    # ---------- self-check ----------
    def diagnose(self) -> List[Tuple[str, bool, str]]:
        """Return a list of (check_name, passed, message). Does not raise."""
        checks: List[Tuple[str, bool, str]] = []

        def add(name: str, ok: bool, msg: str):
            checks.append((name, ok, msg))

        # 3.1(1) multi-period
        add("multi_period", self.H >= 2, f"H={self.H} must be >= 2")

        # activation lag rule used in Chapter 3
        add("activation_lag_one", self.activation_lag == 1, f"activation_lag={self.activation_lag} must be 1")

        # 3.1(7) Markov regime process sanity
        n = len(self.regimes)
        ok_shape = len(self.P) == n and all(len(row) == n for row in self.P)
        add("markov_matrix_shape", ok_shape, "P must be n-by-n")
        ok_rows = ok_shape and all(abs(sum(row) - 1.0) < 1e-8 for row in self.P)
        add("markov_rows_sum_to_one", ok_rows, "each row of P must sum to 1")

        # story (3) D1 > D2
        try:
            d1 = at(self.markets[self.core_market_id].d0, 1)
            d2 = at(self.markets[self.additional_market_id].d0, 1)
            add("core_market_larger", d1 > d2, f"d0(D1)={d1} must be > d0(D2)={d2}")
        except Exception as e:
            add("core_market_larger", False, f"failed to read d0 values: {e}")

        # 3.1(10) uniform selling price p_k present and nonnegative
        ok_price = True
        msgs: List[str] = []
        for k, mk in self.markets.items():
            if mk.price < 0:
                ok_price = False
                msgs.append(f"{k}: price={mk.price} must be >= 0")
        add("market_prices_defined", ok_price, "; ".join(msgs) if msgs else "ok")

        # story edges and tariffs
        ok_edges = True
        msg_edges: List[str] = []
        for j in [self.legacy_plant_id, self.candidate_plant_id]:
            for k in [self.core_market_id, self.additional_market_id]:
                if j not in self.c_out or k not in self.c_out[j]:
                    ok_edges = False
                    msg_edges.append(f"missing c_out {j}->{k}")
                if j not in self.tau or k not in self.tau[j]:
                    ok_edges = False
                    msg_edges.append(f"missing tau {j}->{k}")
                if j in self.tau and k in self.tau[j]:
                    for xi in self.regimes:
                        if xi not in self.tau[j][k]:
                            ok_edges = False
                            msg_edges.append(f"missing tau {j}->{k} regime {xi}")
                        else:
                            tr = float(self.tau[j][k][xi])
                            if not (0.0 < tr < 1.0):
                                ok_edges = False
                                msg_edges.append(f"tau out of (0,1) for {j}->{k} regime {xi}: {tr}")
        add("four_edges_complete", ok_edges, "; ".join(msg_edges) if msg_edges else "ok")

        # story (1.5) early escalation background: xi2 forced + T1(xi2)>T1(xi1)
        if self.xi_2_forced is None:
            add("early_escalation_defined", False, "xi_2_forced must be set")
            add("early_escalation_T1", False, "cannot check without xi_2_forced")
        else:
            add("early_escalation_defined", True, f"xi_2_forced={self.xi_2_forced}")
            t1_x1 = float(self.tau[self.legacy_plant_id][self.core_market_id][self.xi_1])
            t1_x2 = float(self.tau[self.legacy_plant_id][self.core_market_id][self.xi_2_forced])
            add("early_escalation_T1", t1_x2 > t1_x1, f"T1: {t1_x2} must be > {t1_x1}")

        # story (5) candidate has initial tariff advantage on core market
        t1 = float(self.tau[self.legacy_plant_id][self.core_market_id][self.xi_1])
        t3 = float(self.tau[self.candidate_plant_id][self.core_market_id][self.xi_1])
        add("candidate_tariff_advantage_core", t3 < t1, f"T3={t3} must be < T1={t1} at initial regime")

        # story (4) baseline delivered cost ordering on core market absent tariffs
        c_m1 = self.base_delivered_cost(self.legacy_plant_id, self.core_market_id)
        c_m2 = self.base_delivered_cost(self.candidate_plant_id, self.core_market_id)
        add("baseline_cost_order_core", c_m1 < c_m2, f"c(M1,D1)={c_m1} must be < c(M2,D1)={c_m2}")

        # story (4.5) inbound advantage + relationship premium behaviour
        ok_in = True
        msgs = []
        for i, sup in self.suppliers.items():
            if i not in self.c_in or self.legacy_plant_id not in self.c_in[i] or self.candidate_plant_id not in self.c_in[i]:
                ok_in = False
                msgs.append(f"missing c_in entries for supplier {i}")
                continue
            c1 = float(self.c_in[i][self.legacy_plant_id])
            c2 = float(self.c_in[i][self.candidate_plant_id])
            if not (c1 < c2):
                ok_in = False
                msgs.append(f"c_in({i},M1)={c1} not < c_in({i},M2)={c2}")
            if sup.delta_rel < 0:
                ok_in = False
                msgs.append(f"delta_rel for {i} is negative")
        add("supplier_inbound_advantage", ok_in, "; ".join(msgs) if msgs else "ok")

        ok_rel = True
        msgs = []
        for i in self.suppliers:
            base = self.material_unit_cost(i, self.legacy_plant_id, candidate_age=0)
            early = self.material_unit_cost(i, self.candidate_plant_id, candidate_age=0)
            late = self.material_unit_cost(i, self.candidate_plant_id, candidate_age=self.H_rel)
            if not (early >= base):
                ok_rel = False
                msgs.append(f"{i}: early M2 mat {early} should be >= M1 mat {base}")
            if abs(late - base) > 1e-8:
                ok_rel = False
                msgs.append(f"{i}: late M2 mat {late} should equal M1 mat {base}")
        add("relationship_premium_behavior", ok_rel, "; ".join(msgs) if msgs else "ok")

        # story (8) tariff multiplicative cost
        try:
            j = self.legacy_plant_id
            k = self.core_market_id
            xi = self.xi_1
            base_c = self.base_delivered_cost(j, k)
            tilde = self.tariff_inclusive_delivered_cost(j, k, xi)
            tr = float(self.tau[j][k][xi])
            ok = abs(tilde - base_c * (1.0 + tr)) < 1e-8
            add("tariff_multiplicative_cost", ok, f"tilde={tilde}, base*(1+tr)={base_c*(1+tr)}")
        except Exception as e:
            add("tariff_multiplicative_cost", False, f"failed: {e}")

        # story (10) ramp-up: age=4 near full
        r1 = self.ramp_factor(self.candidate_plant_id, 1)
        rT = self.ramp_factor(self.candidate_plant_id, self.ramp_full_age)
        add("ramp_increasing", rT > r1, f"rho(1)={r1}, rho({self.ramp_full_age})={rT}")
        add(
            "ramp_near_full_by_target_age",
            rT >= self.ramp_full_threshold,
            f"rho({self.ramp_full_age})={rT} should be >= {self.ramp_full_threshold}",
        )

        # story (11) initial capacity K1_0 and K2_0 >= D1 + D2 (baseline scale at t=1)
        d1_0 = at(self.markets[self.core_market_id].d0, 1)
        d2_0 = at(self.markets[self.additional_market_id].d0, 1)
        K1 = at(self.plants[self.legacy_plant_id].K_bar, 1)
        K2 = at(self.plants[self.candidate_plant_id].K_bar, 1)
        add("initial_capacity_M1", K1 >= d1_0 + d2_0, f"K1_0={K1} must be >= D1+D2={d1_0+d2_0}")
        add("initial_capacity_M2", K2 >= d1_0 + d2_0, f"K2_0={K2} must be >= D1+D2={d1_0+d2_0}")

        # 3.1(12) supplier total capacity sufficient (scenario-level feasibility sanity)
        W_total = 0.0
        for i, sup in self.suppliers.items():
            W_total += at(sup.W_bar, 1)
        alpha_max = max(self.alpha_by_plant.values()) if self.alpha_by_plant else 1.0
        add(
            "supplier_capacity_total",
            W_total >= alpha_max * (d1_0 + d2_0),
            f"sum W_bar={W_total} must cover alpha*(D1+D2)={alpha_max*(d1_0+d2_0)}",
        )

        # story (17) qualification window is first 3 periods after activation
        add("qualification_window_is_three", self.reconfig.qual_ell == 3, f"qual_ell={self.reconfig.qual_ell} should be 3")
        add("qualification_cost_nonneg", self.reconfig.qual_G >= 0, f"qual_G={self.reconfig.qual_G} must be >=0")

        # story (14) salvage credit nonnegative
        add("salvage_nonneg", self.reconfig.S_salv >= 0, f"S_salv={self.reconfig.S_salv} must be >=0")

        # 3.1(11) route-based multi-stop tours must be defined (minimal check)
        ok_routes = True
        msgs = []
        for j in self.plants:
            prs = self.pickup_routes.get(j, [])
            drs = self.delivery_routes.get(j, [])
            if len(prs) == 0:
                ok_routes = False
                msgs.append(f"no pickup routes for {j}")
            if len(drs) == 0:
                ok_routes = False
                msgs.append(f"no delivery routes for {j}")
        add("route_sets_present", ok_routes, "; ".join(msgs) if msgs else "ok")

        # stronger "multi-stop" check: at least one route visits 2+ nodes (tour)
        ok_multistop = False
        for j in self.plants:
            for r in self.pickup_routes.get(j, []) + self.delivery_routes.get(j, []):
                if len(r.visits) >= 2:
                    ok_multistop = True
                    break
            if ok_multistop:
                break
        add("route_multistop_exists", ok_multistop, "need at least one route with 2+ stops to represent tours")

        return checks

    def validate_assumptions(self) -> None:
        report = self.diagnose()
        failed = [c for c in report if not c[1]]
        if failed:
            lines = [f"- {name}: {msg}" for name, ok, msg in failed]
            raise ValueError("Scenario failed story / Chapter 3.1 checks\n" + "\n".join(lines))

    def print_diagnose(self) -> None:
        for name, ok, msg in self.diagnose():
            mark = "OK " if ok else "FAIL"
            print(f"{mark} {name} -> {msg}")

    def summary(self) -> str:
        return (
            f"SupplyChainScenario(H={self.H}, regimes={self.regimes}, "
            f"plants={list(self.plants)}, markets={list(self.markets)}, suppliers={list(self.suppliers)})"
        )