"""
builders.py — clean-room programmatic construction of RSFQ ``NetlistGraph`` cells.

These builders author RSFQ cells *from scratch* in our own ``NetlistGraph``
representation using published RSFQ physics and the
MIT-LL SFQ5ee+ process facts (Ic density, IcRn, shunt values, betaL targets) —
**no third-party cell-library files are used**. The resulting decks
round-trip losslessly to JoSIM and serve as both seeds for constructive search
and the harness that makes a bare cell simulable.

Provenance note: the JTL
topology is textbook (Likharev & Semenov 1991); the numeric parameters are
process facts also visible in JoSIM's MIT-licensed example fixtures. Everything
emitted here is our own code generating its own netlist — clean IP by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

from margo.netlist.graph import (
    Inductor,
    JJ,
    Model,
    Mutual,
    NetlistGraph,
    RawLine,
    Resistor,
    Source,
)


# --------------------------------------------------------------------------- #
# Process / physics constants (MIT-LL SFQ5ee+ style; published facts)
# --------------------------------------------------------------------------- #

PHI0_WB = 2.06783e-15                   # flux quantum, V·s (= Wb)

# A single SFQ pulse, injected as a triangular voltage pwl with area == Φ0.
# Triangle base = 2*HALF_WIDTH_PS, peak = SFQ_PEAK_UV; area = 0.5*base*peak = Φ0.
SFQ_HALF_WIDTH_PS = 2.5
SFQ_PEAK_UV = PHI0_WB / (SFQ_HALF_WIDTH_PS * 1e-12) * 1e6   # ~827.1 uV


@dataclass
class JTLSpec:
    """Design point for a Josephson Transmission Line. Every field is a knob the
    constructive search can perturb."""
    n_jj: int = 2
    area: float = 2.16            # JJ area -> Ic = 100uA/um^2 * area = 216uA
    l_in_ph: float = 2.0          # input series inductor (pH)
    l_series_ph: float = 2.425    # inter-stage series inductor (pH)
    l_out_ph: float = 2.031       # output series inductor (pH)
    bias_ua: float = 150.0        # per-JJ bias current (uA) ~0.7*Ic(216uA)
    bias_ramp_ps: float = 50.0    # bias turn-on ramp (gentle -> no startup fluxon)
    rb_ohm: float = 5.23          # shunt (damping) resistor across each JJ
    lp_ph: float = 0.086          # JJ bottom -> ground parasitic
    lrb_ph: float = 0.086         # shunt series inductor
    lpr_ph: float = 0.278         # bias injection inductor
    r_out_ohm: float = 2.0        # output load resistor


def jmitll_model() -> Model:
    """The standard MIT-LL-style JJ model used across these cells."""
    raw = (".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, "
           "r0=160, rN=16, icrit=0.1mA)")
    return Model(
        name="jmitll", mtype="jj", raw=raw,
        params={"rtype": "1", "vg": "2.8mV", "cap": "0.07pF",
                "r0": "160", "rn": "16", "icrit": "0.1mA"},
    )


def _ind(name: str, a: str, b: str, ph: float) -> Inductor:
    return Inductor(name=name, nodes=[a, b], raw=f"{ph:g}p")


def _res(name: str, a: str, b: str, ohm: float) -> Resistor:
    return Resistor(name=name, nodes=[a, b], raw=f"{ohm:g}")


def _jj(name: str, a: str, b: str, area: float) -> JJ:
    return JJ(name=name, nodes=[a, b], model="jmitll",
              params={"area": f"{area:g}"})


def _mutual(name: str, l1: str, l2: str, k: float) -> Mutual:
    """Mutual inductive coupling between two named inductors:  Kname L1 L2 k.

    `k` is the dimensionless coupling coefficient (-1..1). A NEGATIVE k makes the
    induced current in L2 *oppose* the driving current in L1 — the transformer
    polarity that lets a stored fluxon SUBTRACT current from a path (the
    ingredient unipolar SFQ addition cannot provide)."""
    return Mutual(name=name, nodes=[l1, l2], raw=f"{k:g}", value=k, letter="K")


def build_jtl(spec: JTLSpec) -> NetlistGraph:
    """Build a bare (no-testbench) JTL cell of `spec.n_jj` junctions.

    Topology (per junction k): a JJ Bk sits on the top rail node s{k}, shunted by
    a damping R across it, with its bottom tied near ground through a small
    parasitic L, and biased by a current source injected at s{k}. Junctions are
    chained by series inductors. Net "in" is the input, "out" the output.
    """
    g = NetlistGraph()
    g.add(jmitll_model())

    n = spec.n_jj
    # input inductor: in -> s1
    g.add(_ind("LIN", "in", "s1", spec.l_in_ph))

    for k in range(1, n + 1):
        s = f"s{k}"
        gnd_node = f"g{k}"        # JJ bottom
        q = f"q{k}"               # shunt mid-node
        b = f"b{k}"               # bias inject node
        # Junction + its bottom parasitic to ground.
        g.add(_jj(f"B{k:02d}", s, gnd_node, spec.area))
        g.add(_ind(f"LP{k:02d}", gnd_node, "0", spec.lp_ph))
        # Damping shunt across the JJ (s -> q -> gnd_node).
        g.add(_ind(f"LRB{k:02d}", gnd_node, q, spec.lrb_ph))
        g.add(_res(f"RB{k:02d}", q, s, spec.rb_ohm))
        # Per-JJ bias: IB{k} 0 -> b{k}, injected into the rail via LPR.
        g.add(_bias(f"IB{k:02d}", "0", b, spec.bias_ua, spec.bias_ramp_ps))
        g.add(_ind(f"LPR{k:02d}", s, b, spec.lpr_ph))
        # Series link to the next stage.
        if k < n:
            g.add(_ind(f"L{k:02d}", s, f"s{k + 1}", spec.l_series_ph))

    # output inductor + load
    g.add(_ind("LOUT", f"s{n}", "out", spec.l_out_ph))
    g.add(_res("ROUT", "out", "0", spec.r_out_ohm))
    return g


def _bias(name: str, a: str, b: str, ua: float, ramp_ps: float = 50.0) -> Source:
    """A DC-ish bias current source: ramp to `ua` over `ramp_ps` then hold. A
    gentle ramp avoids injecting a startup fluxon when the bias turns on."""
    func = f"pwl(0 0 {ramp_ps:g}p {ua:g}u)"
    return Source(name=name, nodes=[a, b], letter="I", func=func)


# --------------------------------------------------------------------------- #
# Testbench harness (makes a bare cell simulable + probeable)
# --------------------------------------------------------------------------- #


def _sfq_pulse_train(pulse_ps: list[float]) -> str:
    """A pwl voltage drive emitting one Φ0 SFQ pulse at each center time (ps)."""
    toks = ["0", "0"]
    peak = SFQ_PEAK_UV
    for tc in pulse_ps:
        toks += [f"{tc - SFQ_HALF_WIDTH_PS:g}p", "0",
                 f"{tc:g}p", f"{peak:g}u",
                 f"{tc + SFQ_HALF_WIDTH_PS:g}p", "0"]
    return f"pwl({' '.join(toks)})"


def wrap_testbench(cell: NetlistGraph, n_jj: int,
                   pulse_ps: list[float] | None = None,
                   t_end_ps: float | None = None) -> NetlistGraph:
    """Return a complete runnable deck: the cell + an SFQ input drive on net "in",
    `.tran`, and `.print PHASE` on every junction so the verifier can count pulses.

    The input source VIN injects one SFQ pulse per entry in `pulse_ps`; a correct
    JTL propagates each pulse through every junction (pulse count == len(pulse_ps))."""
    if pulse_ps is None:
        pulse_ps = [100.0, 300.0, 500.0]
    if t_end_ps is None:
        t_end_ps = (max(pulse_ps) + 200.0) if pulse_ps else 400.0

    g = NetlistGraph()
    # Copy the cell's items in order (model first, then elements).
    for it in cell.items:
        g.add(it)

    # Input SFQ drive on net "in".
    g.add(Source(name="VIN", nodes=["in", "0"], letter="V",
                 func=_sfq_pulse_train(pulse_ps)))

    # Control directives as raw lines.
    g.add(RawLine(text=f".tran 0.25p {t_end_ps:g}p 0 0.25p"))
    for k in range(1, n_jj + 1):
        g.add(RawLine(text=f".print PHASE B{k:02d}"))
    g.add(RawLine(text=".print DEVI ROUT"))
    g.add(RawLine(text=".end"))
    return g


def build_jtl_testbench(spec: JTLSpec,
                        pulse_ps: list[float] | None = None) -> NetlistGraph:
    """Convenience: bare JTL cell wrapped in its SFQ testbench, ready for JoSIM."""
    cell = build_jtl(spec)
    return wrap_testbench(cell, spec.n_jj, pulse_ps=pulse_ps)


# --------------------------------------------------------------------------- #
# Splitter (1 -> 2 fanout): the first cell with a real functional decision.
# --------------------------------------------------------------------------- #


@dataclass
class SplitterSpec:
    """Design point for an SFQ splitter (1 input fluxon -> 2 output fluxons).

    The input junction receives the pulse; its current then divides between two
    branch junctions. If the per-branch bias is too low, a branch can't reach Ic
    and drops its pulse -> fanout breaks. That threshold is what bias margin
    measures (unlike a JTL, where propagation is unconditional)."""
    area_in: float = 2.16         # input junction area
    area_br: float = 2.16         # branch junction area
    l_in_ph: float = 2.0          # input series inductor
    l_branch_ph: float = 2.425    # split-node -> branch junction inductor
    l_out_ph: float = 2.031       # branch junction -> output
    bias_in_ua: float = 150.0     # input-junction bias
    bias_br_ua: float = 150.0     # per-branch bias (the knob fanout depends on)
    rb_ohm: float = 5.23
    lp_ph: float = 0.086
    lrb_ph: float = 0.086
    lpr_ph: float = 0.278
    r_out_ohm: float = 2.0
    bias_ramp_ps: float = 50.0


def _stage(g: NetlistGraph, tag: str, s: str, area: float, bias_ua: float,
           spec: SplitterSpec) -> None:
    """Add one biased+shunted junction `B{tag}` on rail node `s` (bottom near gnd)."""
    gnd_node = f"g{tag}"
    q = f"q{tag}"
    b = f"bb{tag}"
    g.add(_jj(f"B{tag}", s, gnd_node, area))
    g.add(_ind(f"LP{tag}", gnd_node, "0", spec.lp_ph))
    g.add(_ind(f"LRB{tag}", gnd_node, q, spec.lrb_ph))
    g.add(_res(f"RB{tag}", q, s, spec.rb_ohm))
    g.add(_bias(f"IB{tag}", "0", b, bias_ua, spec.bias_ramp_ps))
    g.add(_ind(f"LPR{tag}", s, b, spec.lpr_ph))


def build_splitter(spec: SplitterSpec) -> NetlistGraph:
    """Build a bare 1->2 splitter. Junctions: BIN (input), BA / BB (branches).
    Outputs on nets ``outa`` / ``outb`` each driven through a series inductor +
    load resistor. Input on net ``in``."""
    g = NetlistGraph()
    g.add(jmitll_model())
    # input stage
    g.add(_ind("LIN", "in", "s0", spec.l_in_ph))
    _stage(g, "IN", "s0", spec.area_in, spec.bias_in_ua, spec)
    # fan to two branches off the input rail node s0
    g.add(_ind("LBA", "s0", "sa", spec.l_branch_ph))
    g.add(_ind("LBB", "s0", "sb", spec.l_branch_ph))
    # branch A
    _stage(g, "A", "sa", spec.area_br, spec.bias_br_ua, spec)
    g.add(_ind("LOUTA", "sa", "outa", spec.l_out_ph))
    g.add(_res("ROUTA", "outa", "0", spec.r_out_ohm))
    # branch B
    _stage(g, "B", "sb", spec.area_br, spec.bias_br_ua, spec)
    g.add(_ind("LOUTB", "sb", "outb", spec.l_out_ph))
    g.add(_res("ROUTB", "outb", "0", spec.r_out_ohm))
    return g


def wrap_testbench_named(cell: NetlistGraph, probe_jjs: list[str],
                         pulse_ps: list[float] | None = None,
                         t_end_ps: float | None = None) -> NetlistGraph:
    """Wrap a cell whose output loads are already present (e.g. splitter) in an SFQ
    input drive on net "in", with `.print PHASE` on each named junction."""
    if pulse_ps is None:
        pulse_ps = [200.0, 400.0]
    if t_end_ps is None:
        t_end_ps = (max(pulse_ps) + 200.0) if pulse_ps else 400.0
    g = NetlistGraph()
    for it in cell.items:
        g.add(it)
    g.add(Source(name="VIN", nodes=["in", "0"], letter="V",
                 func=_sfq_pulse_train(pulse_ps)))
    g.add(RawLine(text=f".tran 0.25p {t_end_ps:g}p 0 0.25p"))
    for jj in probe_jjs:
        g.add(RawLine(text=f".print PHASE {jj}"))
    g.add(RawLine(text=".end"))
    return g


def build_splitter_testbench(spec: SplitterSpec,
                             pulse_ps: list[float] | None = None) -> NetlistGraph:
    """Splitter wrapped in its SFQ testbench, probing BIN/BA/BB."""
    cell = build_splitter(spec)
    return wrap_testbench_named(cell, ["BIN", "BA", "BB"], pulse_ps=pulse_ps)


# --------------------------------------------------------------------------- #
# Storage-loop core (DFF heart): the first cell whose margin is REAL.
# --------------------------------------------------------------------------- #
#
# A quantizing loop stores 0 or 1 fluxon depending on betaL = 2*pi*Lq*Ic/Phi0.
# Data sets the loop (persistent circulating current); clock reads it out
# destructively through a decision junction. Retention/readout depend sharply on
# Lq and bias -> a genuine operating margin, unlike the JTL/splitter.


@dataclass
class DFFSpec:
    """Minimal destructive-readout DFF (storage-loop core).

    Topology (clean-room, authored from RSFQ physics — the storage loop plus a
    clocked decision pair, the standard DFF read mechanism):

        D --LIND--> a --[BIN->0]                  input/set junction (in loop)
                    a --LQ--> b                    the quantizing inductor (betaL)
                    b --[BST->0]                   storage/escape junction (in loop)
      CLK --LINC--> c --[BCLK->0] --LC--> b        clock launches fluxon into b
                    b --[BOUT->o] --LOUT--> q       output/decision junction -> OUT

    The storage loop is a-LQ-b-BST-(gnd)-BIN-a. A data pulse switches BIN, leaving a
    persistent circulating current (one stored fluxon if betaL~1, held by BST). When
    the clock fluxon arrives at b it must switch *either* BST (escape -> no output)
    *or* BOUT (output). With no stored fluxon, BST is the easy path and absorbs the
    clock (read 0). With a stored fluxon, BST is already holding its 2*pi slip, so
    the clock tips BOUT instead -> output pulse, and the read is destructive (loop
    reset). The decision -> a genuine operating margin set by `lq_ph` (betaL) and the
    BST/BOUT bias asymmetry, unlike the JTL/splitter.

    betaL ~ 1 wants LQ ~ Phi0/(2*pi*Ic) ~ 1.6 pH at Ic=200uA (area 2.0)."""
    area_in: float = 2.0
    area_store: float = 2.0
    area_clk: float = 2.0
    area_out: float = 2.0
    l_in_ph: float = 2.0          # data input series inductor
    l_clk_ph: float = 2.0         # clock input series inductor
    lq_ph: float = 1.6            # *** quantizing inductor (the betaL knob) ***
    lc_ph: float = 2.0            # clock->decision coupling inductor
    l_out_ph: float = 2.0         # readout output inductor
    bias_in_ua: float = 140.0
    bias_store_ua: float = 140.0  # storage/escape JJ bias
    bias_clk_ua: float = 140.0
    bias_out_ua: float = 110.0    # output JJ: sub-threshold unless a fluxon is held
    rb_ohm: float = 5.23
    lp_ph: float = 0.086
    lrb_ph: float = 0.086
    lpr_ph: float = 0.278
    r_out_ohm: float = 2.0
    bias_ramp_ps: float = 50.0


def _shunt_bias(g: NetlistGraph, tag: str, top: str, bot: str, area: float,
                bias_ua: float, spec: DFFSpec, bias_into: str) -> None:
    """Add junction B{tag} (top->bot) with damping shunt across it and a bias
    current injected into net `bias_into`."""
    q = f"q{tag}"
    bb = f"bb{tag}"
    g.add(_jj(f"B{tag}", top, bot, area))
    # shunt across the junction
    g.add(_ind(f"LRB{tag}", bot, q, spec.lrb_ph))
    g.add(_res(f"RB{tag}", q, top, spec.rb_ohm))
    # bias
    g.add(_bias(f"IB{tag}", "0", bb, bias_ua, spec.bias_ramp_ps))
    g.add(_ind(f"LPR{tag}", bias_into, bb, spec.lpr_ph))


def build_dff(spec: DFFSpec) -> NetlistGraph:
    """Build the bare storage-loop DFF core. Junctions BIN/BST/BCLK/BOUT; inputs on
    nets ``D``/``CLK``; output on ``q``. The a-LQ-b loop is the storage element and
    BST/BOUT form the clocked decision pair."""
    g = NetlistGraph()
    g.add(jmitll_model())
    # data input -> set junction at node a (in the storage loop)
    g.add(_ind("LIND", "D", "a", spec.l_in_ph))
    _shunt_bias(g, "IN", "a", "0", spec.area_in, spec.bias_in_ua, spec, "a")
    # quantizing inductor a -> b (the betaL element)
    g.add(_ind("LQ", "a", "b", spec.lq_ph))
    # storage/escape junction at b, closes the loop a-LQ-b-BST-gnd-BIN-a
    _shunt_bias(g, "ST", "b", "0", spec.area_store, spec.bias_store_ua, spec, "b")
    # clock input -> clock junction at node c, fluxon coupled into b
    g.add(_ind("LINC", "CLK", "c", spec.l_clk_ph))
    _shunt_bias(g, "CLK", "c", "0", spec.area_clk, spec.bias_clk_ua, spec, "c")
    g.add(_ind("LC", "c", "b", spec.lc_ph))
    # output/decision junction b -> o, then to the load
    _shunt_bias(g, "OUT", "b", "o", spec.area_out, spec.bias_out_ua, spec, "o")
    g.add(_ind("LOUT", "o", "q", spec.l_out_ph))
    g.add(_res("ROUT", "q", "0", spec.r_out_ohm))
    return g


def build_dff_testbench(spec: DFFSpec, data_ps: list[float],
                        clk_ps: list[float],
                        t_end_ps: float | None = None) -> NetlistGraph:
    """Wrap the DFF core: separate SFQ drives on D and CLK at the given times,
    probing BIN/BST/BCLK/BOUT."""
    if t_end_ps is None:
        t_end_ps = max(data_ps + clk_ps) + 200.0
    g = NetlistGraph()
    cell = build_dff(spec)
    for it in cell.items:
        g.add(it)
    g.add(Source(name="VD", nodes=["D", "0"], letter="V",
                 func=_sfq_pulse_train(data_ps)))
    g.add(Source(name="VC", nodes=["CLK", "0"], letter="V",
                 func=_sfq_pulse_train(clk_ps)))
    g.add(RawLine(text=f".tran 0.1p {t_end_ps:g}p 0 0.1p"))
    for jj in ("BIN", "BST", "BCLK", "BOUT"):
        g.add(RawLine(text=f".print PHASE {jj}"))
    g.add(RawLine(text=".end"))
    return g


# --------------------------------------------------------------------------- #
# Quantizing storage loop: the first cell with a REAL, optimizable margin.
# --------------------------------------------------------------------------- #
#
# A minimal 2-junction loop a-LQ-b: an input/set junction BIN at node a, a storage
# junction BST at node b, joined by the quantizing inductor LQ. A single data
# fluxon switches BIN; whether the loop *traps* the resulting circulating current
# (BST holds, doesn't slip) or lets it *pass through* (BST slips) depends sharply
# on LQ (betaL) and the BST bias. That pass-through->trap boundary is a genuine
# operating margin (verified in JoSIM: storage bias margin widens monotonically
# with LQ), unlike the marginless JTL/splitter. This is the first real
# inverse-design target.


@dataclass
class StorageLoopSpec:
    """Design point for a 2-junction quantizing storage loop. The knobs the
    constructive search tunes to maximize the storage operating margin."""
    area: float = 2.0             # JJ area -> Ic = 100uA/um^2 * area = 200uA
    lq_ph: float = 16.0           # *** quantizing inductor (the betaL knob) ***
    bias_in_ua: float = 140.0     # set-junction bias
    bias_st_ua: float = 80.0      # storage-junction bias (margin is swept on this)
    l_in_ph: float = 2.0          # data input series inductor
    rb_ohm: float = 5.23
    lrb_ph: float = 0.086
    lpr_ph: float = 0.278
    bias_ramp_ps: float = 50.0

    @property
    def betaL(self) -> float:
        ic_a = 100e-6 * self.area     # Ic in A
        return 2 * 3.141592653589793 * self.lq_ph * 1e-12 * ic_a / PHI0_WB


def _loop_jj(g: NetlistGraph, tag: str, top: str, area: float, bias_ua: float,
             spec: StorageLoopSpec) -> None:
    """Add a shunted, biased junction B{tag} (top->ground) for the storage loop."""
    g.add(_jj(f"B{tag}", top, "0", area))
    g.add(_ind(f"LRB{tag}", "0", f"q{tag}", spec.lrb_ph))
    g.add(_res(f"RB{tag}", f"q{tag}", top, spec.rb_ohm))
    g.add(_bias(f"IB{tag}", "0", f"bb{tag}", bias_ua, spec.bias_ramp_ps))
    g.add(_ind(f"LPR{tag}", top, f"bb{tag}", spec.lpr_ph))


def build_storage_loop(spec: StorageLoopSpec) -> NetlistGraph:
    """Build the bare 2-junction quantizing loop. Junctions BIN (set) / BST
    (storage); data input on net ``D``; storage loop is a-LQ-b-BST-(gnd)-BIN-a."""
    g = NetlistGraph()
    g.add(jmitll_model())
    g.add(_ind("LIN", "D", "a", spec.l_in_ph))
    _loop_jj(g, "IN", "a", spec.area, spec.bias_in_ua, spec)
    g.add(_ind("LQ", "a", "b", spec.lq_ph))
    _loop_jj(g, "ST", "b", spec.area, spec.bias_st_ua, spec)
    return g


def build_storage_testbench(spec: StorageLoopSpec,
                            data_ps: list[float] | None = None,
                            t_end_ps: float | None = None) -> NetlistGraph:
    """Wrap the storage loop: a single SFQ data drive on ``D`` (default one pulse
    at 200 ps), probing BIN and BST. Trapping signature = BIN slips, BST holds."""
    if data_ps is None:
        data_ps = [200.0]
    if t_end_ps is None:
        t_end_ps = (max(data_ps) + 300.0) if data_ps else 500.0
    g = NetlistGraph()
    cell = build_storage_loop(spec)
    for it in cell.items:
        g.add(it)
    g.add(Source(name="VD", nodes=["D", "0"], letter="V",
                 func=_sfq_pulse_train(data_ps)))
    g.add(RawLine(text=f".tran 0.1p {t_end_ps:g}p 0 0.1p"))
    g.add(RawLine(text=".print PHASE BIN"))
    g.add(RawLine(text=".print PHASE BST"))
    g.add(RawLine(text=".end"))
    return g


# --------------------------------------------------------------------------- #
# Realistic SFQ source: a feeder junction that LAUNCHES a fluxon.
# --------------------------------------------------------------------------- #
#
# The stiff voltage-pwl drive (_sfq_pulse_train) forces exactly Phi0 of flux into
# the input node regardless of what the cell does — an idealization that can pin
# the loop's trapping behavior at an unphysical betaL. A real RSFQ cell instead
# receives its pulse from an *upstream junction*: when that junction's current is
# pushed past Ic, it performs exactly one 2*pi phase slip and emits one fluxon,
# which then propagates downstream through a series inductor (a JTL link). The
# downstream cell sees a current-mode fluxon — softer, finite source impedance,
# and only as much flux as the loop will accept. This is the physically faithful
# input we want before trusting any betaL-dependent margin number.


@dataclass
class FeederSpec:
    """A current-triggered SFQ source: one biased junction BFEED that emits exactly
    one fluxon when a brief trigger current pushes it over Ic, then feeds the cell
    through an output inductor (a real JTL-style link, not a stiff voltage drive)."""
    area: float = 2.0             # feeder JJ area -> Ic = 100uA/um^2 * area
    bias_ua: float = 130.0        # standing bias (held below Ic so it idles quietly)
    trigger_ua: float = 200.0     # trigger pulse height (bias+trigger must cross Ic;
                                  # >=180uA fires a single clean fluxon at bias 130uA)
    trigger_ps: float = 200.0     # when the trigger fires
    trigger_width_ps: float = 4.0 # trigger half-width (brief -> single slip)
    l_out_ph: float = 2.0         # feeder -> downstream cell input inductor
    rb_ohm: float = 5.23
    lrb_ph: float = 0.086
    lpr_ph: float = 0.278
    bias_ramp_ps: float = 50.0


def _trigger_pulse(peak_ua: float, t_center_ps: float, half_ps: float) -> str:
    """A brief triangular *current* pulse (uA) that rides on top of the standing
    bias to push the feeder junction over Ic exactly once."""
    t0, t1, t2 = t_center_ps - half_ps, t_center_ps, t_center_ps + half_ps
    return f"pwl(0 0 {t0:g}p 0 {t1:g}p {peak_ua:g}u {t2:g}p 0)"


def build_feeder(g: NetlistGraph, out_node: str, spec: FeederSpec,
                 tag: str = "FEED") -> None:
    """Attach a current-triggered SFQ feeder to an existing graph, launching one
    fluxon onto ``out_node`` at ``spec.trigger_ps``. Adds junction B{tag} at node
    f{tag}, its shunt, a standing bias, a trigger current pulse, and the output
    inductor f{tag} -> out_node. Probe B{tag} to confirm exactly one slip."""
    f = f"f{tag}"
    # feeder junction + damping shunt
    g.add(_jj(f"B{tag}", f, "0", spec.area))
    g.add(_ind(f"LRB{tag}", "0", f"q{tag}", spec.lrb_ph))
    g.add(_res(f"RB{tag}", f"q{tag}", f, spec.rb_ohm))
    # standing bias (ramped) injected into f
    g.add(_bias(f"IB{tag}", "0", f"bb{tag}", spec.bias_ua, spec.bias_ramp_ps))
    g.add(_ind(f"LPR{tag}", f, f"bb{tag}", spec.lpr_ph))
    # trigger current pulse straight into f -> one slip -> one fluxon out
    g.add(Source(name=f"IT{tag}", nodes=["0", f], letter="I",
                 func=_trigger_pulse(spec.trigger_ua, spec.trigger_ps,
                                     spec.trigger_width_ps)))
    # output link carrying the launched fluxon to the cell
    g.add(_ind(f"LO{tag}", f, out_node, spec.l_out_ph))


def build_feeder_testbench(spec: FeederSpec,
                           t_end_ps: float | None = None) -> NetlistGraph:
    """A standalone feeder driving a JTL-terminated load, to verify it emits exactly
    one fluxon. The feeder launches onto net ``out``; a load resistor terminates it."""
    if t_end_ps is None:
        t_end_ps = spec.trigger_ps + 300.0
    g = NetlistGraph()
    g.add(jmitll_model())
    build_feeder(g, "out", spec)
    g.add(_res("RLOAD", "out", "0", 2.0))
    g.add(RawLine(text=f".tran 0.1p {t_end_ps:g}p 0 0.1p"))
    g.add(RawLine(text=".print PHASE BFEED"))
    g.add(RawLine(text=".end"))
    return g


def build_storage_testbench_fed(spec: StorageLoopSpec, feeder: FeederSpec | None = None,
                                buffer: bool = True,
                                t_end_ps: float | None = None) -> NetlistGraph:
    """Storage loop driven by a REALISTIC feeder junction instead of the
    stiff voltage drive. The feeder launches one fluxon; with ``buffer=True`` it
    passes through one JTL buffer junction (BBUF) before reaching the loop, which
    isolates the source impedance from the loop's LQ so the test of the loop's
    *intrinsic* trapping betaL isn't confounded by source loading. Probes BFEED
    (source), BBUF (buffer), BIN, BST."""
    if feeder is None:
        feeder = FeederSpec()
    if t_end_ps is None:
        t_end_ps = feeder.trigger_ps + 300.0
    g = NetlistGraph()
    g.add(jmitll_model())
    # storage loop: two junctions joined by LQ (loop = a-LQ-b-BST-gnd-BIN-a).
    _loop_jj(g, "IN", "a", spec.area, spec.bias_in_ua, spec)
    g.add(_ind("LQ", "a", "b", spec.lq_ph))
    _loop_jj(g, "ST", "b", spec.area, spec.bias_st_ua, spec)
    probes = ["BFEED"]
    if buffer:
        # feeder -> BUF junction at node p -> series L -> loop input a.
        build_feeder(g, "p", feeder)
        _loop_jj(g, "BUF", "p", spec.area, spec.bias_in_ua, spec)
        g.add(_ind("LBUF", "p", "a", spec.l_in_ph))
        probes.append("BBUF")
    else:
        build_feeder(g, "a", feeder)
    probes += ["BIN", "BST"]
    g.add(RawLine(text=f".tran 0.1p {t_end_ps:g}p 0 0.1p"))
    for jj in probes:
        g.add(RawLine(text=f".print PHASE {jj}"))
    g.add(RawLine(text=".end"))
    return g
