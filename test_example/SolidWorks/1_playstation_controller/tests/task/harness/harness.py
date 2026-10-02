#!/usr/bin/env python3
"""
Task 1 (SolidWorks) -- PS3 controller: widen 15 mm + left-handed mirror.

Grades a candidate .SLDPRT against prompt/input.json (frozen measurements of
PS3-Controller_Base.SLDPRT).  Grading is geometry-only: mass properties,
inertia signs, centroids, face-area moments -- never feature or body names
(body order is unstable and candidates may remodel).

The controller's mirror plane is x = plane_x_m (NOT x=0).  A correct edit:
  - widens by 15 mm total (mirror-pair separations grow +15 mm)
  - moves the dpad diamond to the right and the face-button diamond to the
    left of the plane (left-handed layout), each as a RIGID unit
  - keeps chiral one-sided housing features (port lights, text) on their
    original side -- the housing is updated, not mirrored wholesale
  - adds no interference among the controls
  - changes nothing else (Y/Z spans, non-control body shapes)

--------------------------------------------------------------------------
SCORING MODEL
--------------------------------------------------------------------------
Every criterion returns a continuous subscore in [0, 1], so near-misses
separate from each other and from outright failures.

  * UNITS.  The SolidWorks API is metric SI: metres, m^2, m^3.  All scoring
    formulas work in MILLIMETRES; conversion happens exactly once, at the
    point of measurement, via MM.  Do not mix the two -- a millimetre
    tolerance applied to a metre value silently zeroes every score.

  * AGGREGATION.  Multi-item criteria average the PER-ITEM SCORES, never the
    per-item errors.  Averaging errors lets +30 mm on one pair cancel 0 mm on
    another and awards full marks to a deformed part.

  * C1 is asymmetric about the target.  Overshoot means the operation was
    performed and mis-sized; undershoot to zero means it was not performed.
    Undershoot reaches 0 at delta=0, overshoot decays more slowly.

  * C1 and C2 are independent.  C2 measures cluster placement against the
    widening the candidate ACTUALLY applied (half_actual, derived from the
    measured pair growth), not against the nominal +7.5 mm.  "Is the widening
    the right size?" is C1; "are the clusters consistent with the shell?" is
    C2.  One mistake is therefore charged once.

  * C2 and C4 do not overlap.  C2 owns geometry (placement + rigidity), C4
    owns sidedness.

  * C4 is multiplicative.  Its witnesses are of two kinds.  POSITIVE ones
    ("the conversion was performed"): cluster sides, body chirality.  GUARDS
    ("...and not by mirroring the whole part"): the port-light glyph cluster,
    the housing side signature.  Under a flat mean a candidate that did
    nothing still satisfies both guards, having never mirrored the housing,
    and collects their weight.  Guards are therefore a multiplier on positive
    credit, floored at GUARD_FLOOR, and can only subtract.  The port-light
    cluster dominates the guard: it is the sole detector of the naive flip
    instruction.md warns about.

  * ROLES ARE WEIGHTED (ROLE_WEIGHT).  instruction.md is about the button
    clusters; the stick/trigger/bumper pairs are already what C1 grades.

  * UNVERIFIABLE IS NOT FREE.  Unreadable witnesses drop out of their
    weighted mean, but a wholly unreadable class resolves to
    NEUTRAL_UNVERIFIABLE (0.5), never 1.0 -- otherwise destroying the
    evidence is a winning strategy, which an RL policy will find.

  * HEALTH GATE KEEPS THE ENVELOPE SHAPE.  A failed gate zeroes all five
    criteria but still emits all five subscores, so max_score stays constant
    and reports remain comparable across candidates.

  * BODY MATCHING IS INDEPENDENT OF THE HYPOTHESES IT FEEDS.  Matching runs
    on shape fingerprints (translation- and mirror-invariant) with position
    only as a tie-break, and that tie-break uses the MIRROR hypothesis alone
    -- never the widening hypothesis C1 grades.  Both the mirrored and the
    identity hypothesis are costed and the cheaper one wins, so a candidate
    that did not mirror is matched body-to-itself.  The chosen hypothesis and
    its margin over the runner-up are reported.

  * A LABEL THE HARNESS CANNOT MEASURE IS NOT USED.  The two button diamonds
    are told apart by the small faces the d-pad's engraved arrows add.  When
    a candidate leaves them indistinguishable the label falls out of body
    enumeration order, which is not stable, so the identity is refused rather
    than guessed: the role whose signature has no counterpart is scored as
    missing, and the surviving one keeps the diamond nearer its expected
    position.  See Grader._resolve_button_clusters().

Scoring weights live in ALL_CRITERIA below and nowhere else.  task.toml used
to carry a second copy of them, needed because a Harbor verifier container
is given only solution/ and tests/ and so never sees that file.  Two copies
with nothing comparing them is worse than one: task.toml now carries
max_score and the harness carries the rubric.

CLI (run `harness.py --help` for the full list):
    python harness.py [candidate.SLDPRT]      grade (no arg = active document)
    python harness.py --capture-only P        measure only, store the capture
    python harness.py --score-from cap.json   score a capture, no SolidWorks
    python harness.py --batch A B DIR ...     grade many, tabulate, write files
    python harness.py --capture-baseline P    re-freeze prompt/input.json
    python harness.py --capture-seed-rebuild  refresh only the rebuild census
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent


def _find_repo_root(start: Path) -> Path:
    """Directory holding the shared `common` package.

    Searched for rather than hard-coded by depth: the harness ships at two
    different tree depths in this repo, so a fixed `parents[N]` breaks on one
    of them.
    """
    d = start
    for _ in range(8):
        if (d / "common" / "__init__.py").is_file():
            return d
        if d.parent == d:
            break
        d = d.parent
    return start.parents[4] if len(start.parents) > 4 else start


_REPO_ROOT = _find_repo_root(HERE)
#: `/opt` is where `common/` sits inside a Harbor verifier
#: container, which has no repository around it to find.
for _p in ("/opt", str(_REPO_ROOT / "SolidWorks"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import solidworks_measure as M                      # noqa: E402
from common.solidworks_measure import z                         # noqa: E402
from common import solidworks_capture as SC                     # noqa: E402
from common import solidworks_session as SW                     # noqa: E402
from common import harness_cli as HC                            # noqa: E402
from common import harness_base as HB                           # noqa: E402
from common.harness_base import (Harness, finalize,             # noqa: E402
                                 score_error, write_env)

BASELINE_PATH = TASK_DIR / "prompt" / "input.json"
SOLUTION_REFERENCE_PATH = HERE / "solution_reference.json"

PASS, PARTIAL, FAIL, UNVERIFIABLE = "PASS", "PARTIAL", "FAIL", "UNVERIFIABLE"

HARNESS_VERSION = "2.3.1"
# Bumped whenever capture() changes what it records or what a field means.
# /3 changed plane_x_m from "the baseline's plane" to "the plane this part
# actually has". A /2 capture still scores correctly -- its plane equals the
# baseline's, so the normalisation is a no-op -- but it cannot exercise the
# translation handling, hence the warning rather than a refusal.
CAPTURE_SCHEMA = "ps3-capture/4"

#: HOW FAR THE MEASURED MIRROR MAY SIT from where the seed's mirror
#: travelled to, before it is called a mis-detection rather than an
#: edit. The widening this task asks for moves CONTROLS apart; it
#: does not move the plane they are mirrored about. The slips seen
#: in practice are tens of millimetres (80.5 -> 137.9, -> 122.9),
#: so 25 mm separates them from anything an author would do on
#: purpose without a re-freeze.
PLANE_SLIP_MAX_MM = 25.0

# The frozen seed measurement every candidate is compared against.
# /2 added the modelling census and the per-role bounding boxes: without them
# the hygiene criterion silently degrades to "new rebuild warnings only", and
# nothing in the score says the seed is at fault. A /1 baseline still grades,
# so the mismatch is reported rather than refused.
BASELINE_SCHEMA = "ps-annotation-baseline/3"

# Metres -> millimetres.  Applied once, where a measurement is taken.
MM = 1000.0

POLICY = {
    "width_delta_mm": 15.0,      # +15 mm total
    "half_mm": 7.5,              # 7.5 mm per side (nominal)
    "handedness": "mirror about plane_x (dpad <-> face_buttons swap sides)",
}

C_HEALTH = "rebuild health"
C_HYGIENE = "modelling hygiene"
C_MARKINGS = "markings preserved"
#: The rubric, in the order every report and batch column reads it. These
#: numbers were duplicated in task.toml, which a Harbor verifier container
#: never sees -- so the harness already carried them against exactly that
#: case, and the file's copy was the one that could go stale. One copy now.
#: WHAT TO KNOW ABOUT THIS TASK AND A JUDGEMENT. Every harness in
#: this repository declares one of these, the ones that use no
#: judge included: an absence cannot be told apart from an
#: oversight, and `harness.py --judge` prints what it finds here.
#: See `judge_runner.stance`.
JUDGE_NOTE = (
    "The one thing here that no measurement settles is also invisible "
    "to a judge. The port-light glyphs are left-right symmetric split "
    "faces, so a mirror of them is geometrically undetectable, and "
    "whether a standard part is the right standard part is a name "
    "question this repository grades nothing by. What is left -- a "
    "shell widened without its controls moving -- is measurable and "
    "simply not scored yet; the X span is already in the capture, "
    "marked diagnostic. A wider witness base, not an opinion, is what "
    "this task is short of."
)


ALL_CRITERIA = {
    C_HEALTH: 0.5,
    C_HYGIENE: 0.5,
    "widened by 15 mm": 1.5,
    "clusters at mirrored positions": 1.5,
    "no new control interference": 0.5,
    "left-handed layout achieved": 2.0,
    "no unrequested changes": 0.5,
    # instruction.md: "any text, logos, and standard/purchased parts (e.g.
    # joystick caps) must remain legible and correctly oriented". The
    # naive-flip guard inside `left-handed layout achieved` reads the
    # "correctly oriented" half. This reads the "remain" half, which had no
    # reader at all -- a candidate could erase all four face-button symbols
    # and score 7.000. Its own criterion rather than a component of `no
    # unrequested changes`, which is worth 0.5 in total and cannot express
    # this without being re-weighted itself; max_score becomes 8.0.
    C_MARKINGS: 1.0,
}
GEOMETRY_CRITERIA = tuple(k for k in ALL_CRITERIA
                          if k not in (C_HEALTH, C_HYGIENE))

TOL = {
    # -- C0: rebuild health, graded on the fraction of features newly
    # broken so it transfers to trees of a different size -------------------
    "health_perfect_frac": 0.0,
    "health_zero_frac": 0.20,
    # -- C0: modelling hygiene, all deltas against the seed -----------------
    "warn_perfect": 0.0,
    "warn_zero": 5.0,
    "sketch_perfect": 0.0,      # sketches in a status the seed never had
    "sketch_zero": 5.0,
    # Calibrated: the reference suppresses one feature as part of its own
    # edit, so a growth of one must not cost marks. Revisit once the corpus
    # contains a candidate that suppresses features to dodge a check.
    "suppressed_perfect": 1.0,
    "suppressed_zero": 6.0,
    "extref_perfect": 0.0,      # new external references
    "extref_zero": 3.0,
    # -- C1: widening, asymmetric about the +15 mm target -------------------
    "width_perfect_mm": 0.5,     # full marks inside +/-0.5 mm
    "width_zero_under_mm": 15.0,   # undershoot: 0.0 when delta == 0
    "width_zero_over_mm": 22.5,    # overshoot:  0.0 when delta == 37.5
    # -- C2: cluster geometry ----------------------------------------------
    "intra_perfect_mm": 0.5,     # rigid-diamond internal spacing
    "intra_zero_mm": 5.0,
    # Calibrated against solution.SLDPRT, which places its clusters 3.00 mm
    # (dpad) and 2.52 mm (face_buttons) from the ideal mirror+widen point:
    # the reference remodels rather than translating rigid bodies, so a few
    # millimetres of drift is correct behaviour, not error.
    "cluster_perfect_mm": 4.0,   # cluster centre vs adaptively expected pos
    "cluster_zero_mm": 12.0,
    # -- C3: interference ---------------------------------------------------
    "intf_perfect_m3": 5e-9,     # 5 mm^3 of new control interference is free
    "intf_zero_m3": 5e-8,        # 50 mm^3 scores nothing
    # -- C4: handedness witnesses ------------------------------------------
    # A rigid X translation smaller than this is measurement noise, not a
    # moved origin, and is left alone.
    "plane_shift_ignore_mm": 1.0,
    "straddle_mm": 12.0,         # cluster centred on the plane = no side
    "chiral_floor": 1e-3,        # inertia-product ratio below = unverifiable
    "sig_floor_frac": 0.2,       # candidate |moment3| vs baseline's to count

    # -- housing engraved-detail side (the second naive-flip guard) --------
    # Faces this small are markings, not structure: a re-shell adds large
    # interior faces and none of these, which is what makes the witness
    # survive the remodel that kills sig_floor_frac above.
    # UNVERIFIABLE below the floor rather than scored: an even left/right
    # split carries no side, and reading noise as a verdict is worse than
    # admitting the measurement failed.
    "detail_asym_floor": 0.05,      # |right-left| / total, below = no signal
    "detail_deadband_mm": 2.0,      # faces nearer the plane have no side
    "housing_remodel_fp": 0.5,   # fingerprint drift above = housing remodeled
    # -- C5: unrequested changes -------------------------------------------
    # Calibrated: the reference drifts 1.24 mm in Z and 0.01 mm in Y, which
    # is its remodelling noise rather than an unrequested change.
    "span_perfect_mm": 2.0,      # Y/Z span drift
    "span_zero_mm": 6.0,
    # Calibrated tight: measured worst-case fingerprint drift on the
    # non-exempt bodies is 2e-05 across every model that grades, i.e.
    # sticks/triggers/bumpers come through untouched.  A wide band here would
    # would leave several percent of trigger deformation unscored.
    "fp_perfect": 0.01,
    "fp_zero": 0.08,

    # Finished-housing reference comparison for C5.
    # Geometry-only: volume, area and inertia; no body/feature names.
    "housing_ref_fp_perfect": 0.00010,
    "housing_ref_fp_zero": 0.00150,
    "housing_ref_bbox_mm": 0.10,
    # -- body matching ------------------------------------------------------
    "match_fp_weight": 1.0,      # cost per unit of fingerprint distance
    "match_pos_weight_per_mm": 0.02,   # cost per mm of positional residual
}

MIRROR_ROLES = ["dpad", "face_buttons", "sticks", "triggers", "bumpers",
                "centre"]
# Clusters with a determinate expected position under mirror+widen: only the
# two button diamonds.
#
# A mirror-symmetric PAIR (sticks, triggers, bumpers) is invariant as a set
# under reflection -- its mean centre sits on the plane before and after --
# so "did this cluster reach its mirrored position?" is trivially true for
# pairs regardless of what the candidate did, and duplicates what c1 already
# measures via their separation.  Pair handedness is real but unreadable from
# position; body chirality witnesses it in c4 instead.
#
# centre (select/start/ps) is excluded for a different reason: it hugs the
# plane, and "adjust spacing accordingly" is satisfiable by mirror-only or
# mirror+spread -- the reference uses mirror-only.
POSITION_ROLES = ["dpad", "face_buttons"]
CHIRAL_ROLES = ["sticks", "triggers", "bumpers", "dpad"]
RIGID_ROLES = ["dpad", "face_buttons"]
# engineer-remodel exemptions for the reshape check
RESHAPE_EXEMPT = ["dpad", "face_buttons", "housing", "centre"]
# pairs whose separation witnesses the widening
WIDTH_PAIR_ROLES = ("sticks", "triggers", "bumpers")

# Relative importance of each role, used by cluster placement (c2) and
# cluster sides (c4).  instruction.md is about the button clusters; the
# stick/trigger/bumper pairs are already the subject of c1.  Equal weighting
# would price skipping an entire button cluster at a 1/5 penalty.
ROLE_WEIGHT = {
    "dpad": 0.25,
    "face_buttons": 0.25,
    "sticks": 0.15,
    "triggers": 0.15,
    "bumpers": 0.15,
    "centre": 0.05,
}

# Handedness evidence splits into two kinds that must NOT be averaged
# together:
#   POSITIVE witnesses say "the left-hand conversion was performed".
#   GUARD witnesses say "...and it was not done by mirroring the whole part".
# Averaged flat, a candidate that did nothing collects the guard weight for
# inaction (never having mirrored the housing, it passes both guards).
# Multiplying instead keeps the incentive pointing the right way: guards can
# only REDUCE credit that positive evidence earned.
HANDEDNESS_POSITIVE = {"cluster_sides": 0.6, "body_chirality": 0.4}
# Guard weights. No single witness may exceed half, and that is the whole
# point: when one dies the remainder still carries the guard at full strength
# (weighted_evidence renormalises over what is readable), instead of the class
# going wholly unreadable and falling back to NEUTRAL_UNVERIFIABLE -- which
# paid a candidate MORE for destroying the evidence than for being caught.
#: THE GUARD IS TWO WITNESSES, NOT THREE. `housing_detail_side` used to
#: carry 0.3 here on the reasoning that a naive whole-part mirror drags the
#: engraved labels across the plane with everything else. It does -- and so
#: does a correct conversion, because instruction.md asks for the
#: START/SELECT labels to "end up in mirrored positions". Measured on this
#: corpus the naive-flip adversary reads +0.1901 and the reference +0.2025:
#: the two the guard exists to separate are indistinguishable on it, while
#: the only model it passed was the one that never converted the controller
#: at all. Area balance cannot tell a relocated label from a mirrored one;
#: only orientation can, and it does not measure orientation. It is still
#: computed and reported -- see DETAIL_SIDE_IS_DIAGNOSTIC -- with no weight.
HANDEDNESS_GUARD = {"port_lights_side": 0.7,
                    "housing_side_signature": 0.3}
# Fraction of positive credit a naive whole-part flip keeps.  Non-zero
# because such a candidate did produce a left-handed layout; small because
# instruction.md names this failure explicitly.
GUARD_FLOOR = 0.15
# Rigidity qualifies cluster placement rather than standing beside it: summed
# 50/50 it awards 0.5 to a candidate that never moved a cluster (undeformed,
# because untouched).  As a floored multiplier: placed and rigid = 1.0,
# placed but deformed = 0.5, never placed = 0.
RIGID_FLOOR = 0.5
# Used when no witness of a kind can be read at all.  0.5 rather than 1.0:
# "nothing could be checked" must not pay better than "checked and half
# right", or destroying the evidence becomes a winning strategy.
NEUTRAL_UNVERIFIABLE = 0.5

#: Fraction of the seed's face-button engraving a candidate must keep for
#: full marks on C_MARKINGS. Every model in this corpus that keeps its
#: symbols keeps 50 of 50 faces and every model that does not keeps 0, so
#: nothing in the data forces a particular value; 0.75 leaves room for a
#: candidate that re-cuts the symbols with slightly different topology
#: without leaving room for one that removes them.
MARKINGS_FULL_FRACTION = 0.75

# body-volume windows (m^3) used by role assignment
BUTTON_VOL_MIN = 0.4e-6
BUTTON_VOL_MAX = 5.0e-6
PAIR_VOL_MIN = 5.0e-6
DIAMOND_LINK_M = 0.035          # X-Z proximity that chains a button diamond
HOUSING_FRACTION = 0.5          # of the largest body's volume
SMALL_FACE_AREA = 15e-6         # arrow/glyph engraving faces (dpad witness)
# Mean small-face count by which the two button diamonds must differ before
# their identity counts as measured rather than guessed. The seed separates
# them by 11.0 (d-pad arrows) against 0.0 (plain face buttons), so anything
# from ~1 upwards is safely clear of measurement noise while still refusing
# to call a dead heat.
CLUSTER_ID_MIN_SEP = 1.0

#: HOW DIFFERENT TWO BUTTON DIAMONDS MUST LOOK to be called different
#: clusters, as the largest relative difference among (plan aspect, plan
#: footprint, diamond radius).  Measured across this corpus: the seed's own
#: two clusters are 0.337 apart, every correctly matched pair is within
#: 0.014, every mismatched pair is at least 0.675, and the models that lost
#: a cluster sit at exactly 0.000.  0.10 is 7x the worst match and 6.7x
#: below the best mismatch.
CLUSTER_SHAPE_MIN_SEP = 0.10

# port-light glyph cluster constants
LIGHT_FACE_MAX_AREA = 8e-6
LIGHT_MIN_OFFPLANE_M = 0.040
LIGHT_AREA_TOL_FRAC = 0.02
LIGHT_CLUSTER_RADIUS_M = 0.020
LIGHT_MIN_MATCHES = 3


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------

def clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v != v:                                  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def mean_scores(values, default=1.0):
    """Mean of per-item SCORES.

    Never call this on per-item errors: averaging errors lets a +30 mm
    mistake on one pair cancel a 0 mm mistake on another.
    """
    vals = [clamp01(v) for v in values]
    return sum(vals) / len(vals) if vals else default


def status_of(score, floor=1e-9):
    if score >= 1.0 - floor:
        return PASS
    if score <= floor:
        return FAIL
    return PARTIAL


def asymmetric_target_score(value_mm, target_mm, perfect_mm,
                            zero_under_mm, zero_over_mm):
    """Score a measurement against a target, penalising undershoot harder.

    Symmetric |value - target| would give "did nothing" (0 mm) and "did it
    twice over" (30 mm) the identical score, which collapses two genuinely
    different candidates onto one number.  Overshoot demonstrates the
    operation was understood and performed at the wrong magnitude;
    undershoot to zero means it was never performed.
    """
    err = value_mm - target_mm
    if abs(err) <= perfect_mm:
        return 1.0
    if err < 0:
        return score_error(-err, perfect_mm, zero_under_mm)
    return score_error(err, perfect_mm, zero_over_mm)


def weighted_evidence(entries, neutral=NEUTRAL_UNVERIFIABLE):
    """Weighted mean over readable witnesses only.

    `entries` is {name: (weight, score_or_None)}; None means UNVERIFIABLE --
    dropped from both numerator and denominator.  When nothing is readable
    the caller gets `neutral`, never 1.0.

    Returns (score, diagnostics) where diagnostics records how much of the
    evidence weight actually survived, so a grade resting on one witness is
    visibly different from one resting on all of them.
    """
    total_w = sum(w for w, _ in entries.values()) or 1.0
    used = {k: (w, s) for k, (w, s) in entries.items() if s is not None}
    verified_w = sum(w for w, _ in used.values())
    frac = verified_w / total_w
    if not used:
        return neutral, {"verified_weight_fraction": 0.0,
                         "readable": [], "unreadable": sorted(entries),
                         "note": "no witness of this kind could be read; "
                                 f"defaulted to the neutral {neutral}"}
    if verified_w <= 0:
        # Readable, but every readable witness carries zero weight -- a
        # weighting choice, not a measurement. No opinion can be formed, so
        # this is the same situation as nothing being readable at all.
        return neutral, {"verified_weight_fraction": 0.0,
                         "readable": sorted(used),
                         "unreadable": sorted(k for k in entries
                                              if k not in used),
                         "note": "every readable witness of this kind is "
                                 f"weighted zero; defaulted to {neutral}"}
    score = sum(w * clamp01(s) for w, s in used.values()) / verified_w
    return score, {
        "verified_weight_fraction": round(frac, 3),
        "readable": sorted(used),
        "unreadable": sorted(k for k in entries if k not in used),
    }


def weighted_role_mean(scores_by_role, default=1.0):
    """Mean of per-role scores weighted by ROLE_WEIGHT, renormalised over
    whatever roles were actually measurable."""
    items = [(ROLE_WEIGHT.get(r, 0.1), s) for r, s in scores_by_role.items()]
    tot = sum(w for w, _ in items)
    if not items or tot <= 0:
        return default
    return sum(w * clamp01(s) for w, s in items) / tot


# --------------------------------------------------------------------------
# role assignment (pure geometry)
# --------------------------------------------------------------------------

#: How far apart two pair-midpoints may be and still be called the same
#: plane. Was implicit in `round(p * 1000)` plus a +-1.5 mm membership
#: window, which are not the same number and disagreed at bucket edges.
PLANE_WINDOW_MM = 1.5


def sym_plane(bodies, expected=None):
    """Mirror plane x from the mode of same-fingerprint pair midpoints.

    DETERMINISTIC, WHICH IT WAS NOT. The old body was

        buckets = Counter(round(p * 1000) for p in planes)
        best = buckets.most_common(1)[0][0]

    and `Counter.most_common` breaks ties by INSERTION ORDER. Insertion
    order here is `itertools.combinations(bodies, 2)`, so it is the order
    SolidWorks handed the bodies back -- which is not stable across
    opens. Measured on 27.09: the same .SLDPRT captured twice in one
    batch gave plane_x_m 0.13795 and 0.08059 (the part's real plane is
    0.0805). Everything plane-relative is then measured about the wrong
    mirror, and `clusters at mirrored positions` went 1.00 -> 0.00 --
    1.5 of 7.0, on identical geometry, run to run.

    Three changes, all of them about making the same input give the same
    answer:

      * each candidate plane is scored by HOW MANY midpoints fall within
        PLANE_WINDOW_MM of it, not by how many share its 1 mm bucket. A
        plane sitting on a bucket edge used to have its own support split
        between two buckets and could lose to a spurious cluster.
      * ties are broken by the value itself (and by distance to
        `expected` when a baseline plane is known), never by the order
        the pairs arrived in.
      * `expected` only orders ties. It cannot invent a plane the
        geometry does not support, so a part whose mirror genuinely
        moved is still measured where it actually is.
    """
    planes = []
    for a, b in itertools.combinations(bodies, 2):
        if abs(a["volume_m3"] - b["volume_m3"]) > 1e-9:
            continue
        if abs(a["area_m2"] - b["area_m2"]) > 1e-7:
            continue
        ca, cb = a["centroid_m"], b["centroid_m"]
        if abs(ca[1] - cb[1]) > 2e-3 or abs(ca[2] - cb[2]) > 2e-3:
            continue
        planes.append((ca[0] + cb[0]) / 2)
    if not planes:
        return None
    planes.sort()
    win = PLANE_WINDOW_MM / 1000.0

    def support(p):
        return sum(1 for q in planes if abs(q - p) <= win)

    def rank(p):
        # -support first; then nearest to the expected plane when one is
        # known; then the value, so equal candidates always order the
        # same way whatever order the pairs arrived in.
        return (-support(p),
                abs(p - expected) if expected is not None else 0.0,
                p)

    best = min(planes, key=rank)
    members = [q for q in planes if abs(q - best) <= win]
    return sum(members) / len(members)


def _diamond_groups(small):
    def near(a, b):
        ca, cb = a["centroid_m"], b["centroid_m"]
        return math.hypot(ca[0] - cb[0], ca[2] - cb[2]) < DIAMOND_LINK_M

    groups, used = [], set()
    for b in small:
        if b["id"] in used:
            continue
        stack, comp = [b], []
        used.add(b["id"])
        while stack:
            x = stack.pop()
            comp.append(x)
            for o in small:
                if o["id"] not in used and near(x, o):
                    used.add(o["id"])
                    stack.append(o)
        groups.append(comp)
    return groups


def _diamond_plan_shape(group):
    """(mean plan aspect, mean plan footprint mm2) of a four-body diamond.

    Plan, not volume: the reference rescales the buttons, and aspect and
    footprint in the XZ plane are what survive that. A body without a
    bounding box contributes an aspect of 1.0, which sorts it with the
    round buttons rather than inventing an elongation for it.
    """
    asp, foot = [], []
    for b in group:
        bb = b.get("bbox_m")
        if not bb or len(bb) < 6:
            asp.append(1.0)
            foot.append(0.0)
            continue
        dx, dz = abs(bb[3] - bb[0]) * MM, abs(bb[5] - bb[2]) * MM
        lo, hi = sorted((dx, dz))
        asp.append(hi / lo if lo > 0 else 1.0)
        foot.append(dx * dz)
    n = len(group) or 1
    return (sum(asp) / n, sum(foot) / n)


def assign_roles(bodies, small_face_count):
    """{role: [ids]} from geometry alone."""
    roles = {}

    bysize = sorted(bodies, key=lambda b: -b["volume_m3"])
    housing = [bysize[0]["id"]]
    for b in bysize[1:]:
        if b["volume_m3"] > HOUSING_FRACTION * bysize[0]["volume_m3"]:
            housing.append(b["id"])
    roles["housing"] = housing

    small = [b for b in bodies
             if BUTTON_VOL_MIN < b["volume_m3"] < BUTTON_VOL_MAX]
    diamonds = [g for g in _diamond_groups(small) if len(g) == 4]
    # THE D-PAD IS THE MORE ELONGATED DIAMOND, not the more engraved one.
    #
    # It used to be the one carrying more small engraved faces -- its arrows
    # -- which is a property of the seed rather than of a controller, and it
    # inverts the moment the round buttons are engraved. The grader
    # re-decides a CANDIDATE's labels by plan shape and could afford to
    # ignore this ordering, but the BASELINE is never re-resolved: Broles
    # comes straight out of prompt/input.json and is what every candidate's
    # clusters are positioned against. A seed frozen with its labels crossed
    # would measure every candidate against the wrong cluster.
    #
    #     d-pad arm     plan 13.8 x 17.0 mm   aspect 1.23   footprint 234 mm2
    #     round button  plan 18.8 x 18.8 mm   aspect 1.00   footprint 353 mm2
    #
    # A cross arm is a rectangle in plan; a round button's bounding box is
    # square by construction. Neither depends on what is engraved on it.
    #
    # Engraving and x-centroid are kept BELOW the shape keys. When the two
    # diamonds really are the same shape -- a candidate that dropped a copy
    # of one cluster where the other belonged -- the shape keys tie and
    # those still decide, so the capture stays reproducible instead of
    # falling out of SolidWorks' body enumeration order, which is not
    # stable. The grader does not trust a tied label either way; see
    # Grader._resolve_button_clusters().
    diamonds.sort(key=lambda g: (-round(_diamond_plan_shape(g)[0], 3),
                                 round(_diamond_plan_shape(g)[1], 1),
                                 -sum(small_face_count.get(b["id"], 0)
                                      for b in g) / len(g),
                                 sum(b["centroid_m"][0] for b in g) / len(g)))
    if len(diamonds) >= 2:
        roles["dpad"] = [b["id"] for b in diamonds[0]]
        roles["face_buttons"] = [b["id"] for b in diamonds[1]]
    elif len(diamonds) == 1:
        roles["dpad"] = [b["id"] for b in diamonds[0]]
        roles["face_buttons"] = []
    else:
        roles["dpad"], roles["face_buttons"] = [], []

    diamond_ids = {b["id"] for g in diamonds for b in g}
    roles["centre"] = [b["id"] for b in small if b["id"] not in diamond_ids]

    taken = set(housing) | diamond_ids | set(roles["centre"])
    mids = [b for b in bodies
            if b["id"] not in taken and b["volume_m3"] >= PAIR_VOL_MIN]
    pairs = []
    for a, b in itertools.combinations(mids, 2):
        if abs(a["volume_m3"] - b["volume_m3"]) > 1e-8:
            continue
        if abs(a["area_m2"] - b["area_m2"]) > 1e-6:
            continue
        if abs(a["centroid_m"][2] - b["centroid_m"][2]) > 3e-3:
            continue
        pairs.append((a, b))

    def pz(p):
        return (p[0]["centroid_m"][2] + p[1]["centroid_m"][2]) / 2

    used = set()
    if pairs:
        st = max(pairs, key=pz)          # sticks sit highest (z)
        roles["sticks"] = [st[0]["id"], st[1]["id"]]
        used = set(roles["sticks"])
        low = [p for p in pairs
               if p[0]["id"] not in used and p[1]["id"] not in used]
        low.sort(key=lambda p: -p[0]["volume_m3"])
        if low:                          # larger low pair = triggers
            roles["triggers"] = [low[0][0]["id"], low[0][1]["id"]]
            used |= set(roles["triggers"])
        if len(low) >= 2:
            roles["bumpers"] = [low[1][0]["id"], low[1][1]["id"]]
    return roles


# --------------------------------------------------------------------------
# capture helpers -- everything below knows about THIS part
#
# The task-agnostic half of capture lives in common/solidworks_capture.py
# (session attach, document open, body capture, feature and modelling
# census, glyph discovery). What stays here is what could only ever apply
# to a controller: which body is a d-pad, where the port-light glyphs sit,
# which pairs of controls may not interfere.
# --------------------------------------------------------------------------

def _small_face_counts(raw_bodies):
    out = {}
    for idx, b in enumerate(raw_bodies):
        n = 0
        for f in (z(b.GetFaces) or []):
            try:
                if float(z(f.GetArea)) < SMALL_FACE_AREA:
                    n += 1
            except Exception:
                continue
        out[f"c{idx:02d}"] = n
    return out


def housing_faces(raw_bodies, housing_ids):
    """One pass over housing faces: (area, box-centre xyz) per face."""
    out = []
    for hid in housing_ids:
        idx = int(hid[1:])
        body = raw_bodies[idx]
        for f in (z(body.GetFaces) or []):
            try:
                a = float(z(f.GetArea))
                box = z(f.GetBox)
                c = [(float(box[k]) + float(box[k + 3])) / 2 for k in range(3)]
            except Exception:
                continue
            out.append((a, c))
    return out


def housing_signature(faces, plane_x):
    """Area-weighted 3rd moment of housing face box-centres about x=plane.
    Sign flips under a whole-part mirror."""
    m3, area_total = 0.0, 0.0
    for a, c in faces:
        m3 += a * (c[0] - plane_x) ** 3
        area_total += a
    return {"moment3_m5": m3, "face_area_m2": area_total,
            "faces": len(faces), "sign": 1 if m3 >= 0 else -1}


def housing_detail_side(faces, plane_x):
    """Which side of the plane the housing's ENGRAVED detail sits on.

    housing_signature() above asks the same question over every housing
    face, and that is exactly why it dies on real candidates: widening by
    15 mm means re-shelling, a re-shell adds a great many large interior
    faces, and their distribution swamps the exterior asymmetry. The moment
    then measures the shell rather than the markings, its sign proves
    nothing, and the check reports UNVERIFIABLE -- on every candidate that
    actually did the work.

    Filtering to faces below SMALL_FACE_AREA removes precisely the faces a
    re-shell adds. What survives is engraving-scale detail -- glyphs, text,
    moulded markings -- which is one-sided by nature and must stay where it
    was. The statistic is an area-weighted left/right balance rather than a
    moment: scale-free, bounded in [-1, 1], and not dominated by whichever
    mark happens to sit furthest from the plane.

    Faces inside the deadband have no meaningful side and are dropped, so a
    marking centred on the plane cannot tip the result by rounding.
    """
    left = right = 0.0
    n_left = n_right = 0
    band = TOL["detail_deadband_mm"] / MM
    for a, c in faces:
        if a >= SMALL_FACE_AREA:
            continue
        d = c[0] - plane_x
        if abs(d) < band:
            continue
        if d > 0:
            right += a
            n_right += 1
        else:
            left += a
            n_left += 1
    total = left + right
    return {
        "asymmetry": ((right - left) / total) if total > 0 else 0.0,
        "area_right_m2": right, "area_left_m2": left,
        "faces_right": n_right, "faces_left": n_left,
        "max_face_area_m2": SMALL_FACE_AREA,
        "deadband_mm": TOL["detail_deadband_mm"],
    }


def find_light_cluster_seed(faces, plane_x, bbox):
    """Seed-side identification: tiny faces on the top-rear edge, well off
    the mirror plane."""
    y_top = bbox[4]          # y max
    z_rear = bbox[2]         # z min
    cand = [(a, c) for a, c in faces
            if a < LIGHT_FACE_MAX_AREA
            and abs(c[0] - plane_x) > LIGHT_MIN_OFFPLANE_M
            and (y_top - c[1]) < 0.030
            and (c[2] - z_rear) < 0.020]
    if len(cand) < LIGHT_MIN_MATCHES:
        return None
    best = None
    for a0, c0 in cand:
        grp = [(a, c) for a, c in cand
               if math.hypot(c[0] - c0[0], c[2] - c0[2])
               < LIGHT_CLUSTER_RADIUS_M]
        if best is None or len(grp) > len(best):
            best = grp
    if not best or len(best) < LIGHT_MIN_MATCHES:
        return None
    cx = sum(c[0] for _, c in best) / len(best)
    return {
        "areas_m2": sorted(a for a, _ in best),
        "centroid_m": [sum(c[k] for _, c in best) / len(best)
                       for k in range(3)],
        "side": 1 if cx >= plane_x else -1,
        "n_faces": len(best),
    }


def find_light_cluster_candidate(faces, plane_x, seed_cluster):
    """Candidate-side: match the seed's area multiset among housing faces."""
    if not seed_cluster:
        return None
    hits = []
    for want in seed_cluster["areas_m2"]:
        best, bd = None, None
        for a, c in faces:
            d = abs(a - want) / max(want, 1e-30)
            if d < LIGHT_AREA_TOL_FRAC and (bd is None or d < bd):
                best, bd = (a, c), d
        if best:
            hits.append(best)
    if len(hits) < LIGHT_MIN_MATCHES:
        return None
    best = None
    for a0, c0 in hits:
        grp = [(a, c) for a, c in hits
               if math.hypot(c[0] - c0[0], c[2] - c0[2])
               < LIGHT_CLUSTER_RADIUS_M]
        if best is None or len(grp) > len(best):
            best = grp
    if not best or len(best) < LIGHT_MIN_MATCHES:
        return None
    cx = sum(c[0] for _, c in best) / len(best)
    return {
        "areas_m2": sorted(a for a, _ in best),
        "centroid_m": [sum(c[k] for _, c in best) / len(best)
                       for k in range(3)],
        "side": 1 if cx >= plane_x else -1,
        "n_faces": len(best),
    }


def control_interference(raw_bodies, bodies, roles):
    """Boolean-intersect interference among control bodies and each control
    vs housing."""
    control_ids = []
    for role in ("dpad", "face_buttons", "sticks", "triggers", "bumpers",
                 "centre"):
        control_ids.extend(roles.get(role, []))
    housing_ids = roles.get("housing", [])

    idx_of = {b["id"]: i for i, b in enumerate(bodies)}
    test_pairs = []
    for a, b in itertools.combinations(control_ids, 2):
        test_pairs.append((a, b))
    for a in control_ids:
        for h in housing_ids:
            test_pairs.append((a, h))

    boxes = {bid: bodies[idx_of[bid]]["bbox_m"]
             for bid in set(control_ids) | set(housing_ids)}
    pairs, total, failures = [], 0.0, 0
    for a, b in test_pairs:
        if not M._boxes_overlap(boxes[a], boxes[b], M.BBOX_PAD):
            continue
        try:
            copy_a = z(raw_bodies[int(a[1:])].Copy)
            copy_b = z(raw_bodies[int(b[1:])].Copy)
            res, code = M._operations2(copy_a, copy_b)
            vol = 0.0
            if res:
                for rb in res:
                    try:
                        vol += float(rb.GetMassProperties(0.0)[3])
                    except Exception:
                        pass
            if vol > M.MIN_INTERFERENCE_VOLUME or code != 0:
                total += vol
                if code != 0:
                    failures += 1
                pairs.append({"a": a, "b": b, "volume_m3": vol,
                              "error_code": code})
        except Exception:
            failures += 1
            pairs.append({"a": a, "b": b, "volume_m3": 0.0,
                          "error_code": -1})
    return {"tested_pairs": len(test_pairs), "pairs": pairs,
            "total_volume_m3": total, "boolean_failures": failures}


def role_bboxes(bodies, roles):
    """Union bounding box per role, in metres.

    Costs nothing extra -- per-body boxes are already collected.  A role's own
    span is a far more specific witness of a resize than the global bounding
    box of every body, which is dominated by whichever control happens to
    stick out furthest.
    """
    by_id = {b["id"]: b for b in bodies}
    out = {}
    for role, ids in (roles or {}).items():
        boxes = [by_id[i]["bbox_m"] for i in ids if i in by_id]
        if not boxes:
            continue
        out[role] = ([min(b[k] for b in boxes) for k in range(3)]
                     + [max(b[k + 3] for b in boxes) for k in range(3)])
    return out


def capture(doc, baseline=None, plane_x=None):
    rebuild = SC.health_gate(doc, baseline)
    raw, bodies, boxes, gmin, gmax = SC.capture_bodies(doc)
    smf = _small_face_counts(raw)
    roles = assign_roles(bodies, smf)

    # The candidate's OWN symmetry plane is preferred over the baseline's.
    #
    # Everything plane-relative -- the housing moment, which side the
    # port-light glyphs sit on -- is meaningless if measured about a plane the
    # part no longer has. Translating the whole part is a legitimate edit (an
    # author may reset the origin), so the plane must follow the part rather
    # than the part being judged against a stale plane. Grader then normalises
    # the offset so positions stay comparable with the baseline.
    #
    # Sanity-checked before use: a plane outside the part's own X extent is a
    # mis-detection, not a translation, and the baseline is used instead.
    # The baseline's plane, carried forward by however far the part has
    # been translated. A legitimate edit may reset the origin, so the
    # plane is allowed to travel WITH the part -- what it may not do is
    # wander inside a part that has not moved.
    expected_plane = None
    bbb = ((baseline or {}).get("global") or {}).get("bbox_m")
    if baseline and baseline.get("plane_x_m") is not None:
        expected_plane = baseline["plane_x_m"]
        if bbb and len(bbb) >= 4:
            shift = ((gmin[0] + gmax[0]) / 2.0
                     - (float(bbb[0]) + float(bbb[3])) / 2.0)
            expected_plane += shift
    measured_plane = sym_plane(bodies, expected=expected_plane)

    # SANITY, AND THE OLD ONE WAS NOT ONE. "inside the part's own X
    # extent" accepted 137.95 mm on a part 323 mm wide -- every
    # mis-detection this task has ever produced passes that test. A plane
    # that has slipped is one that sits far from where the seed's plane
    # travelled to, on a part whose bounding box says it did not travel
    # that far.
    usable = (measured_plane is not None
              and gmin[0] <= measured_plane <= gmax[0])
    plane_slip_mm = None
    if usable and expected_plane is not None:
        plane_slip_mm = (measured_plane - expected_plane) * MM
        if abs(plane_slip_mm) > PLANE_SLIP_MAX_MM:
            usable = False
    P = plane_x if plane_x is not None else None
    plane_source = "argument"
    if P is None:
        if usable:
            P, plane_source = measured_plane, "measured"
        elif baseline and baseline.get("plane_x_m") is not None:
            P, plane_source = baseline["plane_x_m"], "baseline"
            if measured_plane is not None:
                plane_source = (
                    f"baseline (measured plane slipped "
                    f"{plane_slip_mm:+.1f} mm from the seed's, "
                    f"past the {PLANE_SLIP_MAX_MM:.0f} mm limit)"
                    if plane_slip_mm is not None
                    and abs(plane_slip_mm) > PLANE_SLIP_MAX_MM
                    else "baseline (measured plane outside part extent)")
        else:
            plane_source = "undetermined"

    hfaces = housing_faces(raw, roles.get("housing", []))
    if P is not None:
        sig = housing_signature(hfaces, P)
        detail = housing_detail_side(hfaces, P)
        seed_cluster = (baseline or {}).get("light_cluster")
        if seed_cluster:
            lights = find_light_cluster_candidate(hfaces, P, seed_cluster)
        else:
            lights = find_light_cluster_seed(hfaces, P, gmin + gmax)
    else:
        sig = {"moment3_m5": 0.0, "sign": 0, "faces": 0}
        detail = None
        lights = None

    intf = control_interference(raw, bodies, roles)
    return {
        "schema": CAPTURE_SCHEMA,
        "harness_version": HARNESS_VERSION,
        "document": str(z(doc.GetTitle)),
        "rebuild": rebuild,
        "modelling": SC.modelling_census(doc),
        "global": {"bbox_m": gmin + gmax,
                   "width_m": gmax[0] - gmin[0]},
        "plane_x_m": P,
        "plane_x_measured_m": measured_plane,
        "plane_source": plane_source,
        "bodies": bodies,
        "roles": roles,
        "role_bbox_m": role_bboxes(bodies, roles),
        "small_face_counts": smf,
        "housing_signature": sig,
        "housing_detail": detail,
        "light_cluster": lights,
        "interference": intf,
    }


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def d3(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def sgn(v):
    return 1.0 if v >= 0 else -1.0


def fingerprint(b):
    I = b["inertia_com"]
    return [b["volume_m3"], b["area_m2"]] + sorted(
        [I["Ixx"], I["Iyy"], I["Izz"]])


def fp_distance(a, b):
    s = 0.0
    fa, fb = fingerprint(a), fingerprint(b)
    for u, v in zip(fa, fb):
        sc = max(abs(u), abs(v), 1e-30)
        s += ((u - v) / sc) ** 2
    return math.sqrt(s / len(fa))


def chirality(b):
    I = b["inertia_com"]
    diag = max(abs(I["Ixx"]), abs(I["Iyy"]), abs(I["Izz"]))
    return 0.0 if diag <= 0 else max(abs(I["Ixy"]), abs(I["Izx"])) / diag


def chirality_witness(b):
    I = b["inertia_com"]
    return "Ixy" if abs(I["Ixy"]) >= abs(I["Izx"]) else "Izx"


def mirror_x(x, plane):
    """Reflect x about the plane -- no widening."""
    return 2 * plane - x


def mirror_widen_x(x, plane, half_m):
    """Reflect about the plane AND push outward by `half_m` metres."""
    o = x - plane
    return plane - sgn(o) * (abs(o) + half_m)


# --------------------------------------------------------------------------
# grader
# --------------------------------------------------------------------------

def normalise_plane_offset(baseline, measured):
    """Shift the candidate's X coordinates so both parts share a plane.

    A part translated bodily along X is still a valid answer -- the author may
    have reset the origin -- but every position comparison against the
    baseline would read as a gross error. Sliding the candidate by the
    difference between the two symmetry planes removes the rigid offset and
    leaves every relative measurement untouched.

    Quantities already computed about the candidate's own plane (the housing
    moment, the port-light side) are unaffected: capture() measured them
    against the plane the part actually has.

    Returns (measured, offset_m). The input is not mutated.
    """
    pb = baseline.get("plane_x_m")
    pc = measured.get("plane_x_m")
    if pb is None or pc is None:
        return measured, 0.0
    dx = pb - pc
    if abs(dx) * MM < TOL["plane_shift_ignore_mm"]:
        return measured, 0.0

    out = dict(measured)
    out["bodies"] = []
    for b in measured.get("bodies", []):
        nb = dict(b)
        nb["centroid_m"] = [b["centroid_m"][0] + dx] + list(b["centroid_m"][1:])
        bb = list(b.get("bbox_m") or [])
        if len(bb) == 6:
            bb[0] += dx
            bb[3] += dx
            nb["bbox_m"] = bb
        out["bodies"].append(nb)

    g = dict(measured.get("global") or {})
    gb = list(g.get("bbox_m") or [])
    if len(gb) == 6:
        gb[0] += dx
        gb[3] += dx
        g["bbox_m"] = gb
        out["global"] = g

    rb = {}
    for role, box in (measured.get("role_bbox_m") or {}).items():
        nbx = list(box)
        nbx[0] += dx
        nbx[3] += dx
        rb[role] = nbx
    if rb:
        out["role_bbox_m"] = rb

    lc = measured.get("light_cluster")
    if lc and lc.get("centroid_m"):
        nlc = dict(lc)
        nlc["centroid_m"] = [lc["centroid_m"][0] + dx] + \
            list(lc["centroid_m"][1:])
        out["light_cluster"] = nlc

    out["plane_x_m"] = pb
    out["plane_normalised_by_mm"] = round(dx * MM, 3)
    return out, dx


def ungradable_reason(baseline, measured):
    """Why this candidate cannot be measured at all, or None.

    Distinct from "the candidate did badly". A part that scores zero on every
    criterion has been measured and found wrong; a part that lands here could
    not be measured, so no score is meaningful. Both end at 0, but only the
    second is reported as ungradable, and the reason travels in the envelope
    instead of a traceback.
    """
    if measured.get("open_error"):
        return f"document could not be opened: {measured['open_error']}"
    if not measured.get("bodies"):
        return "part contains no solid bodies"
    if measured.get("plane_x_m") is None:
        return ("mirror plane could not be determined, from the part or from "
                "the baseline -- every side and position judgement depends "
                "on it")
    return None


class Grader:
    def __init__(self, baseline, measured):
        self.bl = baseline
        self.ref = load_solution_reference()
        # A bodily translation along X is normalised away before anything is
        # compared, so a part with a reset origin is judged on its geometry.
        measured, self.plane_shift_m = normalise_plane_offset(baseline,
                                                              measured)
        self.ms = measured
        self.P = baseline["plane_x_m"]
        self.B = {b["id"]: b for b in baseline["bodies"]}
        self.C = {c["id"]: c for c in measured["bodies"]}
        self.Broles = baseline["roles"]
        self.Croles = dict(measured["roles"])
        self.cluster_identity = self._resolve_button_clusters()
        self.match_diag = {}
        self.match = self._match()
        self._pair_rows_cache = None
        # Roles present in the baseline that the candidate cannot supply a
        # full set of bodies for -- deleted, or remodelled past recognition.
        # Either way the requested edit is not demonstrated, so these score
        # zero.  Skipping them instead would let a candidate delete a whole
        # cluster and be graded as having nothing to check.
        self.unmatched_roles = {}
        for role, bids in self.Broles.items():
            if not bids:
                continue
            missing = [b for b in bids if not self.match[b].get("cand")]
            if missing:
                self.unmatched_roles[role] = {
                    "baseline_bodies": len(bids),
                    "candidate_bodies": len(self.Croles.get(role, [])),
                    "unmatched": len(missing),
                }

    def _role_unmatched(self, role):
        return role in self.unmatched_roles

    # -- button-cluster identity -----------------------------------------
    def _cluster_signature(self, capture, roles, role):
        """Mean small engraved faces per body of a button diamond.

        NO LONGER USED FOR IDENTITY -- see _cluster_shape(). It used to be,
        on the reasoning that the arrows cut into the d-pad add small faces
        the round buttons do not have; that is true of the seed and false of
        the reference, which engraves the four glyphs on the buttons and
        simplifies the arrows, inverting the key. It is reported beside the
        shape because it remains the witness that separates `missing_glyphs`
        from a model that kept its glyphs.
        """
        ids = roles.get(role) or []
        if not ids:
            return None
        smf = capture.get("small_face_counts") or {}
        # _small_face_counts() emits a key for EVERY body, so a missing key
        # means the census was not taken, not that the body has no small
        # faces. Reading the gap as a measured zero would let two clusters
        # look identical whenever the data is simply absent -- and that is
        # exactly the condition this class treats as a destroyed d-pad.
        if any(i not in smf for i in ids):
            return None
        return sum(smf[i] for i in ids) / len(ids)

    def _cluster_shape(self, bodies, roles, role):
        """(plan aspect, plan footprint mm2, diamond radius mm), or None.

        WHAT THE ENGRAVING COUNT WAS STANDING IN FOR. `_cluster_signature`
        identified the d-pad as the diamond with more small engraved faces.
        That held for the seed, where the arrows are cut into the d-pad and
        the round buttons are bare -- and it inverted the moment a candidate
        engraved the four glyphs on the buttons, which is what the reference
        does.  The reference's buttons read 15/11/0/24 small faces against
        its d-pad's 5/5/5/5, so the d-pad was labelled `face_buttons`, the
        clusters looked unmoved, and a model that performed the swap scored
        4.300 of 7.0 for it.

        Shape does not invert.  A d-pad arm is a rectangle in plan, 13.8 by
        17.0 mm, aspect 1.23; a round button's bounding box is square by
        construction, 18.8 by 18.8, aspect 1.00 and half again the
        footprint.  Nothing engraved on a face changes either, and nothing
        this task asks for does: widening moves the clusters apart, it does
        not reshape a button.

        Returned as a vector rather than a scalar so the caller can MATCH
        against the seed instead of thresholding.  A candidate that rebuilds
        both clusters at a different size still matches the seed's pairing
        correctly; an absolute cut-off would not survive that.
        """
        ids = roles.get(role) or []
        if len(ids) < 2:
            return None
        bs = [bodies.get(i) for i in ids]
        if any(b is None or not b.get("bbox_m") for b in bs):
            return None
        cx = sum(b["centroid_m"][0] for b in bs) / len(bs)
        cz = sum(b["centroid_m"][2] for b in bs) / len(bs)
        rad = sum(math.hypot(b["centroid_m"][0] - cx,
                             b["centroid_m"][2] - cz) for b in bs) / len(bs)
        asp, foot = [], []
        for b in bs:
            bb = b["bbox_m"]
            dx, dz = (bb[3] - bb[0]) * MM, (bb[5] - bb[2]) * MM
            lo, hi = sorted((abs(dx), abs(dz)))
            if lo <= 0:
                return None
            asp.append(hi / lo)
            foot.append(abs(dx * dz))
        return (sum(asp) / len(asp), sum(foot) / len(foot), rad * MM)

    @staticmethod
    def _shape_dist(a, b):
        """Largest relative difference between two cluster shapes."""
        if a is None or b is None:
            return None
        return max(abs(x - y) / max(abs(x), abs(y), 1e-9)
                   for x, y in zip(a, b))

    def _cluster_centre_x(self, ids):
        return sum(self.C[i]["centroid_m"][0] for i in ids) / len(ids)

    def _expected_cluster_x(self, role):
        """Where the mirror alone would put this cluster's centre.

        Widening is deliberately left out, the same way _perm_cost() leaves it
        out: this is a tie-break between two diamonds ~176 mm apart, and
        folding in the candidate's widening would make c2's own answer depend
        on c1's.
        """
        bids = self.Broles.get(role) or []
        if not bids:
            return None
        return sum(mirror_x(self.B[i]["centroid_m"][0], self.P)
                   for i in bids) / len(bids)

    def _resolve_button_clusters(self):
        """Decide which candidate diamond is the d-pad, or refuse to.

        assign_roles() labels the two diamonds by their engraving signature
        and falls back to position when that ties. A tie is not a detail: it
        means the candidate no longer has two distinguishable button
        clusters. Accepting the fallback label would make the score depend on
        which way the tie fell -- on this corpus that moves c2 between 0.0 and
        1.0 for the same part -- and would report a confident reason for it.

        Three outcomes:

        * candidate separates the clusters -> trust the signature, and force
          the labels to follow it even if assign_roles() ordered them the
          other way round;
        * seed cannot separate them either -> nothing to measure, leave the
          labels alone and say so;
        * seed separates them and the candidate does not -> the role whose
          signature has no counterpart is GONE. It is emptied so the existing
          unmatched-role machinery scores it zero, and the surviving role
          keeps whichever diamond fits its expected position best, so the
          coin toss cannot also cost the candidate the cluster it did place
          correctly.
        """
        roles = ("dpad", "face_buttons")
        if not all(self.Croles.get(r) for r in roles):
            return None
        if not all(self.Broles.get(r) for r in roles):
            return None

        bsh = {r: self._cluster_shape(self.B, self.Broles, r)
               for r in roles}
        csh = {r: self._cluster_shape(self.C, self.Croles, r)
               for r in roles}
        if any(v is None for v in list(bsh.values()) + list(csh.values())):
            return None

        seed_sep = self._shape_dist(bsh["dpad"], bsh["face_buttons"])
        cand_sep = self._shape_dist(csh["dpad"], csh["face_buttons"])

        # Kept as evidence, no longer as identity.  It is still what tells
        # `missing_glyphs` from the reference; it is simply not what tells a
        # d-pad from a button, because a candidate is free to engrave the
        # buttons and the reference does.
        bsig = {r: self._cluster_signature(self.bl, self.Broles, r)
                for r in roles}
        csig = {r: self._cluster_signature(self.ms, self.Croles, r)
                for r in roles}

        def _sh(v):
            return {"plan_aspect": round(v[0], 3),
                    "plan_footprint_mm2": round(v[1], 1),
                    "diamond_radius_mm": round(v[2], 2)}

        out = {"seed_shape": {r: _sh(bsh[r]) for r in roles},
               "candidate_shape": {r: _sh(csh[r]) for r in roles},
               "seed_separation": round(seed_sep, 3),
               "candidate_separation": round(cand_sep, 3),
               "min_separation": CLUSTER_SHAPE_MIN_SEP,
               "engraved_faces_per_body": {
                   "seed": {r: (None if bsig[r] is None else round(bsig[r], 2))
                            for r in roles},
                   "candidate": {r: (None if csig[r] is None
                                     else round(csig[r], 2)) for r in roles},
                   "note": "reported, not used for identity: a candidate may "
                           "engrave the face buttons, and the reference "
                           "does, which inverts this key"}}

        if seed_sep < CLUSTER_SHAPE_MIN_SEP:
            out["verdict"] = "seed_indistinct"
            out["note"] = ("the seed's own button clusters have the same "
                           "plan shape, so identity cannot be measured for "
                           "either part; labels left as found")
            return out

        if cand_sep >= CLUSTER_SHAPE_MIN_SEP:
            # Both assignments are priced against the seed and the cheaper
            # one wins.  No absolute cut-off: a candidate that rebuilds both
            # clusters at a different size still pairs up correctly.
            keep = (self._shape_dist(csh["dpad"], bsh["dpad"])
                    + self._shape_dist(csh["face_buttons"],
                                       bsh["face_buttons"]))
            swap = (self._shape_dist(csh["dpad"], bsh["face_buttons"])
                    + self._shape_dist(csh["face_buttons"], bsh["dpad"]))
            out["match_cost"] = {"as_labelled": round(keep, 3),
                                 "swapped": round(swap, 3)}
            if swap < keep:
                self.Croles["dpad"], self.Croles["face_buttons"] = (
                    self.Croles["face_buttons"], self.Croles["dpad"])
                out["relabelled"] = True
            out["verdict"] = "measured"
            return out

        # Dead heat on the candidate. Which seed role do both diamonds look
        # like? The other one is the one that is missing.
        common = tuple((csh["dpad"][i] + csh["face_buttons"][i]) / 2.0
                       for i in range(3))
        survivor = min(roles, key=lambda r: self._shape_dist(bsh[r], common))
        missing = [r for r in roles if r != survivor][0]

        exp = self._expected_cluster_x(survivor)
        cands = [self.Croles[r] for r in roles]
        if exp is not None:
            cands.sort(key=lambda ids: abs(self._cluster_centre_x(ids) - exp))
        self.Croles[survivor] = cands[0]
        self.Croles[missing] = []

        out["verdict"] = "indistinguishable"
        out["survivor"] = survivor
        out["missing"] = missing
        out["note"] = (
            f"the candidate's two button diamonds have the same plan shape "
            f"(aspect {round(common[0], 3)}, footprint "
            f"{round(common[1], 1)} mm2), which matches the seed's "
            f"{survivor}; the seed separates its clusters by "
            f"{round(seed_sep, 3)}. The {missing} is therefore absent as a "
            f"distinct cluster and is scored as an unmatched role. The "
            f"{survivor} keeps the diamond nearer its expected position, so "
            f"an unmeasurable label cannot also sink a cluster the candidate "
            f"did place correctly.")
        return out

    # -- matching ---------------------------------------------------------
    def _perm_cost(self, bids, perm, expected):
        """Shape-first cost of one baseline->candidate assignment.

        Fingerprints are invariant under both translation and mirroring, so
        they identify WHICH button is which (cross vs circle vs triangle)
        without assuming anything about where it was supposed to move.
        Position only breaks ties between genuinely identical bodies, and it
        is measured against the MIRROR hypothesis alone -- never against the
        widening hypothesis, which is what c1 is supposed to be testing.
        """
        cost = 0.0
        for b, c in zip(bids, perm):
            cost += TOL["match_fp_weight"] * fp_distance(self.B[b], self.C[c])
            resid_mm = d3(expected[b], self.C[c]["centroid_m"]) * MM
            cost += TOL["match_pos_weight_per_mm"] * resid_mm
        return cost

    def _best_assignment(self, bids, cids, expected):
        best, best_cost, second = None, float("inf"), float("inf")
        for perm in permutations(cids):
            cost = self._perm_cost(bids, perm, expected)
            if cost < best_cost:
                second, best_cost, best = best_cost, cost, perm
            elif cost < second:
                second = cost
        margin = (second - best_cost) if second < float("inf") else None
        return best, best_cost, margin

    def _match(self):
        """Per-role matching, baseline -> candidate.

        Two hypotheses are costed for every role -- "the candidate mirrored
        this role" and "the candidate left it where it was" -- and the
        cheaper assignment wins.  Costing only the mirrored hypothesis would
        assume the answer to the questions c1/c2/c4 go on to ask.
        """
        m = {}
        for role, bids in self.Broles.items():
            cids = self.Croles.get(role, [])
            if not cids:
                for bid in bids:
                    m[bid] = {"cand": None, "role": role}
                self.match_diag[role] = {"hypothesis": None,
                                         "note": "role absent on candidate"}
                continue

            if len(bids) == len(cids) and len(bids) <= 6:
                exp_mirror = {bid: [mirror_x(self.B[bid]["centroid_m"][0],
                                             self.P)]
                              + self.B[bid]["centroid_m"][1:]
                              for bid in bids}
                exp_ident = {bid: list(self.B[bid]["centroid_m"])
                             for bid in bids}
                mp, mc, mm_ = self._best_assignment(bids, cids, exp_mirror)
                ip, ic, im = self._best_assignment(bids, cids, exp_ident)
                if mc <= ic:
                    perm, hyp, margin, other = mp, "mirrored", mm_, ic
                else:
                    perm, hyp, margin, other = ip, "identity", im, mc
                self.match_diag[role] = {
                    "hypothesis": hyp,
                    "cost_mirrored": round(mc, 5),
                    "cost_identity": round(ic, 5),
                    "runner_up_margin": (round(margin, 5)
                                         if margin is not None else None),
                }
                for b, c in zip(bids, perm):
                    m[b] = {"cand": c, "role": role, "hypothesis": hyp,
                            "residual_m": d3(self.B[b]["centroid_m"],
                                             self.C[c]["centroid_m"])}
            else:
                bs = sorted(bids, key=lambda i: -self.B[i]["volume_m3"])
                cs = sorted(cids, key=lambda i: -self.C[i]["volume_m3"])
                self.match_diag[role] = {
                    "hypothesis": "volume_rank",
                    "note": f"body count changed {len(bids)} -> {len(cids)}",
                }
                for i, b in enumerate(bs):
                    m[b] = {"cand": cs[i] if i < len(cs) else None,
                            "role": role, "via": "volume_rank"}
        return m

    # -- shared measurement ----------------------------------------------
    def _pair_rows(self):
        """Mirror-pair separation growth, in millimetres, per role."""
        if self._pair_rows_cache is not None:
            return self._pair_rows_cache
        rows = []
        for role in WIDTH_PAIR_ROLES:
            bids = self.Broles.get(role, [])
            if len(bids) != 2:
                continue
            cs = [self.match[b]["cand"] for b in bids]
            if None in cs:
                continue
            seed = abs(self.B[bids[0]]["centroid_m"][0]
                       - self.B[bids[1]]["centroid_m"][0]) * MM
            got = abs(self.C[cs[0]]["centroid_m"][0]
                      - self.C[cs[1]]["centroid_m"][0]) * MM
            rows.append({"role": role, "seed_mm": round(seed, 3),
                         "got_mm": round(got, 3),
                         "delta_mm": round(got - seed, 3)})
        self._pair_rows_cache = rows
        return rows

    def _actual_half_m(self):
        """Half of the widening the candidate ACTUALLY applied, in metres.

        c2 uses this instead of the nominal 7.5 mm so that "is the widening
        the right size?" (c1) and "are the clusters consistent with the
        shell the candidate built?" (c2) stay independent questions.  An
        unwidened shell with correctly swapped clusters then loses c1 only,
        instead of being charged twice for a single mistake.
        """
        rows = self._pair_rows()
        if not rows:
            return POLICY["half_mm"] / MM
        mean_delta_mm = sum(r["delta_mm"] for r in rows) / len(rows)
        return (mean_delta_mm / 2.0) / MM

    # -- c0: rebuild health ---------------------------------------------
    def c0_health(self):
        """How badly the feature tree is broken, relative to the seed.

        Graded on the FRACTION of features newly broken rather than the raw
        count, so the criterion transfers to tasks whose trees are a
        different size.
        """
        rb = self.ms.get("rebuild", {})
        n_feat = rb.get("features") or 0
        broken = rb.get("errors")
        if broken is None:
            return {"score": 1.0, "status": UNVERIFIABLE,
                    "evidence": "no rebuild census available"}
        frac = (broken / n_feat) if n_feat else (1.0 if broken else 0.0)
        score = score_error(frac, TOL["health_perfect_frac"],
                            TOL["health_zero_frac"])
        return {"score": round(score, 4), "status": status_of(score),
                "newly_broken": broken,
                "features": n_feat,
                "broken_fraction": round(frac, 5),
                "broken_features": rb.get("broken_features", [])[:10],
                "evidence": "fraction of features newly broken versus the "
                            "seed census. Pre-existing seed errors are not "
                            "charged to the candidate."}

    # -- c0: modelling hygiene -------------------------------------------
    def c0_hygiene(self):
        """Model quality that geometry does not expose.

        New rebuild warnings, sketches falling out of their seed constraint
        state, features suppressed rather than removed, and new external
        references. Everything is a DELTA against the seed: the seed may
        itself contain under-defined sketches and the candidate must not be
        charged for them.
        """
        bm = self.bl.get("modelling")
        cm = self.ms.get("modelling")
        parts, detail = {}, {}

        # New rebuild warnings are available even without a modelling census.
        rb = self.ms.get("rebuild", {})
        new_warn = len(rb.get("new_warnings", []) or [])
        parts["new_warnings"] = score_error(
            new_warn, TOL["warn_perfect"], TOL["warn_zero"])
        detail["new_warnings"] = {"count": new_warn,
                                  "score": round(parts["new_warnings"], 4)}

        if not bm or not cm:
            # No seed census to compare against: this is a gap in the
            # baseline, not a fault of the candidate, so it must not cost
            # marks. It is reported loudly instead.
            score = parts["new_warnings"]
            return {"score": round(score, 4), "status": status_of(score),
                    "components": {k: round(v, 4) for k, v in parts.items()},
                    "detail": detail,
                    "evidence": "only rebuild warnings could be graded -- the "
                                "baseline carries no modelling census. "
                                "Re-freeze it with --capture-baseline to "
                                "enable sketch, suppression and external "
                                "reference checks."}

        # -- sketches: statuses the seed never had -------------------------
        # Counting sketches outside the seed's *dominant* status looks
        # reasonable and is not: the seed legitimately mixes statuses, and any
        # real edit adds sketches across the same mix. Measured on the corpus,
        # that metric moved identically for the reference and for almost every
        # adversarial -- no signal, and it docked the reference.
        #
        # What does carry signal is a status the seed never exhibited at all.
        # The seed defines which states are normal FOR THIS MODEL; a new one
        # appearing means a sketch entered a state the author never had, which
        # is degradation regardless of what the numeric code means.
        bsk = (bm.get("sketches") or {}).get("status_counts") or {}
        csk = (cm.get("sketches") or {}).get("status_counts") or {}
        if bsk and csk:
            known = set(bsk)
            novel = {k: v for k, v in csk.items()
                     if k not in known and k != "unreadable"}
            n_novel = sum(novel.values())
            parts["sketch_definition"] = score_error(
                n_novel, TOL["sketch_perfect"], TOL["sketch_zero"])
            detail["sketch_definition"] = {
                "seed_statuses": sorted(known),
                "candidate_statuses": sorted(csk),
                "novel_statuses": novel,
                "sketches_in_novel_statuses": n_novel,
                "score": round(parts["sketch_definition"], 4),
                "note": "sketches in a constraint status the seed never had"}

        # -- suppressed features ------------------------------------------
        b_sup = (bm.get("suppressed") or {}).get("count")
        c_sup = (cm.get("suppressed") or {}).get("count")
        if b_sup is not None and c_sup is not None:
            growth = max(0, c_sup - b_sup)
            parts["suppressed_features"] = score_error(
                growth, TOL["suppressed_perfect"], TOL["suppressed_zero"])
            detail["suppressed_features"] = {
                "seed": b_sup, "candidate": c_sup, "growth": growth,
                "names": (cm.get("suppressed") or {}).get("names", [])[:10],
                "score": round(parts["suppressed_features"], 4)}

        # -- external references ------------------------------------------
        b_ref = (bm.get("external_refs") or {}).get("count")
        c_ref = (cm.get("external_refs") or {}).get("count")
        if b_ref is not None and c_ref is not None:
            growth = max(0, c_ref - b_ref)
            parts["external_refs"] = score_error(
                growth, TOL["extref_perfect"], TOL["extref_zero"])
            detail["external_refs"] = {
                "seed": b_ref, "candidate": c_ref, "growth": growth,
                "names": (cm.get("external_refs") or {}).get("names", [])[:10],
                "score": round(parts["external_refs"], 4)}

        unavailable = [k for k, v in (cm.get("available") or {}).items()
                       if not v]
        score = mean_scores(list(parts.values()))
        out = {"score": round(score, 4), "status": status_of(score),
               "components": {k: round(v, 4) for k, v in parts.items()},
               "detail": detail,
               "evidence": "model quality as a delta against the seed: new "
                           "rebuild warnings, sketches leaving their seed "
                           "constraint state, features suppressed rather "
                           "than removed, new external references."}
        if unavailable:
            out["unavailable_probes"] = unavailable
            out["notes"] = (cm.get("notes") or [])[:5]
        return out

    # -- c1 ------------------------------------------------------------
    def c1_width(self):
        target = POLICY["width_delta_mm"]
        rows = self._pair_rows()
        for r in rows:
            r["score"] = round(asymmetric_target_score(
                r["delta_mm"], target,
                TOL["width_perfect_mm"],
                TOL["width_zero_under_mm"],
                TOL["width_zero_over_mm"]), 4)
        # Mean of per-pair SCORES, not a score of the mean delta: averaging
        # deltas would let +30 mm on one pair cancel 0 mm on another and
        # hand full marks to a visibly deformed part.
        score = mean_scores([r["score"] for r in rows], default=0.0)
        return {"score": round(score, 4), "status": status_of(score),
                "target_delta_mm": target, "pairs": rows,
                "evidence": "growth of mirror-pair separation "
                            "(sticks/triggers/bumpers); robust to a housing "
                            "remodel, global bbox not used. Asymmetric: "
                            "undershoot reaches 0 at delta=0, overshoot "
                            "decays more slowly."}

    # -- c2 ------------------------------------------------------------
    def c2_spacing(self):
        half = self._actual_half_m()
        det = {"half_applied_mm": round(half * MM, 3)}

        # 2.1 -- diamonds must translate as rigid units
        intra_by_role, intra_det = {}, {}
        for role in RIGID_ROLES:
            bids = self.Broles.get(role, [])
            if not bids:
                continue
            if self._role_unmatched(role):
                intra_by_role[role] = 0.0
                intra_det[role] = {"score": 0.0, "status": FAIL,
                                   "note": "cluster missing or unrecognisable "
                                           "on the candidate",
                                   **self.unmatched_roles[role]}
                continue
            worst = 0.0
            seen = False
            for i in range(len(bids)):
                for j in range(i + 1, len(bids)):
                    a, b = bids[i], bids[j]
                    ca, cb = self.match[a]["cand"], self.match[b]["cand"]
                    if not ca or not cb:
                        continue
                    seen = True
                    seed = d3(self.B[a]["centroid_m"],
                              self.B[b]["centroid_m"]) * MM
                    got = d3(self.C[ca]["centroid_m"],
                             self.C[cb]["centroid_m"]) * MM
                    worst = max(worst, abs(got - seed))
            if not seen:
                continue
            s = score_error(worst, TOL["intra_perfect_mm"],
                            TOL["intra_zero_mm"])
            intra_by_role[role] = s
            intra_det[role] = {"max_spacing_drift_mm": round(worst, 3),
                               "score": round(s, 4)}
        intra = weighted_role_mean(intra_by_role)
        det["intra_cluster"] = intra_det

        # 2.2 -- cluster centres, against the candidate's OWN widening
        pos_by_role, pos_det = {}, {}
        for role in POSITION_ROLES:
            bids = self.Broles.get(role, [])
            if not bids:
                continue
            if self._role_unmatched(role):
                pos_by_role[role] = 0.0
                pos_det[role] = {"score": 0.0, "status": FAIL,
                                 "note": "cluster missing or unrecognisable "
                                         "on the candidate -- the requested "
                                         "move is not demonstrated",
                                 **self.unmatched_roles[role]}
                continue
            cs = [self.match[i]["cand"] for i in bids]
            exp = [
                sum(mirror_widen_x(self.B[i]["centroid_m"][0], self.P, half)
                    for i in bids) / len(bids),
                sum(self.B[i]["centroid_m"][1] for i in bids) / len(bids),
                sum(self.B[i]["centroid_m"][2] for i in bids) / len(bids),
            ]
            got = [sum(self.C[c]["centroid_m"][k] for c in cs) / len(cs)
                   for k in range(3)]
            resid = d3(exp, got) * MM
            s = score_error(resid, TOL["cluster_perfect_mm"],
                            TOL["cluster_zero_mm"])
            pos_by_role[role] = s
            pos_det[role] = {"expected_x_mm": round(exp[0] * MM, 2),
                             "got_x_mm": round(got[0] * MM, 2),
                             "residual_mm": round(resid, 2),
                             "score": round(s, 4)}
        # Weighted by ROLE_WEIGHT: skipping a whole button cluster must cost
        # more than nudging a bumper, and the stick/trigger/bumper pairs are
        # already what c1 grades.
        position = weighted_role_mean(pos_by_role)
        det["cluster_position"] = pos_det
        det["role_weights"] = ROLE_WEIGHT

        rigid_mult = RIGID_FLOOR + (1.0 - RIGID_FLOOR) * clamp01(intra)
        score = clamp01(position) * rigid_mult
        return {"score": round(score, 4), "status": status_of(score),
                "model": "cluster_placement x rigidity",
                "components": {"cluster_position": round(position, 4),
                               "intra_cluster_rigidity": round(intra, 4),
                               "rigidity_multiplier": round(rigid_mult, 4)},
                "detail": det,
                "evidence": "button diamonds must land where the mirror "
                            "plus the candidate's own widening puts them, "
                            "and travel as rigid units. Graded against the "
                            "applied widening rather than the nominal "
                            "7.5 mm, which keeps it independent of c1. "
                            "Symmetric pairs are excluded: invariant as a "
                            "set, their mean centre proves nothing."}

    # -- c3 ------------------------------------------------------------
    def c3_interference(self):
        seed_v = self.bl["interference"]["total_volume_m3"]
        got_v = self.ms["interference"]["total_volume_m3"]
        growth = max(0.0, got_v - seed_v)
        score = score_error(growth, TOL["intf_perfect_m3"],
                            TOL["intf_zero_m3"])
        return {"score": round(score, 4), "status": status_of(score),
                "seed_total_m3": seed_v, "measured_total_m3": got_v,
                "growth_m3": growth, "growth_mm3": round(growth * 1e9, 4),
                "free_growth_mm3": TOL["intf_perfect_m3"] * 1e9,
                "zero_at_mm3": TOL["intf_zero_m3"] * 1e9,
                "measured_pairs": len(self.ms["interference"]["pairs"]),
                "boolean_failures":
                    self.ms["interference"].get("boolean_failures"),
                "evidence": "boolean-intersect volume among control bodies "
                            "and control-vs-housing.  Hardware the candidate "
                            "adds (screws in bosses) is out of scope -- the "
                            "reference seats screws by design"}

    # -- c4 ------------------------------------------------------------
    def c4_handedness(self):
        checks = {}

        # -- witness 1: cluster sides (a weighted FRACTION, not one bit) ---
        sides, side_by_role = {}, {}
        for role in MIRROR_ROLES:
            bids = self.Broles.get(role, [])
            if not bids:
                continue
            if self._role_unmatched(role):
                sides[role] = {"status": FAIL,
                               "note": "cluster missing or unrecognisable on "
                                       "the candidate",
                               **self.unmatched_roles[role]}
                side_by_role[role] = 0.0
                continue
            cs = [self.match[i]["cand"] for i in bids]
            seed_c = sum(self.B[i]["centroid_m"][0] for i in bids) / len(bids)
            got_c = sum(self.C[c]["centroid_m"][0] for c in cs) / len(cs)
            straddle = abs(seed_c - self.P) * MM < TOL["straddle_mm"]
            if straddle:
                # A cluster centred on the plane -- every mirror-symmetric
                # pair, and the centre buttons -- has no side to change.
                # Scoring these as passed would award three of six roles to
                # any candidate unconditionally.  Sidedness is genuinely
                # unverifiable here; body chirality witnesses them instead.
                sides[role] = {"seed_x_mm": round(seed_c * MM, 1),
                               "got_x_mm": round(got_c * MM, 1),
                               "status": UNVERIFIABLE, "on_plane": True,
                               "note": "cluster straddles the plane; it has "
                                       "no side to change"}
                continue
            flipped = sgn(seed_c - self.P) * sgn(got_c - self.P) < 0
            sides[role] = {"seed_x_mm": round(seed_c * MM, 1),
                           "got_x_mm": round(got_c * MM, 1),
                           "status": PASS if flipped else FAIL,
                           "on_plane": False}
            side_by_role[role] = 1.0 if flipped else 0.0
        e_sides = (weighted_role_mean(side_by_role, default=None)
                   if side_by_role else None)
        checks["cluster_sides"] = {
            "score": None if e_sides is None else round(e_sides, 4),
            "status": UNVERIFIABLE if e_sides is None else status_of(e_sides),
            "roles": sides,
            "note": "ROLE-WEIGHTED fraction of off-plane clusters that "
                    "changed side, so moving one button cluster across and "
                    "forgetting the other scores about half, not zero. "
                    "Clusters straddling the plane are UNVERIFIABLE, not "
                    "free passes.",
        }

        # -- witness 2: body chirality -------------------------------------
        chir, chir_flags = {}, []
        for role in CHIRAL_ROLES:
            for bid in self.Broles.get(role, []):
                m = self.match[bid]
                if not m["cand"]:
                    continue
                sb = self.B[bid]
                ch = chirality(sb)
                if ch < TOL["chiral_floor"]:
                    chir[bid] = {"role": role, "status": UNVERIFIABLE,
                                 "chirality": ch,
                                 "note": "near mirror-symmetric; a mirror "
                                         "cannot be read from its inertia"}
                    continue
                key = chirality_witness(sb)
                s = sb["inertia_com"][key]
                c = self.C[m["cand"]]["inertia_com"][key]
                flipped = (s * c) < 0
                chir_flags.append(1.0 if flipped else 0.0)
                chir[bid] = {"role": role,
                             "status": PASS if flipped else FAIL,
                             "witness": key, "seed": s, "measured": c}
        e_chir = (sum(chir_flags) / len(chir_flags)) if chir_flags else None
        checks["body_chirality"] = {
            "score": None if e_chir is None else round(e_chir, 4),
            "status": UNVERIFIABLE if e_chir is None else status_of(e_chir),
            "bodies": chir,
            "verifiable_bodies": len(chir_flags),
        }

        # -- witness 3: port-light glyphs (the naive-flip detector) --------
        seed_l = self.bl.get("light_cluster")
        cand_l = self.ms.get("light_cluster")
        if not seed_l:
            e_lights = None
            entry = {"score": None, "status": UNVERIFIABLE,
                     "note": "no light cluster recorded in the baseline"}
        elif not cand_l:
            e_lights = None
            entry = {"score": None, "status": UNVERIFIABLE,
                     "note": "port-light glyph faces not found on the "
                             "candidate (housing re-glyphed?) -- side "
                             "cannot be witnessed"}
        else:
            same = seed_l["side"] == cand_l["side"]
            e_lights = 1.0 if same else 0.0
            entry = {"score": e_lights,
                     "status": PASS if same else FAIL,
                     "seed_side": seed_l["side"],
                     "candidate_side": cand_l["side"],
                     "candidate_x_mm": round(cand_l["centroid_m"][0] * MM, 1),
                     "matched_faces": cand_l["n_faces"],
                     "note": "port-light glyphs are wireless-port "
                             "indicators, not a handed control -- they must "
                             "stay on their original side of the plane. "
                             "This is the ONLY witness that separates a "
                             "proper left-hand conversion from a naive "
                             "whole-part mirror, hence its weight."}
        checks["port_lights_side"] = entry

        # -- witness 4: housing one-sided moment ---------------------------
        bs = self.bl.get("housing_signature", {})
        cs_sig = self.ms.get("housing_signature", {})
        bm, cm = bs.get("moment3_m5", 0.0), cs_sig.get("moment3_m5", 0.0)
        b_house = self.Broles.get("housing", [])
        c_house = self.Croles.get("housing", [])
        remodeled = True
        if len(b_house) == len(c_house) == 1:
            remodeled = fp_distance(self.B[b_house[0]],
                                    self.C[c_house[0]]) > \
                TOL["housing_remodel_fp"]
        entry2 = {"baseline_moment3": bm, "measured_moment3": cm,
                  "housing_remodeled": remodeled}
        if remodeled:
            e_sig = None
            entry2["score"] = None
            entry2["status"] = UNVERIFIABLE
            entry2["note"] = ("housing was re-shelled/remodeled -- interior "
                              "faces dominate the moment, sign is not a "
                              "mirror witness (port_lights_side still is)")
        elif abs(bm) <= 0 or abs(cm) < TOL["sig_floor_frac"] * abs(bm):
            e_sig = None
            entry2["score"] = None
            entry2["status"] = UNVERIFIABLE
            entry2["note"] = "housing asymmetry signal too weak"
        else:
            same_side = (bm * cm) > 0
            e_sig = 1.0 if same_side else 0.0
            entry2["score"] = e_sig
            entry2["status"] = PASS if same_side else FAIL
        checks["housing_side_signature"] = entry2

        # -- witness 5: housing engraved-detail side -----------------------
        # The reason this exists: witness 4 above is UNVERIFIABLE on every
        # candidate that actually re-shelled, which is every candidate that
        # actually did the work, leaving the guard resting on witness 3
        # alone. One witness means one thing to destroy, and destroying it
        # sent the whole class to NEUTRAL_UNVERIFIABLE -- worth MORE than
        # being caught. This witness reads different faces and fails for
        # different reasons, so the two do not die together.
        bd = self.bl.get("housing_detail")
        cd = self.ms.get("housing_detail")
        floor = TOL["detail_asym_floor"]
        entry3 = {}
        if not bd or not cd:
            e_detail = None
            entry3 = {"score": None, "status": UNVERIFIABLE,
                      "note": "no engraved-detail measurement in the "
                              "baseline or the capture -- re-freeze the "
                              "baseline and re-capture to enable this "
                              "witness (schema ps3-capture/4)"}
        else:
            ba, ca = bd.get("asymmetry", 0.0), cd.get("asymmetry", 0.0)
            entry3 = {"seed_asymmetry": round(ba, 4),
                      "candidate_asymmetry": round(ca, 4),
                      "floor": floor,
                      "seed_faces": [bd.get("faces_left"),
                                     bd.get("faces_right")],
                      "candidate_faces": [cd.get("faces_left"),
                                          cd.get("faces_right")]}
            if abs(ba) < floor:
                e_detail = None
                entry3["score"] = None
                entry3["status"] = UNVERIFIABLE
                entry3["note"] = ("the seed's engraved detail is balanced "
                                  "about the plane, so it has no side to "
                                  "keep; nothing to witness on this part")
            elif abs(ca) < floor:
                e_detail = None
                entry3["score"] = None
                entry3["status"] = UNVERIFIABLE
                entry3["note"] = ("the candidate's engraved detail no longer "
                                  "favours either side -- markings removed "
                                  "or flattened, so the side cannot be read")
            else:
                # DETAIL_SIDE_IS_DIAGNOSTIC. The old rule was
                # `same = (ba * ca) > 0` scored 1.0 -- the engraved detail
                # must stay on its side -- and it is backwards here:
                # instruction.md asks for the START/SELECT labels to end up
                # in mirrored positions, and those labels ARE the
                # engraving-scale housing faces this measures. It passed one
                # model in the corpus, the one that never converted the
                # controller, and failed every model that did.
                #
                # Kept, with its polarity corrected to what it actually
                # shows, and with no weight: `moved` is what the task asks
                # for. Not promoted to positive evidence either, because as
                # a signal it says what cluster_sides already says and those
                # weights are calibrated.
                moved = (ba * ca) < 0
                e_detail = None
                entry3["score"] = None
                entry3["moved_to_other_side"] = moved
                entry3["status"] = UNVERIFIABLE
                entry3["note"] = ("DIAGNOSTIC ONLY, CARRIES NO WEIGHT. "
                                  "Area-weighted left/right balance of "
                                  "engraving-scale housing faces -- the "
                                  "START/SELECT labels among them. A sign "
                                  "flip means they crossed the plane, which "
                                  "instruction.md requires, so this cannot "
                                  "guard against a naive mirror: both move "
                                  "them. Measured, the naive-flip adversary "
                                  "and the reference read +0.19 and +0.20. "
                                  "port_lights_side is what separates those "
                                  "two.")
        checks["housing_detail_side"] = entry3

        checks["symbol_glyphs"] = {
            "score": None,
            "status": UNVERIFIABLE,
            "note": "PS triangle/circle/cross/square are left-right "
                    "symmetric split-faces; a mirror is geometrically "
                    "undetectable on the symbols themselves.  Handedness is "
                    "graded from cluster sides, body inertia signs and the "
                    "housing side signature instead.  Carries no weight -- "
                    "it is UNVERIFIABLE by construction, not by accident.",
        }

        # -- combine ------------------------------------------------------
        # Positive evidence earns the credit; the guard can only reduce it.
        positive, pos_diag = weighted_evidence({
            "cluster_sides": (HANDEDNESS_POSITIVE["cluster_sides"], e_sides),
            "body_chirality": (HANDEDNESS_POSITIVE["body_chirality"], e_chir),
        })
        # e_detail is deliberately absent: see DETAIL_SIDE_IS_DIAGNOSTIC and
        # the note on HANDEDNESS_GUARD. It is measured, reported, and worth
        # nothing here, because on this task it fires on the correct answer.
        guard, guard_diag = weighted_evidence({
            "port_lights_side": (HANDEDNESS_GUARD["port_lights_side"],
                                 e_lights),
            "housing_side_signature": (
                HANDEDNESS_GUARD["housing_side_signature"], e_sig),
        })
        multiplier = GUARD_FLOOR + (1.0 - GUARD_FLOOR) * clamp01(guard)
        score = clamp01(positive) * multiplier

        low_conf = (pos_diag["verified_weight_fraction"] < 1.0
                    or guard_diag["verified_weight_fraction"] < 1.0)
        return {"score": round(score, 4), "status": status_of(score),
                "model": "positive_evidence x naive_flip_guard",
                "positive_evidence": {
                    "score": round(clamp01(positive), 4),
                    "weights": HANDEDNESS_POSITIVE, **pos_diag},
                "naive_flip_guard": {
                    "score": round(clamp01(guard), 4),
                    "multiplier": round(multiplier, 4),
                    "floor": GUARD_FLOOR,
                    "weights": HANDEDNESS_GUARD, **guard_diag},
                "low_confidence": low_conf,
                "checks": checks,
                "match_hypotheses": self.match_diag}

    # -- c7 ------------------------------------------------------------
    def c7_markings(self):
        """Did the face-button symbols survive the edit?

        instruction.md asks for text and logos to REMAIN legible. Nothing
        read the "remain": `adversarial_missing_glyphs` is the reference
        with the four symbols removed and nothing else touched, and it
        scored full marks.

        Counted on the engraving-scale faces of the round-button cluster,
        after `_resolve_button_clusters()` has settled which cluster that
        is. The seed's four buttons carry 0, 11, 15 and 24 such faces --
        the symbols differ, so their face counts differ -- and every model
        in this corpus reproduces that multiset exactly or carries none at
        all. 50 against 0; the fraction retained is what is scored, with
        full marks from MARKINGS_FULL_FRACTION so a candidate that re-cuts
        the symbols differently is not punished for the difference.

        THE D-PAD ARROWS ARE NOT COUNTED. The seed's cross carries 44
        engraved faces, 11 a limb, and every model that performed the
        conversion carries 20, 5 a limb: the reference simplifies the
        arrows when it rebuilds the cross and the adversaries inherit
        that. There is no signal in it, only a way to take marks off every
        model including the reference. Reported below, not scored.
        """
        smf_b = self.bl.get("small_face_counts") or {}
        smf_c = self.ms.get("small_face_counts") or {}
        bids = self.Broles.get("face_buttons") or []
        cids = self.Croles.get("face_buttons") or []

        def census(ids, smf):
            if not ids or any(i not in smf for i in ids):
                return None
            return [smf[i] for i in ids]

        b_faces = census(bids, smf_b)
        c_faces = census(cids, smf_c)

        # The d-pad, for the report only.
        arrows = {"seed": census(self.Broles.get("dpad") or [], smf_b),
                  "candidate": census(self.Croles.get("dpad") or [], smf_c),
                  "note": "NOT SCORED -- the reference rebuilds the cross "
                          "and simplifies its arrows (44 faces in the seed, "
                          "20 in every converted model), so this separates "
                          "nothing and would charge the reference for it."}

        if b_faces is None or not sum(b_faces):
            return {"score": NEUTRAL_UNVERIFIABLE, "status": UNVERIFIABLE,
                    "detail": {"seed_engraved_faces": b_faces,
                               "dpad_arrows": arrows},
                    "evidence": "the seed's face buttons carry no engraving, "
                                "so there is nothing to preserve and this "
                                "cannot be measured either way",
                    "caveat": "scored NEUTRAL_UNVERIFIABLE, not full marks: "
                              "an unreadable witness must not pay better "
                              "than a readable one that half passes."}

        b_total = sum(b_faces)
        if c_faces is None:
            return {"score": 0.0, "status": FAIL,
                    "detail": {"seed_engraved_faces": b_faces,
                               "candidate_engraved_faces": None,
                               "dpad_arrows": arrows},
                    "evidence": f"the seed's face buttons carry {b_total} "
                                f"engraving-scale faces; the candidate has "
                                f"no face-button cluster to read them on"}

        c_total = sum(c_faces)
        retained = c_total / float(b_total)
        score = clamp01(retained / MARKINGS_FULL_FRACTION)
        return {"score": round(score, 4), "status": status_of(score),
                # NUMBERS ONLY IN `components`: summarise() prints each one
                # through a float format, so a list here raises TypeError at
                # the end of an otherwise complete grade.  The per-body
                # censuses live in `detail`.
                "components": {"retained_fraction": round(retained, 4),
                               "full_marks_from": MARKINGS_FULL_FRACTION},
                "detail": {"seed_engraved_faces": sorted(b_faces),
                           "candidate_engraved_faces": sorted(c_faces),
                           "seed_total": b_total,
                           "candidate_total": c_total,
                           "dpad_arrows": arrows},
                "evidence": f"face-button engraving: {c_total} of the seed's "
                            f"{b_total} engraving-scale faces retained "
                            f"({retained:.0%})",
                "caveat": "counts engraving-scale faces on the round-button "
                          "cluster, which is where the four PS symbols are "
                          "cut.  It reads PRESENCE, not orientation -- the "
                          "symbols are left-right symmetric, so a mirrored "
                          "symbol is geometrically identical to an upright "
                          "one and is the naive-flip guard's business, not "
                          "this criterion's."}

    # -- c5 ------------------------------------------------------------
    def c5_unrequested(self):
        det = {}

        # 5.1 -- Y/Z spans
        sb = self.bl["global"]["bbox_m"]
        mb = self.ms["global"]["bbox_m"]
        worst_span, span_det = 0.0, {}
        for ax, nm in ((1, "Y"), (2, "Z")):
            want = (sb[ax + 3] - sb[ax]) * MM
            got = (mb[ax + 3] - mb[ax]) * MM
            drift = abs(got - want)
            worst_span = max(worst_span, drift)
            span_det[nm] = {"seed_mm": round(want, 1), "got_mm": round(got, 1),
                            "drift_mm": round(drift, 2)}
        span = score_error(worst_span, TOL["span_perfect_mm"],
                           TOL["span_zero_mm"])
        det["spans"] = span_det
        det["worst_span_drift_mm"] = round(worst_span, 2)
        # X is not scored (growing it is the point of the task) but is
        # recorded: a shell widened without its controls moving is invisible
        # to the mirror-pair proxy in c1, and this makes that case
        # diagnosable.
        det["x_span_mm"] = {
            "seed": round((sb[3] - sb[0]) * MM, 2),
            "got": round((mb[3] - mb[0]) * MM, 2),
            "growth": round(((mb[3] - mb[0]) - (sb[3] - sb[0])) * MM, 2),
            "note": "diagnostic only -- not scored",
        }

        # 5.2 -- shapes of the bodies the candidate was not asked to touch
        exempt = set()
        for role in RESHAPE_EXEMPT:
            exempt.update(self.Broles.get(role, []))
        worst_fp, reshaped = 0.0, {}
        for bid, m in self.match.items():
            if bid in exempt or not m["cand"]:
                continue
            d = fp_distance(self.B[bid], self.C[m["cand"]])
            worst_fp = max(worst_fp, d)
            if d > TOL["fp_perfect"]:
                reshaped[bid] = {"role": m["role"], "distance": round(d, 5)}
        shape = score_error(worst_fp, TOL["fp_perfect"], TOL["fp_zero"])
        det["worst_fingerprint_drift"] = round(worst_fp, 5)
        det["reshaped"] = reshaped

        # 5.3 -- finished housing geometry.
        # The housing cannot be compared with the seed because the requested
        # widening legitimately remodels it. Compare it with the finished
        # reference instead. Selection is by geometry (largest volumes), not
        # feature names or body IDs.
        ref_bodies = sorted(
            self.ref["bodies"], key=lambda b: -b["volume_m3"]
        )[:2]
        cand_bodies = sorted(
            self.ms["bodies"], key=lambda b: -b["volume_m3"]
        )[:2]
        housing_dists = [
            fp_distance(rb, cb)
            for rb, cb in zip(ref_bodies, cand_bodies)
        ]
        worst_housing_fp = max(housing_dists) if housing_dists else 0.0

        bbox_drift_mm = 0.0
        for rb, cb in zip(ref_bodies, cand_bodies):
            for rv, cv in zip(rb["bbox_m"], cb["bbox_m"]):
                bbox_drift_mm = max(
                    bbox_drift_mm, abs(rv - cv) * MM
                )

        same_housing_envelope = (
            bbox_drift_mm <= TOL["housing_ref_bbox_mm"]
        )

        housing_shape = score_error(
            worst_housing_fp,
            TOL["housing_ref_fp_perfect"],
            TOL["housing_ref_fp_zero"],
        )

        det["finished_housing"] = {
            "fingerprint_distances":
                [round(d, 7) for d in housing_dists],
            "worst_fingerprint_drift": round(worst_housing_fp, 7),
            "score": round(housing_shape, 4),
            "evidence":
                "finished geometry: volume, surface area and inertia",
        }

        # 5.3 -- REPORTED, NOT CHARGED: what the exemption hides.
        #
        # The exemption above is load-bearing and correct -- the reference
        # really does remodel the housing and the diamonds, so their
        # shape cannot be compared against the seed's. It is also a door,
        # and `adversarial_unrequested_change_elsewhere` walks through it:
        # it is the reference plus one small cut on the housing, scores
        # 7.000 of 7.0, and every criterion is right to give it full
        # marks, because every criterion is measuring something else.
        #
        # NO THRESHOLD IS SET, and that is a measured decision rather than
        # timidity. Across this corpus the housing runs 185 676 to
        # 1 515 892 mm3, its face count 522 to 810, the tree 58 to 65
        # sketches and 199 to 284 features. The model named for the
        # unrequested change sits at 810 faces / 64 sketches / 280
        # features -- and `adversarial_widened_by_30mm`, whose defect is
        # something else entirely, sits above it at 799 / 65 / 284. A
        # cut-off that catches the first flags the second. These are
        # authored models, not one author's variations, and their trees
        # have no common scale.
        #
        # So the numbers are printed where the score is, and the score is
        # left alone. A 7.000 that also says "three features and nine
        # faces the seed does not have" is not the same artefact as a
        # bare 7.000, and the next person should not have to diff two
        # captures by hand to learn it.
        seed_m = (self.bl.get("modelling") or {})
        cand_m = (self.ms.get("modelling") or {})

        def _n(d, *path):
            for k in path:
                d = (d or {}).get(k)
            return d

        seed_faces = _n(self.bl, "housing_signature", "faces")
        cand_faces = _n(self.ms, "housing_signature", "faces")
        tf = {"note": "diagnostic only -- not scored. The reshape check "
                      "cannot see inside an exempt body, so these are the "
                      "only trace an unrequested edit there leaves. They "
                      "carry no threshold: this corpus's trees differ too "
                      "much between authors for one to be honest.",
              "housing_faces": {"seed": seed_faces, "got": cand_faces},
              "sketches": {"seed": _n(seed_m, "sketches", "count"),
                           "got": _n(cand_m, "sketches", "count")},
              "features_scanned": {
                  "seed": _n(seed_m, "suppressed", "features_scanned"),
                  "got": _n(cand_m, "suppressed", "features_scanned")}}
        for k in ("housing_faces", "sketches", "features_scanned"):
            a, b = tf[k]["seed"], tf[k]["got"]
            tf[k]["delta"] = (b - a) if isinstance(a, int) \
                and isinstance(b, int) else None
        det["tree_and_faces"] = tf

        # THE WEAKER HALF DECIDES, NOT THE AVERAGE. This was
        # `0.5 * span + 0.5 * shape`, and the average is the wrong shape for
        # a constraint: `yz_spans` is 1.000 for every candidate that did not
        # resize the whole part, so it held the criterion at 0.500 however
        # badly a body was reshaped. Measured by simulation on the
        # reference's own capture, cutting a trigger by 18.6% of its volume
        # moved this criterion from 1.000 to 0.500 and the total from 8.000
        # to 7.750 -- a quarter of a point, the most an unrequested change
        # could ever cost. Either half failing means an unrequested change
        # was made, so the criterion now reports the one that failed.
        controls_are_correct = self.c2_spacing()["score"] >= 0.999

        charge_housing_reference = (
            same_housing_envelope and controls_are_correct
        )

        score = (
            min(span, shape, housing_shape)
            if charge_housing_reference
            else min(span, shape)
        )
        return {"score": round(score, 4), "status": status_of(score),
                "components": {"yz_spans": round(span, 4),
               "body_shapes": round(shape, 4),
               "finished_housing": round(housing_shape, 4)},
                "detail": det,
        "caveat": "housing, button diamonds and centre buttons are "
                      "exempt from the seed reshape check because the "
                      "requested edit legitimately remodels them. "
                      "Finished housing geometry is compared with the "
                      "reference when the housing envelope and control "
                      "layout indicate that comparison is applicable, "
                      "avoiding double penalties for other known defects."}

    # -- assemble --------------------------------------------------------
    def grade(self):
        report = {"document": self.ms.get("document"),
                  "rebuild": self.ms.get("rebuild", {}),
                  "policy": POLICY,
                  "plane_x_m": self.P,
                  "plane_x_measured_m": self.ms.get("plane_x_measured_m"),
                  "criteria": {}, "notes": []}

        names = GEOMETRY_CRITERIA

        reason = ungradable_reason(self.bl, self.ms)
        if reason is None and self.Broles and self.unmatched_roles:
            if set(self.unmatched_roles) >= {r for r, ids in
                                             self.Broles.items() if ids}:
                reason = ("no baseline role could be matched on this "
                          "candidate -- the part does not resemble the seed "
                          "closely enough to compare")
        if reason:
            for k in ALL_CRITERIA:
                report["criteria"][k] = {
                    "score": 0.0, "status": FAIL,
                    "evidence": "not evaluated -- candidate is ungradable"}
            report["ungradable"] = {"reason": reason,
                                    "bodies": len(self.ms.get("bodies", [])),
                                    "plane_x_m": self.ms.get("plane_x_m")}
            report["overall_score"] = 0.0
            report["overall"] = FAIL
            report["notes"].append(f"UNGRADABLE: {reason}")
            return report

        if self.plane_shift_m:
            report["plane_normalised_by_mm"] = round(self.plane_shift_m * MM,
                                                     2)
            report["notes"].append(
                f"candidate translated {self.plane_shift_m * MM:+.1f} mm "
                "along X; normalised to the baseline plane before comparing, "
                "so the offset itself is not penalised")

        # Health and hygiene are scored for EVERY candidate, including one
        # whose tree does not rebuild.  Only the geometry criteria are gated:
        # measurements taken from a broken rebuild are not trustworthy, but
        # "how badly is it broken" is itself measurable and worth a score --
        # otherwise every broken model collapses onto the same zero.
        report["criteria"][C_HEALTH] = self.c0_health()
        report["criteria"][C_HYGIENE] = self.c0_hygiene()

        rb = self.ms.get("rebuild", {})
        if not rb.get("ok", False):
            for k in names:
                report["criteria"][k] = {
                    "score": 0.0, "status": FAIL,
                    "evidence": "not evaluated -- the feature tree does not "
                                "rebuild clean, so geometry measurements are "
                                "not trustworthy"}
            report["overall_score"] = round(
                sum(v["score"] for v in report["criteria"].values())
                / len(report["criteria"]), 4)
            report["overall"] = FAIL
            report["notes"].append(
                "Rebuild gate failed: the feature tree does not rebuild clean "
                f"vs the seed census (errors={rb.get('errors')}). Geometry "
                "criteria are zeroed; health and hygiene are still scored.")
            return report

        ci = self.cluster_identity
        if ci and ci.get("verdict") != "measured":
            report["cluster_identity"] = ci
            report["notes"].append(f"button-cluster identity: "
                                   f"{ci['verdict']} -- {ci.get('note', '')}")
        elif ci and ci.get("relabelled"):
            report["cluster_identity"] = ci

        if self.unmatched_roles:
            report["unmatched_roles"] = self.unmatched_roles
            report["notes"].append(
                "roles present in the baseline but not matchable on the "
                f"candidate: {sorted(self.unmatched_roles)} -- scored as "
                "failures, not skipped")
        report["criteria"][names[0]] = self.c1_width()
        report["criteria"][names[1]] = self.c2_spacing()
        report["criteria"][names[2]] = self.c3_interference()
        report["criteria"][names[3]] = self.c4_handedness()
        report["criteria"][names[4]] = self.c5_unrequested()
        report["criteria"][C_MARKINGS] = self.c7_markings()

        # A plane the harness had to borrow from the baseline means the
        # part's own symmetry could not be read; sidedness rests on an
        # assumption rather than a measurement.
        src = self.ms.get("plane_source")
        if src and src.startswith("baseline"):
            report["notes"].append(
                f"symmetry plane taken from the baseline ({src}); the part's "
                "own plane could not be measured, so side judgements rest on "
                "the seed's geometry")

        st = [v["score"] for v in report["criteria"].values()]
        report["overall_score"] = round(sum(st) / len(st), 4)
        report["overall"] = (PASS if all(s >= 1.0 for s in st)
                             else (FAIL if all(s <= 0.0 for s in st)
                                   else PARTIAL))
        return report


def summarise(report, weights=None):
    weights = weights or {}
    out = [f"document : {report.get('document')}",
           f"rebuild  : ok={report.get('rebuild', {}).get('ok')} "
           f"errors={report.get('rebuild', {}).get('errors')}", ""]
    for k, v in report["criteria"].items():
        w = weights.get(k, 1.0)
        out.append(f"  {k:32} {v['score']:.3f}  {v['status']:<8} (w={w:g})")
        for cname, cval in (v.get("components") or {}).items():
            out.append(f"        . {cname:28} {cval:.3f}")
        if "positive_evidence" in v:
            out.append(f"        . {'positive evidence':28} "
                       f"{v['positive_evidence']['score']:.3f}")
            out.append(f"        . {'naive-flip guard (x)':28} "
                       f"{v['naive_flip_guard']['multiplier']:.3f}")
        for name, entry in (v.get("checks") or {}).items():
            if isinstance(entry, dict) and entry.get("status") == UNVERIFIABLE:
                out.append(f"        ! {name}: UNVERIFIABLE")
        if v.get("low_confidence"):
            out.append("        ! low confidence: some witnesses unreadable")
    for n in report.get("notes", []):
        out.append(f"  note: {n}")
    out += ["", f"  MEAN SUBSCORE    {report.get('overall_score', 0.0):.3f}"
                f"   ({report['overall']})"]
    return "\n".join(out)


# --------------------------------------------------------------------------
# task.toml integration
# --------------------------------------------------------------------------

def annotate_envelope(envelope, report):
    """Carry the few report-level facts a consumer needs into the envelope.

    finalize() emits a fixed shape; an ungradable verdict and a normalised
    translation are things a pipeline must see without opening the full
    report, so they are attached here rather than buried.
    """
    if not report:
        return envelope
    if report.get("ungradable"):
        envelope["ungradable"] = report["ungradable"]
    if report.get("plane_normalised_by_mm") is not None:
        envelope["plane_normalised_by_mm"] = report["plane_normalised_by_mm"]
    return envelope


# --------------------------------------------------------------------------
# The two halves of grading, separately invocable.
#
# measure_candidate() is the expensive half: it needs Windows, a licensed
# SolidWorks and 15-30 s per part.  score_capture() is pure arithmetic over
# two dictionaries and runs in milliseconds on any machine.
#
# Keeping them separable is what makes threshold work tractable: a stored
# capture can be re-scored instantly after any rubric change, and a corpus of
# stored captures doubles as a regression suite that needs no CAD at all.
# --------------------------------------------------------------------------

def measure_candidate(path=None, close_after=False, baseline=None):
    """Open a part and take every measurement. Returns the capture dict.

    A part SolidWorks refuses to open produces a capture marked with
    `open_error` rather than an exception: a grading pipeline is better served
    by an envelope scoring zero with a stated reason than by a traceback that
    has to be read by hand.
    """
    baseline = baseline if baseline is not None else load_baseline()
    app = SC.attach_app()
    try:
        doc = SC.open_or_active(app, path)
    except Exception as exc:
        return {"schema": CAPTURE_SCHEMA, "harness_version": HARNESS_VERSION,
                "document": Path(path).name if path else None,
                "source_path": os.path.abspath(path) if path else None,
                "open_error": f"{type(exc).__name__}: {exc}",
                "bodies": [], "roles": {}, "plane_x_m": None}
    measured = capture(doc, baseline=baseline, plane_x=None)
    measured["source_path"] = os.path.abspath(path) if path else None
    if close_after and path:
        # Through the shared sweep rather than a bare CloseDoc: CloseDoc is
        # the one route that can raise the save-changes dialog, and capture
        # has just rebuilt the document, so it is dirty by construction.
        try:
            SW.close_all_documents(app)
        except Exception:
            pass
    for line in SW.session_report():
        print(f"  ! {line}", file=sys.stderr)
    return measured


def score_capture(measured, baseline=None, weights=None, quiet=False):
    """Grade a capture dict. No SolidWorks involved."""
    baseline = baseline if baseline is not None else load_baseline()
    schema = measured.get("schema")
    hard = ungradable_reason(baseline, measured)
    if hard:
        # Grader cannot run on a part with no bodies or no plane; produce the
        # same shaped report it would have.
        report = {"document": measured.get("document"),
                  "rebuild": measured.get("rebuild", {}),
                  "criteria": {k: {"score": 0.0, "status": FAIL,
                                   "evidence": "not evaluated -- candidate "
                                               "is ungradable"}
                               for k in ALL_CRITERIA},
                  "ungradable": {"reason": hard,
                                 "bodies": len(measured.get("bodies", [])),
                                 "plane_x_m": measured.get("plane_x_m")},
                  "overall_score": 0.0, "overall": FAIL,
                  "notes": [f"UNGRADABLE: {hard}"]}
    else:
        report = Grader(baseline, measured).grade()
    if schema and schema != CAPTURE_SCHEMA:
        report.setdefault("notes", []).append(
            f"capture schema {schema} differs from this harness's "
            f"{CAPTURE_SCHEMA}; some measurements may be missing")
    if not quiet:
        print(summarise(report, weights), file=sys.stderr)
    return report


def grade_candidate(path=None, close_after=False, weights=None):
    """Measure and score in one pass -- the default single-part flow."""
    baseline = load_baseline()
    measured = measure_candidate(path, close_after=close_after,
                                 baseline=baseline)
    # KEEP THE MEASUREMENT WHEN THE RUNNER ASKED FOR IT.
    #
    # `harness_cli._run_child` sets HARNESS_CAPTURE_JSON for every model of
    # a `--batch` grade run, and `batch_grade` explains itself: "Grading
    # keeps the capture. Measuring a corpus costs an hour of CAD and used to
    # leave nothing behind to re-score." This harness never read the
    # variable. It wrote HARNESS_REPORT_JSON and dropped the capture, so a
    # nine-model run left nine reports and no way to score any of them
    # again without re-opening SolidWorks nine times.
    #
    # Same shape as `--capture-only` writes, because `--score-from` reads
    # them back interchangeably and a capture that is nearly the right
    # shape is worse than none.
    write_env("HARNESS_CAPTURE_JSON",
              json.dumps(measured, indent=1, default=str))
    return score_capture(measured, baseline=baseline, weights=weights)


class PS3Harness(Harness):
    # No score-zeroing gates: every criterion carries its own weight, so a
    # single drifted body cannot wipe out an otherwise correct answer.  The
    # rebuild health gate in grade() is the only hard stop.
    MUST_PASS = ()
    CANDIDATE_OPTIONAL = True  # without an arg, grades the live document
    WEIGHTS = ALL_CRITERIA

    def main(self):
        # Same flow as the base class, but stamps this harness's version into
        # the envelope instead of harness_base's contract version.
        from common import harness_base as HB
        if HB.is_run_request():
            HB.runner_main(*sys.argv[2:6], fmt=self.RUNNER_FMT)
            raise SystemExit(0)
        candidate = (None if (self.CANDIDATE_OPTIONAL and len(sys.argv) == 1)
                     else HB.candidate_from_argv())
        self.timeout = HB.timeout_from_argv(self.BUILD_TIMEOUT_S)
        state = self.build_state(candidate)
        env = finalize(self.task_dir(), self.checks(state),
                       version=HARNESS_VERSION,
                       must_pass=self.MUST_PASS, weights=self.WEIGHTS)
        return annotate_envelope(env, getattr(self, "report", None))

    def build_state(self, candidate_path):
        return candidate_path

    def checks(self, state):
        """state is the candidate path (or None for the live document)."""
        report = grade_candidate(state, close_after=bool(state),
                                 weights=getattr(self, "WEIGHTS", None))
        self.report = report
        # Batch runs want every measurement, not just the score envelope.
        # Opt-in via the environment so the stdout contract (exactly one JSON
        # envelope) is untouched.
        dump = os.environ.get("HARNESS_REPORT_JSON")
        if dump:
            try:
                p = Path(dump)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(report, indent=1, default=str),
                             encoding="utf-8")
            except Exception as exc:
                print(f"[warn] could not write {dump}: {exc}", file=sys.stderr)
        registry = {}
        for cname, crit in report["criteria"].items():
            registry[cname] = (cname, clamp01(crit.get("score", 0.0)),
                               crit.get("evidence", ""))
        return registry


main = PS3Harness.as_main()


# --------------------------------------------------------------------------
# baseline capture -- the CLI referenced by task.toml and solidworks_capture
# --------------------------------------------------------------------------

#: Sections the grader compares against; freeze() refuses without them.
#: `modelling` is here because losing it once switched two thirds of the
#: hygiene criterion off silently (journal v2.1.1).
BASELINE_REQUIRED_KEYS = ("bodies", "roles", "plane_x_m", "modelling",
                          "role_bbox_m")


#: The frozen seed measurement. One object owns the path, the schema and
#: the completeness guard; the grader still receives a plain dict.
BASELINE = HB.Baseline(BASELINE_PATH, BASELINE_SCHEMA,
                       BASELINE_REQUIRED_KEYS)


def load_baseline():
    return BASELINE.load()

def load_solution_reference():
    with open(SOLUTION_REFERENCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def capture_baseline(path, out_path=None):
    """Re-freeze prompt/input.json from an unmodified seed part."""
    out_path = Path(out_path) if out_path else BASELINE_PATH
    app = SC.attach_app()
    doc = SC.open_or_active(app, path)

    census, meta = SC.feature_census(doc)
    census = census or {}
    raw, bodies, boxes, gmin, gmax = SC.capture_bodies(doc)
    smf = _small_face_counts(raw)
    roles = assign_roles(bodies, smf)
    plane = sym_plane(bodies)   # no baseline yet: pure geometry
    if plane is None:
        raise RuntimeError("could not determine the symmetry plane from this "
                           "part; refusing to write a baseline without one")
    hfaces = housing_faces(raw, roles.get("housing", []))
    sig = housing_signature(hfaces, plane)
    detail = housing_detail_side(hfaces, plane)
    lights = find_light_cluster_seed(hfaces, plane, gmin + gmax)
    intf = control_interference(raw, bodies, roles)

    hard = {n: c for n, (c, w) in census.items() if not w}
    warn = {n: c for n, (c, w) in census.items() if w}
    doc_title = str(z(doc.GetTitle))

    baseline = {
        "schema": BASELINE_SCHEMA,
        "row": "widen 15mm + left-handed mirror (PS3 controller)",
        "seed_document": doc_title,
        "units": "ALL lengths in metres, volumes m^3 -- SolidWorks API units",
        "plane_x_m": plane,
        # The hygiene criterion grades every probe as a delta against these
        # seed counts. Omit them and c0 quietly falls back to rebuild warnings
        # alone, which is the one probe that needs no baseline -- a re-freeze
        # would then look successful while removing two thirds of the check.
        "modelling": SC.modelling_census(doc),
        "global": {"bbox_m": gmin + gmax, "width_m": gmax[0] - gmin[0]},
        "roles": roles,
        "bodies": bodies,
        "role_bbox_m": role_bboxes(bodies, roles),
        "small_face_counts": smf,
        "housing_signature": sig,
        "housing_detail": detail,
        "light_cluster": lights,
        "interference": {"tested_pairs": intf["tested_pairs"],
                         "pairs": intf["pairs"],
                         "total_volume_m3": intf["total_volume_m3"]},
        "rebuild": {
            "captured_from": doc_title,
            "features": meta.get("features", 0),
            "errors": len(hard),
            "warnings": len(warn),
            "feature_errors": census,
            "note": "per-feature error census of the unmodified seed; the "
                    "health gate grades candidates as a delta vs this",
        },
        "measurement_notes": {
            "roles": "assigned geometrically; see assign_roles()",
            "plane": "mirror plane x from the mode of identical-pair "
                     "centroid midpoints",
            "housing_signature": "area-weighted 3rd moment of housing face "
                                 "box-centres about the plane",
            "interference": "control bodies pairwise + control vs housing",
        },
    }
    BASELINE.freeze(baseline, out_path,
                    dump=lambda r: json.dumps(r, indent=1))
    print(f"  "
          f"({len(bodies)} bodies, plane_x={plane * MM:.2f} mm, "
          f"{meta.get('features', 0)} features, {len(hard)} hard errors)",
          file=sys.stderr)
    return baseline


def capture_seed_rebuild(path=None, out_path=None):
    """Refresh only the rebuild census inside an existing baseline."""
    out_path = Path(out_path) if out_path else BASELINE_PATH
    baseline = json.loads(out_path.read_text(encoding="utf-8"))
    app = SC.attach_app()
    doc = SC.open_or_active(app, path)
    census, meta = SC.feature_census(doc)
    census = census or {}
    hard = {n: c for n, (c, w) in census.items() if not w}
    warn = {n: c for n, (c, w) in census.items() if w}
    baseline["rebuild"] = {
        "captured_from": str(z(doc.GetTitle)),
        "features": meta.get("features", 0),
        "errors": len(hard),
        "warnings": len(warn),
        "feature_errors": census,
        "note": "per-feature error census of the unmodified seed; the health "
                "gate grades candidates as a delta vs this",
    }
    out_path.write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    print(f"refreshed rebuild census in {out_path} "
          f"({meta.get('features', 0)} features, {len(hard)} hard errors)",
          file=sys.stderr)
    return baseline


def _out_flag(argv, default=None):
    """Pull `-o PATH` / `--out PATH` out of argv, returning (path, rest)."""
    rest, out = [], default
    i = 0
    while i < len(argv):
        if argv[i] in ("-o", "--out") and i + 1 < len(argv):
            out, i = argv[i + 1], i + 2
            continue
        rest.append(argv[i])
        i += 1
    return out, rest


# --------------------------------------------------------------------------
# batch mode -- many parts in one command
#
# One part is the graded contract: `harness.py part.SLDPRT` prints exactly one
# JSON envelope and nothing else, because that is what the evaluation pipeline
# reads.  Everything below is the development-time counterpart: a list of
# parts, a directory of them, or a directory of stored captures, tabulated
# side by side.  It lives here rather than in a companion script so there is
# one entry point to keep working and one place where a criterion name, a
# label or an output layout is defined.
#
# Each part is still graded in its own child process.  Measurement talks to a
# live SolidWorks over COM, and a call that wedges there would otherwise take
# the whole batch with it; a child can be timed out and the run continues.
# --------------------------------------------------------------------------

# Reference first, then the adversarials, then the loose ends.  Anything not
# named here still runs -- it is appended in the order it was given.
BATCH_ORDER = [
    "solution",
    "adversarial_widened_15mm_clusters_at_original_spacing",
    "adversarial_unwidened_shell_with_correct_clusters",
    "adversarial_widened_by_30mm",
    "adversarial_text_mirrored_incorrectly",
    "adversarial_only_one_button_cluster_mirrored",
    "adversarial_feature_tree_with_errors",
    "gpt5",
    "examples_solution_copy",
    "input",
]

SHORT = {C_HEALTH: "heal", C_HYGIENE: "hyg"}
SHORT.update({c: f"C{i + 1}" for i, c in enumerate(GEOMETRY_CRITERIA)})


def discover_models(task_dir):
    """{label: path} for a task directory laid out like the shipped one.

    Falls back to a recursive glob for any other directory, so pointing the
    batch at a folder of candidate submissions works without a convention.
    """
    task_dir = Path(task_dir)
    found = {}

    ref = task_dir / "solution" / "solution.SLDPRT"
    if ref.is_file():
        found["solution"] = ref
    ex = task_dir / "examples"
    if ex.is_dir():
        for p in sorted(ex.glob("*.SLDPRT")):
            # examples/solution.SLDPRT is byte-identical to the reference
            # (same sha256 in task.toml). Kept as a determinism check, but
            # labelled so it is not mistaken for an adversarial.
            found["examples_solution_copy" if p.stem == "solution"
                  else p.stem] = p
    seed = task_dir / "environment" / "input.SLDPRT"
    if seed.is_file():
        found["input"] = seed        # the "did nothing" control

    if not found:
        for p in sorted(task_dir.rglob("*.SLDPRT")):
            label, n = p.stem, 2
            while label in found:
                label, n = f"{p.stem}_{n}", n + 1
            found[label] = p

    ordered = {k: found[k] for k in BATCH_ORDER if k in found}
    for k, v in found.items():
        ordered.setdefault(k, v)
    return ordered


#: Where the dataset sits relative to this file, when nobody says otherwise.
#: From the repository root found by walking up for common/, not
#: from a count of `..`. Counting was right while the harness sat
#: beside the dataset and wrong the moment it moved inside it, and
#: a wrong task dir does not raise -- it finds no models.
DEFAULT_TASK_DIR = HERE.parents[2]
DEFAULT_CAPTURE_DIR = TASK_DIR.parent.parent / "captures"


# --------------------------------------------------------------------------
# command line -- the runner itself is common/harness_cli.py
# --------------------------------------------------------------------------

def envelope_from(report, weights=None):
    """report -> the graded envelope. Same contract as every other task."""
    weights = weights if weights is not None else PS3Harness.WEIGHTS
    checks = {n: (n, clamp01(c.get("score", 0.0)), c.get("evidence", ""))
              for n, c in report["criteria"].items()}
    env = finalize(PS3Harness.task_dir(), checks, version=HARNESS_VERSION,
                   must_pass=PS3Harness.MUST_PASS, weights=weights)
    return annotate_envelope(env, report)


def _score_stored(path):
    measured = json.loads(Path(path).read_text(encoding="utf-8"))
    return score_capture(measured,
                         weights=PS3Harness.WEIGHTS)


def _capture_note(cap):
    """One line about what a capture actually got hold of."""
    return f"{len(cap.get('bodies') or [])} bodies"


SPEC = HC.Spec(
    harness_file=__file__,
    version=HARNESS_VERSION,
    capture_schema=CAPTURE_SCHEMA,
    baseline_schema=BASELINE_SCHEMA,
    criteria=ALL_CRITERIA,
    short=SHORT,
    order=BATCH_ORDER,
    discover=discover_models,
    default_task_dir=DEFAULT_TASK_DIR,
    default_capture_dir=DEFAULT_CAPTURE_DIR,
    model_glob="*.SLDPRT",
    model_noun=("part", "parts"),
    model_width=52,                 # labels here run to 51 characters
    measure=lambda path: measure_candidate(path, close_after=bool(path)),
    dump=lambda cap: json.dumps(cap, indent=1, default=str),
    capture_note=_capture_note,
    score=_score_stored,
    summarise=lambda report: summarise(
        report, PS3Harness.WEIGHTS),
    envelope=envelope_from,
    capture_baseline=capture_baseline,
    # This task's second frozen artefact: the seed's own rebuild state, which
    # the health criterion is a delta from.
    extra_modes={"--capture-seed-rebuild":
                 lambda argv: capture_seed_rebuild(
                     argv[0] if argv else None,
                     argv[1] if len(argv) > 1 else None)},
    extra_usage="  harness.py --capture-seed-rebuild SEED.SLDPRT [out.json]\n",
    harness_cli=PS3Harness.cli,
)


def cli(argv=None):
    HC.cli(SPEC, argv)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cli()
