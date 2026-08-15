"""The identity rules, pinned to the cases that produced them.

Every test here is a defect this pipeline actually shipped, measured on real
footage and written up in docs/tracking-comparison.md. They were prose in a
docstring; prose does not fail when someone changes a threshold.

The rules are worth pinning because each one is a veto with a blast radius:
merge_tracklets decides whether two fragments are one man, the split gate
decides whether a track is named at all, and both have been wrong in a way
that put the wrong name on a player rather than no name. Run with:

    python -m pytest tests -q
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

# The rules under test are pure, but the modules holding them import cv2 at the
# top. Say so in one line rather than through an ImportError traceback.
pytest.importorskip(
    "cv2", reason="run with the project venv, which has the pipeline's deps: "
                  "~/.venvs/courtvision/Scripts/python -m pytest tests")

import identify            # noqa: E402
import overlap             # noqa: E402
import render_final        # noqa: E402
import score               # noqa: E402


# --- decisive ---------------------------------------------------------------

def test_a_coin_toss_has_no_opinion():
    """The 11-10 team split that must not be allowed to veto a merge.

    Brunson's recovered track spanned two different people, so its crops
    clustered 11 against 10. That is not evidence of a club.
    """
    assert not identify.decisive(Counter({"knicks": 11, "pistons": 10}))


def test_a_clear_majority_has_one():
    assert identify.decisive(Counter({"knicks": 30, "pistons": 2}))


def test_an_empty_vote_is_not_decisive():
    assert not identify.decisive(Counter())


# --- one_number_misread -----------------------------------------------------

def test_a_clipped_read_is_one_number():
    """seg_02m44.15s_10s: Bridges read '5' 29 times and '25' 20 times.

    The crop dropped the leading digit. Treating those as two players deleted
    a name the roster matching had already resolved correctly.
    """
    assert identify.one_number_misread("5", "25")


def test_a_clipped_tail_is_also_one_number():
    assert identify.one_number_misread("5", "52")


def test_two_real_numbers_are_not_a_misread():
    """The case the split gate exists for: track 6 read '0' and '32'.

    Duren is Pistons 0, Towns is Knicks 32. The track followed one and then
    the other, and the render put Towns's name on a Piston.
    """
    assert not identify.one_number_misread("0", "32")


@pytest.mark.parametrize("a,b", [("1", "2"), ("11", "12"), ("23", "32")])
def test_unrelated_numbers_are_never_a_misread(a, b):
    assert not identify.one_number_misread(a, b)


# --- merge_tracklets --------------------------------------------------------

def _votes(number, n):
    return Counter({number: n})


def test_a_handover_seam_still_merges():
    """Beasley, seg_01m10.87s_19s: fragments 0-941 and 935-1152.

    Seven frames of double report, 0.12s. Demanding a clean seam left him
    unlabelled from 15.7s onward.
    """
    alias = identify.merge_tracklets(
        life={1: (0, 941), 2: (935, 1152)},
        number_votes={1: _votes("5", 40), 2: _votes("5", 20)},
        team_votes={1: Counter({"pistons": 30}), 2: Counter({"pistons": 15})},
        min_votes=5)
    assert alias == {2: 1}


def test_demanding_a_clean_seam_is_what_lost_him():
    """The same fragments with the tolerance the code used to have."""
    alias = identify.merge_tracklets(
        life={1: (0, 941), 2: (935, 1152)},
        number_votes={1: _votes("5", 40), 2: _votes("5", 20)},
        team_votes={1: Counter({"pistons": 30}), 2: Counter({"pistons": 15})},
        min_votes=5, overlap_frames=0)
    assert alias == {}


def test_real_coexistence_is_a_duplicate_not_a_merge():
    """Two ids on the floor together for seconds is a different fault."""
    alias = identify.merge_tracklets(
        life={1: (0, 1152), 2: (0, 1152)},
        number_votes={1: _votes("5", 40), 2: _votes("5", 20)},
        team_votes={1: Counter({"pistons": 30}), 2: Counter({"pistons": 15})},
        min_votes=5)
    assert alias == {}


def test_one_number_across_two_rosters_does_not_merge():
    """Cunningham is Pistons 2, McBride is Knicks 2."""
    alias = identify.merge_tracklets(
        life={1: (0, 500), 2: (600, 1152)},
        number_votes={1: _votes("2", 40), 2: _votes("2", 20)},
        team_votes={1: Counter({"pistons": 30}), 2: Counter({"knicks": 25})},
        min_votes=5)
    assert alias == {}


def test_an_undecided_team_vote_abstains_rather_than_blocks():
    """The 11-10 split must not veto a merge the numbers agree on."""
    alias = identify.merge_tracklets(
        life={1: (0, 500), 2: (600, 1152)},
        number_votes={1: _votes("5", 40), 2: _votes("5", 20)},
        team_votes={1: Counter({"pistons": 30}),
                    2: Counter({"knicks": 11, "pistons": 10})},
        min_votes=5)
    assert alias == {2: 1}


def test_a_third_fragment_is_measured_against_the_whole_group():
    """A-B and B-C disjoint, but A-C overlap: C must not sneak in.

    The span is checked against the merged group, not against whichever
    member happens to be compared first.
    """
    alias = identify.merge_tracklets(
        life={1: (0, 400), 2: (800, 1152), 3: (200, 600)},
        number_votes={1: _votes("5", 40), 2: _votes("5", 30), 3: _votes("5", 20)},
        team_votes={},
        min_votes=5)
    assert 2 in alias and alias[2] == 1
    assert 3 not in alias


def test_a_track_read_too_few_times_is_not_merged():
    alias = identify.merge_tracklets(
        life={1: (0, 500), 2: (600, 1152)},
        number_votes={1: _votes("5", 40), 2: _votes("5", 2)},
        team_votes={}, min_votes=5)
    assert alias == {}


# --- collapse spans ---------------------------------------------------------

def test_a_track_collapsed_against_two_others_reports_one_span():
    """Brunson sat on both of Duren's ids, so the span arrives twice."""
    spans = overlap.collapse_spans([
        {"kind": "collapse", "pair": (1, 9), "start": 100, "end": 200},
        {"kind": "collapse", "pair": (1, 10), "start": 150, "end": 260},
    ])
    assert spans[1] == [(100, 260)]


def test_a_duplicate_is_not_a_collapse():
    spans = overlap.collapse_spans([
        {"kind": "duplicate", "pair": (9, 10), "start": 0, "end": 500},
    ])
    assert spans == {}


def test_contamination_is_bounded_by_the_span():
    spans = {7: [(100, 200)]}
    assert not overlap.is_contaminated(spans, 7, 99)
    assert overlap.is_contaminated(spans, 7, 100)
    assert overlap.is_contaminated(spans, 7, 200)
    assert not overlap.is_contaminated(spans, 7, 201)
    assert not overlap.is_contaminated(spans, 8, 150)


# --- surname_of -------------------------------------------------------------

@pytest.mark.parametrize("full,expected", [
    ("Tim Hardaway Jr.", "Hardaway Jr."),     # rendered as "#8 Jr."
    ("Ronald Holland II", "Holland II"),
    ("Lindy Waters III", "Waters III"),
    ("Jalen Brunson", "Brunson"),
    ("Cade Cunningham", "Cunningham"),
])
def test_a_suffix_is_not_a_surname(full, expected):
    assert render_final.surname_of(full) == expected


def test_a_single_word_name_survives():
    assert render_final.surname_of("Brunson") == "Brunson"


# --- truth keying -----------------------------------------------------------

def test_identity_is_club_and_number_not_number_alone():
    """#8 is Anunoby for the Knicks and Hardaway Jr. for the Pistons.

    Keying on the number counted the two as one man, and the 13s segment
    reported 90% coverage while labelling all ten correctly on every frame.
    """
    knicks = score.truth_at([["8", 0, 749, "knicks"]], 300)
    pistons = score.truth_at([["8", 0, 749, "pistons"]], 300)
    assert knicks == ("8", "knicks")
    assert pistons == ("8", "pistons")
    assert knicks != pistons


def test_a_segment_written_before_clubs_carries_no_club():
    assert score.truth_at([["8", 0, 749]], 300) == ("8", None)


def test_a_frame_outside_every_segment_has_no_truth():
    assert score.truth_at([["8", 0, 100, "knicks"]], 300) is None
    assert score.truth_at(None, 300) is None


# --- what the renderer would draw -------------------------------------------

def _box(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2), float(y2)]


def test_a_merged_player_is_drawn_once():
    """Measured at 15.60s: tracks 107 and 163, two '#5 BEASLEY' chips on one
    man. The bigger box is the one whose mask actually covers him."""
    frames = {0: [{"tid": 107, "box": _box(1085, 339, 1158, 480)},
                  {"tid": 163, "box": _box(1085, 299, 1160, 480)}]}
    idn = {107: {"number": "5", "merged_into": 107},
           163: {"number": "5", "merged_into": 107}}
    rows = score.drawn(frames, idn, {}, 0)
    assert len(rows) == 1
    assert rows[0]["tid"] == 163          # the taller box


def test_a_contaminated_track_is_not_drawn():
    frames = {0: [{"tid": 1, "box": _box(0, 0, 10, 10)}]}
    idn = {1: {"number": "5"}}
    assert score.drawn(frames, idn, {1: [(0, 100)]}, 0) == []


def test_a_track_without_a_number_is_not_drawn():
    frames = {0: [{"tid": 1, "box": _box(0, 0, 10, 10)}]}
    assert score.drawn(frames, {1: {"number": None}}, {}, 0) == []


def test_an_ignored_track_is_not_drawn():
    frames = {0: [{"tid": 1, "box": _box(0, 0, 10, 10)}]}
    idn = {1: {"number": "5", "ignored": "split-identity"}}
    assert score.drawn(frames, idn, {}, 0) == []
