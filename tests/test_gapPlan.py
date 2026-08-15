"""Gap planning for --smooth_dedup and for fractional interpolation factors.

``gapPlan`` answers one question: given that the previous frame handed to the
interpolation driver sat at source position ``prevPos`` and the current one sits
at ``curPos``, how many frames must be synthesized between them and at which
timesteps. Under --smooth_dedup ``curPos - prevPos`` is wider than 1 because duplicates
were dropped from the stream; without it the interval is always one frame wide.
"""

from fractions import Fraction

import pytest

from src.interpolate._timesteps import cutHoldSplit, gapPlan, interpolateTimestep


def factorParts(factor):
    fraction = Fraction(factor).limit_denominator(100)
    return fraction.numerator, fraction.denominator


# --------------------------------------------------------------------------- #
# Plain CFR: unit-wide gaps must not change
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factor", [2, 3, 4, 8])
@pytest.mark.parametrize("prevPos", [0, 1, 7, 1000])
def testUnitGapMatchesTheDriverDefault(factor, prevPos):
    """An integer factor over one source frame reproduces the driver's own
    ``(i + 1) / factor`` ladder exactly, so routing the common path through
    gapPlan cannot change output."""
    count, timesteps = gapPlan(prevPos, prevPos + 1, factor, 1)

    assert count == factor - 1
    assert timesteps == [interpolateTimestep(i, factor - 1) for i in range(count)]


# --------------------------------------------------------------------------- #
# --smooth_dedup: a wider gap earns proportionally more slots
# --------------------------------------------------------------------------- #


def testOnTwosAtFactorTwo():
    # One duplicate swallowed: the gap covers 2 source frames, so 2*2 output
    # slots, 3 of them synthesized.
    count, timesteps = gapPlan(0, 2, 2, 1)

    assert count == 3
    assert timesteps == pytest.approx([0.25, 0.5, 0.75])


def testOnThreesAtFactorTwo():
    count, timesteps = gapPlan(3, 6, 2, 1)

    assert count == 5
    assert timesteps == pytest.approx([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])


def testOnTwosAtFactorThree():
    count, timesteps = gapPlan(0, 2, 3, 1)

    assert count == 5
    assert timesteps == pytest.approx([i / 6 for i in range(1, 6)])


@pytest.mark.parametrize("span", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("factor", [2, 3])
def testTimestepsStayStrictlyInsideTheOpenInterval(span, factor):
    _, timesteps = gapPlan(5, 5 + span, factor, 1)

    assert all(0.0 < t < 1.0 for t in timesteps)
    assert timesteps == sorted(timesteps)


# --------------------------------------------------------------------------- #
# Frame accounting over a whole clip
# --------------------------------------------------------------------------- #


def runPattern(spans, factor):
    """Total output frames for a source whose kept frames have these spans.

    Mirrors the pipeline: the first kept frame emits nothing (the driver is
    still caching its anchor) and is then written; every later one emits its
    gap's synthesized frames and is written.
    """
    num, den = factorParts(factor)
    pos = 0
    total = 1  # the first frame is written with no gap before it
    for span in spans:
        count, _ = gapPlan(pos, pos + span, num, den)
        total += count + 1
        pos += span
    return total, pos


@pytest.mark.parametrize("factor", [2, 3, 4])
def testOnTwosPreservesDuration(factor):
    # 10 unique drawings held for 2 frames each = 20 source frames after the
    # first. Output must be sourceFrames * factor to keep the same duration at
    # fps * factor.
    spans = [2] * 10
    total, sourceFrames = runPattern(spans, factor)

    assert sourceFrames == 20
    assert total == sourceFrames * factor + 1


def testMixedTwosAndThreesPreservesDuration():
    spans = [2, 3, 2, 2, 3, 1, 2]
    total, sourceFrames = runPattern(spans, 2)

    assert total == sourceFrames * 2 + 1


# --------------------------------------------------------------------------- #
# Factor 1: --smooth_dedup --interpolate_factor 1 regenerates duplicate slots
# in place -- the route for After Effects renders where a time remap was baked
# into the clip before interpolation (already at comp fps, full of dup runs).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prevPos", [0, 1, 7, 1000])
def testUnitGapAtFactorOneEmitsNothing(prevPos):
    # A non-duplicate frame passes straight through: no inserts, no resample.
    count, timesteps = gapPlan(prevPos, prevPos + 1, 1, 1)

    assert count == 0
    assert timesteps == []


@pytest.mark.parametrize("span", [2, 3, 4, 8, 12])
def testWidenedGapAtFactorOneRegeneratesEachDuplicateSlot(span):
    # A run of span-1 dropped duplicates earns exactly span-1 synthesized
    # frames at k/span -- one true in-between per duplicated slot, so the
    # baked slowdown keeps its duration and fps.
    count, timesteps = gapPlan(0, span, 1, 1)

    assert count == span - 1
    assert timesteps == pytest.approx([k / span for k in range(1, span)])


def testFactorOnePreservesDuration():
    # Baked 4x slowdown: every unique frame is followed by 3 dropped dups.
    spans = [4] * 6
    total, sourceFrames = runPattern(spans, 1)

    assert total == sourceFrames + 1


def testFactorOneCutSplitKeepsNothingForTheIncomingShot():
    # count - fromPrev == factor - 1 == 0: on a cut the whole widened gap
    # belongs to the outgoing shot, matching the span-1 invariant below.
    for span in (2, 4, 9):
        count, _ = gapPlan(0, span, 1, 1)
        fromPrev = cutHoldSplit(0, span, 1, 1)
        assert count - fromPrev == 0
        assert 0 <= fromPrev <= count


# --------------------------------------------------------------------------- #
# Fractional factors
# --------------------------------------------------------------------------- #


def testFractionalFactorFillsEveryOutputSlotExactlyOnce():
    """At 2.5x each source frame claims output index floor(pos * 5/2) and the
    gaps fill everything in between -- no slot emitted twice, none skipped."""
    num, den = factorParts(2.5)
    assert (num, den) == (5, 2)

    emitted = []
    pos = 0
    emitted.append(0)  # the first source frame owns output index 0
    for _ in range(8):
        count, timesteps = gapPlan(pos, pos + 1, num, den)
        first = (pos * num) // den + 1
        emitted.extend(range(first, first + count))
        pos += 1
        emitted.append((pos * num) // den)
        # Each timestep must place the frame on its own output slot.
        assert timesteps == pytest.approx(
            [((first + i) * den - pos * num + num) / num for i in range(count)]
        )

    assert emitted == list(range(len(emitted)))


def testFractionalFactorFrameCount():
    total, sourceFrames = runPattern([1] * 20, 2.5)

    assert sourceFrames == 20
    assert total == int(20 * 2.5) + 1


# --------------------------------------------------------------------------- #
# Splitting a held gap between the shot it leaves and the shot it enters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("prevPos", [0, 1, 37])
def testUnitGapGivesTheCutTheWholeHold(factor, prevPos):
    """Without --smooth_dedup a gap is one source frame wide, so a scene cut
    fills all of it from the incoming frame, exactly as it always has."""
    assert cutHoldSplit(prevPos, prevPos + 1, factor, 1) == 0


@pytest.mark.parametrize(
    "span, factor, expected",
    [
        (2, 2, 2),  # 3 slots: 2 belong to the outgoing shot, 1 is the cut
        (3, 2, 4),
        (6, 2, 10),
        (12, 2, 22),
        (2, 3, 3),  # 5 slots: 3 outgoing, 2 cut
        (4, 4, 12),
    ],
)
def testWidenedGapKeepsItsDuplicatesOnTheOutgoingShot(span, factor, expected):
    assert cutHoldSplit(0, span, factor, 1) == expected


@pytest.mark.parametrize("span", [1, 2, 3, 7, 20])
@pytest.mark.parametrize("factor", [2, 3, 5])
@pytest.mark.parametrize("prevPos", [0, 5, 999])
def testSplitNeverExceedsTheGapItSplits(span, factor, prevPos):
    """The cut always keeps at least its own final source step, and the outgoing
    shot never claims a slot that does not exist."""
    count, _ = gapPlan(prevPos, prevPos + span, factor, 1)
    fromPrev = cutHoldSplit(prevPos, prevPos + span, factor, 1)

    assert 0 <= fromPrev <= count
    assert count - fromPrev == factor - 1


def testSplitHandlesFractionalFactors():
    # 2.5x over a gap of 2 source frames: 4 synthesized slots, the last source
    # step owning whichever of them fall inside it.
    count, _ = gapPlan(0, 2, 5, 2)
    fromPrev = cutHoldSplit(0, 2, 5, 2)

    assert count == 4
    assert 0 < fromPrev < count


def testFractionalFactorWithDuplicateSpans():
    # 2.5x on twos: each gap covers 2 source frames, i.e. 5 output slots.
    count, timesteps = gapPlan(0, 2, 5, 2)

    assert count == 4
    assert timesteps == pytest.approx([0.2, 0.4, 0.6, 0.8])
