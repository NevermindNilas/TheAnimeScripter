import pytest

# CI installs pytest and nothing else, so these have to be skipped rather than
# imported: a bare `import cv2` here aborts collection for the WHOLE suite, not
# just this module. src.depth.backends._shared pulls in cv2 and torch too.
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
_shared = pytest.importorskip("src.depth.backends._shared")

SlidingWindowNormalizer = _shared.SlidingWindowNormalizer
VideoRangeNormalizer = _shared.VideoRangeNormalizer


def _robust_normalize(frame, valid=None):
    sample = frame if valid is None else frame[valid]
    low, high = np.quantile(sample, (0.02, 0.98))
    return np.clip((frame - low) / max(high - low, 1e-6), 0.0, 1.0)


def test_affine_ambiguity_is_removed_without_pixel_history():
    y, x = np.mgrid[:96, :160].astype(np.float32)
    scene = 0.2 + 0.5 * x / x.max() + 0.2 * np.sin(y / 9.0)
    normalizer = SlidingWindowNormalizer(flow_size=160)

    first = normalizer.normalize(scene)
    second = normalizer.normalize(scene * 1.7 - 0.35)

    np.testing.assert_allclose(first, second, atol=2e-5)


def test_moving_object_does_not_leave_previous_pixels_behind():
    first = np.tile(np.linspace(0.1, 0.9, 160, dtype=np.float32), (96, 1))
    second = first.copy()
    first[30:60, 30:55] = 0.9
    second[30:60, 85:110] = 0.9
    normalizer = SlidingWindowNormalizer(flow_size=160)

    normalizer.normalize(first)
    output = normalizer.normalize(second)
    current = _robust_normalize(second)

    # The output may receive one global affine correction, but no local data
    # from the old object location may be injected.
    design = np.column_stack((current.ravel(), np.ones(current.size)))
    scale, shift = np.linalg.lstsq(design, output.ravel(), rcond=None)[0]
    reconstructed = np.clip(scale * current + shift, 0.0, 1.0)
    np.testing.assert_allclose(output, reconstructed, atol=2e-5)


def test_invalid_region_does_not_change_valid_normalization():
    frame = np.tile(np.linspace(2.0, 8.0, 160, dtype=np.float32), (96, 1))
    valid = np.ones_like(frame, dtype=bool)
    valid[:, :40] = False
    corrupted = frame.copy()
    corrupted[:, :40] = -10_000.0

    expected = _robust_normalize(frame, valid)
    output = SlidingWindowNormalizer(flow_size=160).normalize(corrupted, mask=valid)

    np.testing.assert_allclose(output[valid], expected[valid], atol=2e-5)


def test_scene_cut_rejects_temporal_correction():
    rng = np.random.default_rng(7)
    first = cv2.GaussianBlur(rng.random((96, 160), dtype=np.float32), (0, 0), 3)
    second = np.flipud(rng.random((96, 160), dtype=np.float32))
    normalizer = SlidingWindowNormalizer(flow_size=160, min_coverage=0.20)

    normalizer.normalize(first)
    output = normalizer.normalize(second)

    np.testing.assert_allclose(output, _robust_normalize(second), atol=2e-5)


def test_empty_mask_returns_finite_zeros():
    frame = np.full((32, 48), np.nan, dtype=np.float32)
    output = SlidingWindowNormalizer().normalize(frame, mask=np.zeros_like(frame, bool))

    assert np.isfinite(output).all()
    assert not output.any()


def test_video_range_normalizer_never_imports_previous_pixels():
    first = np.tile(np.linspace(0.0, 10.0, 160, dtype=np.float32), (96, 1))
    second = np.tile(np.linspace(1.0, 9.0, 160, dtype=np.float32), (96, 1))
    second[25:70, 60:100] += 2.0
    normalizer = VideoRangeNormalizer(adapt_rate=0.0)

    normalizer.normalize(first)
    output = normalizer.normalize(second)

    expected = np.clip(
        (second - normalizer.low) / (normalizer.high - normalizer.low), 0, 1
    )
    np.testing.assert_allclose(output, expected, atol=2e-6)
