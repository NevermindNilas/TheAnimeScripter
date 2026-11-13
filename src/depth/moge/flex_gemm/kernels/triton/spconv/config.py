import triton
from ..utils import get_autotune_config


# Forward / bwd_input shared config (kernel uses B1=M-tile, B2=Co-tile, BK=Ci-tile).
# Configs pruned based on observed autotune cache hits on A100 (V=27 submanifold):
# winners cluster around (B1=128, B2 in {64,128}, BK in {32,64}) and one large
# variant (B1=256, B2=64, BK=64). Configs with B1=32, B2=32, or BK=16 were
# never picked; small B1=64 variants almost never picked. First entry is the
# broadly-good default used when autotune is disabled.
autotune_config = get_autotune_config(
    platform={
        'cuda': [
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=3, num_warps=8),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=5, num_warps=2),
            # ieee-fp32 safe fallback (small tiles, num_stages=2). Picked only
            # when larger tiles OOR (e.g. input_precision='ieee' fp32 path).
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=2, num_warps=4),
        ],
        'hip': [
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 16, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        ]
    },
    device={
        'A100': [
            # Cold-start default: broadly good across mid-large channels.
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=4),
            # Observed winners.
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=2),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 64}, num_stages=4, num_warps=4),
            # Alternatives kept for shapes not yet probed.
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=4, num_warps=8),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=4, num_warps=2),
            # ieee-fp32 safe fallback (small tiles, num_stages=2). Picked only
            # when larger tiles OOR (e.g. input_precision='ieee' fp32 path).
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=2, num_warps=4),
        ],
        'H100': [
            # SMEM budget on H100 = 227 KB. For the masked variant cost is
            # ~max(num_stages, 3) * 2 * BK * (B1+B2) bytes (Triton uses a
            # double-buffered epilogue at stages=2, so stages=2 occupies the
            # same SMEM as stages=3). BK=128 with B1+B2 >= 256 exceeds the
            # budget at any num_stages we tested, so it is omitted.
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64},  num_stages=5, num_warps=4),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64},  num_stages=5, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64},  num_stages=5, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 64},  num_stages=5, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32},  num_stages=5, num_warps=4),
            # Small-tile / high-occupancy entries for small-N shapes. Aligns
            # with v1's empirical H100 winners (B1*B2 <= 64*64 num_warps=2),
            # which let multiple CTAs co-reside per SM and hide indirect-load
            # latency on shapes where the cost is gather-bound, not FMA-bound.
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64},  num_stages=4, num_warps=2),
            triton.Config({'B1': 32,  'B2': 64,  'BK': 32},  num_stages=5, num_warps=2),
            triton.Config({'B1': 64,  'B2': 32,  'BK': 32},  num_stages=5, num_warps=2),
            # ieee-fp32 safe fallback (small tiles, num_stages=2). Picked only
            # when larger tiles OOR (e.g. input_precision='ieee' fp32 path).
            # triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=2, num_warps=4),
        ],
        'MI300X': [
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=2),
        ],
    }
)


# bwd_weight tile semantics differ from fwd: B1 is the Co tile, B2 drives the
# V*Ci tile (split into BCi/BV by heuristic), and BK is the M-reduction tile.
# Since M is typically 10-100k while Ci stays in the hundreds, the optimal BK
# for bwd_weight is much larger than the fwd's Ci-reduction tile.
#
# Configs pruned based on observed autotune cache hits on A100: winners are
# B1=128 (always; B1=64 / B1=256 never picked), B2 in {128, 256}, BK in {64, 128}.
# First entry is the cold-start default.
bwd_weight_autotune_config = get_autotune_config(
    platform={
        'cuda': [
            triton.Config({'B1': 128, 'B2': 128, 'BK': 128}, num_stages=3, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64},  num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64},  num_stages=3, num_warps=8),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 128}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 128}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 128}, num_stages=5, num_warps=2),
            # ieee-fp32 safe fallback (small tiles, num_stages=2). Picked only
            # when larger tiles OOR (e.g. input_precision='ieee' fp32 path).
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32},  num_stages=2, num_warps=4),
        ],
        'hip': [
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 32, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        ]
    },
    device={
        'A100': [
            # Cold-start default: EVEN-path winner on Ci=256.
            triton.Config({'B1': 128, 'B2': 128, 'BK': 128}, num_stages=4, num_warps=4),
            # Observed winners.
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64},  num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64},  num_stages=4, num_warps=8),
            # Alternatives kept for shapes not yet probed.
            triton.Config({'B1': 128, 'B2': 256, 'BK': 128}, num_stages=3, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64},  num_stages=4, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 128}, num_stages=3, num_warps=8),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 128}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 128}, num_stages=4, num_warps=2),
            # ieee-fp32 safe fallback (small tiles, num_stages=2). Picked only
            # when larger tiles OOR (e.g. input_precision='ieee' fp32 path).
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32},  num_stages=2, num_warps=4),
        ],
        'H100': [
            # SMEM budget on H100 = 227 KB. For masked bwd_weight the cost is
            # ~(num_stages+1) * 2 * BK * (B1+B2) bytes for stages>=3, and
            # behaves like stages=3 fwd (4 * 2 * BK * (B1+B2)) at stages=2.
            # Main working set: BK=64 stages=5 (192 KB on 128x128) wins on
            # the production token shapes.
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64},  num_stages=5, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64},  num_stages=4, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64},  num_stages=3, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64},  num_stages=3, num_warps=8),
            # BK=128 fallbacks: kept for hypothetical large-N dense shapes;
            # most production shapes prefer BK=64 stages=5 over these.
            triton.Config({'B1': 128, 'B2': 64,  'BK': 128}, num_stages=3, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 128}, num_stages=4, num_warps=2),
            # Small-tile / high-occupancy entries: critical for the K-tile
            # heuristic to find an occupancy-bound winner on small-token
            # shapes (v1's measured H100 best was B1*B2 <= 64*64 num_warps=2).
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64},  num_stages=4, num_warps=2),
            triton.Config({'B1': 32,  'B2': 64,  'BK': 64},  num_stages=5, num_warps=2),
            triton.Config({'B1': 64,  'B2': 32,  'BK': 64},  num_stages=5, num_warps=2),
            triton.Config({'B1': 32,  'B2': 64,  'BK': 32},  num_stages=5, num_warps=2),
            # ieee-fp32 safe fallback (small tiles, num_stages=2). Picked only
            # when larger tiles OOR (e.g. input_precision='ieee' fp32 path).
            # triton.Config({'B1': 64,  'B2': 64,  'BK': 32},  num_stages=2, num_warps=4),
        ],
        'MI300X': [
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64,  'waves_per_eu': 2}, num_stages=2, num_warps=2),
        ],
    }
)
