import itertools
from typing import Optional, Tuple, Literal
from numbers import Number
import torch
from torch import Tensor

import triton
import triton.language as tl
from .utils import pad_to_size_along_dim


__all__ = [
    "hashmap_build",
    "hashmap_unique",
]

HASHMAP_LOAD_FACTOR = 0.3


@triton.jit
def _reduce_all(x, axis=None):
    return tl.min(x, axis=axis) > 0

@triton.jit
def _reduce_any(x, axis=None):
    return tl.max(x, axis=axis) > 0


@triton.jit
def _vec_load(ptr: tl.pointer_type, mask: tl.tensor, D: tl.constexpr) -> tl.tensor:
    "Load a vector key from memory given a pointer."
    # Hint that the per-row inner D-element load is contiguous and aligned,
    # so Triton can emit a single wide LDG (e.g. LDG.128 for D=4 int32) per
    # (m,v) lane instead of D scalar LDG.32 ops.
    inner = tl.max_contiguous(tl.multiple_of(tl.arange(0, D), D), D)
    vec = tl.load(tl.expand_dims(ptr, -1) + inner, mask=tl.expand_dims(mask, -1), other=0)
    return vec


@triton.jit
def _vec_hash_32bit(vec: tl.tensor, D: tl.constexpr) -> tl.tensor:
    if D == 1:
        # scalar hash
        h = tl.reshape(vec, vec.shape[:-1]).to(tl.uint32)
        h ^= h >> 16
        h *= 0x7FEB352D
        h ^= h >> 15
        h *= 0x846CA68B
        h ^= h >> 16
        return h.to(tl.int32)

    if D == 2:
        # pack to int64 hash
        h64 = tl.sum(vec.to(tl.uint32).to(tl.uint64) << (tl.arange(0, 2).to(tl.uint64) * 32), axis=-1)
        h64 ^= h64 >> 33
        h64 *= 0xFF51AFD7ED558CCD
        h64 ^= h64 >> 33
        h64 *= 0xC4CEB9FE1A85EC53
        h64 ^= h64 >> 33
        return (h64 ^ (h64 >> 32)).to(tl.int32)
    
    # Vectorized hash
    idx = tl.arange(0, D)
    seed = idx.to(tl.uint32) + 0x9E3779B9
    seed = (seed ^ (seed >> 16)) * 0x7FEB352D
    seed = (seed ^ (seed >> 15)) * 0x846CA68B
    seed = seed ^ (seed >> 16)
    mult = seed | 1

    v = vec.to(tl.uint32)
    v = v + seed
    v ^= v >> 15
    v *= 0x2C1B3C6D
    v ^= v >> 12

    h = tl.sum(v * mult, axis=-1)
    h ^= h >> 16
    h *= 0x7FEB352D
    h ^= h >> 15
    h *= 0x846CA68B
    h ^= h >> 16
    return h.to(tl.int32)


@triton.jit
def _scalar_hash_32bit(v: tl.tensor) -> tl.tensor:
    """Murmur-like mixer for a single 32-bit key.

    Used by the ``D_32 == 1`` fast path in the build/lookup/unique kernels, where
    the caller has serialized a multi-dim coordinate into a single int32 (e.g.
    ``neighbor_map`` collapses a 4D coordinate by ``shape``). Avoids the vector
    seed-array & reduce-sum overhead of ``_vec_hash_32bit``.
    """
    v = v.to(tl.uint32)
    v ^= v >> 16
    v *= 0x7FEB352D
    v ^= v >> 15
    v *= 0x846CA68B
    v ^= v >> 16
    return v.to(tl.int32)


@triton.jit
def _vec_pack_little_endian_to_int32(vec: tl.tensor) -> tl.tensor:
    """Pack a little-endian integer vector into int32 words."""
    tl.static_assert(
        vec.dtype.itemsize == 1 or vec.dtype.itemsize == 2 or vec.dtype.itemsize == 4,
        "Unsupported query_vec element width",
    )
    if vec.dtype.itemsize == 4:
        return vec.to(tl.int32)

    if vec.dtype.itemsize == 2:
        vec_u16 = tl.reshape(tl.cast(vec, tl.uint16, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 2, 2)
        return tl.sum(vec_u16.to(tl.uint32) << (tl.arange(0, 2) << 4), axis=-1).to(tl.int32)

    if vec.dtype.itemsize == 1:
        vec_u8 = tl.reshape(tl.cast(vec, tl.uint8, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 4, 4)
        return tl.sum(vec_u8.to(tl.uint32) << (tl.arange(0, 4) << 3), axis=-1).to(tl.int32)


# NOTE: an earlier version of this file used a custom ``tl.inline_asm_elementwise``
# helper to emit a PTX-predicated ``atom.global.cas.b32``, so masked-out
# lanes wouldn't issue the atomic at all. In the build kernel this tripped
# a Triton mis-compile (~2x occupancy) that survived even an explicit
# materialization barrier, and gains in the unique kernel were marginal.
# We now use plain ``tl.atomic_cas`` with the ``tl.where(mask, -1, -2)``
# trick everywhere -- inactive lanes still touch L2 but never collide.


@triton.jit
def _hashmap_build_kernel_32bit(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    keys_ptr: tl.const,
    n_keys: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.static_assert(D * keys_ptr.dtype.element_ty.itemsize % 4 == 0, "keys byte width must be divisible by 4")
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32, const=True))
    D_32: tl.constexpr = D * keys_ptr.dtype.element_ty.itemsize // 4

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_keys   

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    # Compute hash and stored value: upper bits are the tag, lower bits
    # are the key's index. (index must be < hashmap_size, which the host
    # ensures by sizing the table for load factor 0.3.)
    if D_32 == 1:
        key = tl.load(keys_ptr_32 + idx, mask=mask, other=0)
        hash_val = _scalar_hash_32bit(key)
    else:
        key_vec = _vec_load(keys_ptr_32 + idx * D_32, mask=mask, D=D_32)
        hash_val = _vec_hash_32bit(key_vec, D=D_32)
    store_val = (hash_val & TAG_BIT_MASK) | idx

    # Linear-probe insertion loop. Inactive lanes feed an expected value
    # of ``-2`` to ``tl.atomic_cas``; that value can never match the empty
    # sentinel (``-1``) nor any valid stored value (``>= 0``), so the CAS
    # is a memory-level no-op for them.
    to_be_inserted = mask
    target_slot = hash_val & SLOT_BIT_MASK
    probes = 0
    while _reduce_any(to_be_inserted) & (probes < hashmap_size):
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(to_be_inserted, -1, -2), store_val)
        inserted = to_be_inserted & (prev == -1)
        to_be_inserted = to_be_inserted & ~inserted

        # Unconditional advance is fine: already-inserted lanes will just
        # do a no-op CAS on the next slot.
        target_slot = (target_slot + 1) & SLOT_BIT_MASK
        probes += 1
    # Defensive guard: the high-level API sizes the table for load factor
    # 0.3 so this should be unreachable.
    tl.device_assert(tl.max(to_be_inserted) == 0, "hashmap_build: hashmap full -- caller mis-sized the table")


@triton.jit
def _hashmap_lookup_inline_32bit(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    keys_ptr: tl.const,
    query_vec: tl.tensor,   
    mask: tl.tensor,
    D: tl.constexpr
):
    """Lookup the query_vec in the hash map and return the found index or -1 if not found.
    NOTE: keys_ptr must be 4-byte aligned and D must be divisible by 4.
    """
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32, const=True))
    query_vec_32 = _vec_pack_little_endian_to_int32(query_vec)
    D_32: tl.constexpr = D * query_vec.dtype.itemsize // 4
    tl.static_assert(D_32 == query_vec_32.shape[-1], "Invalid query_vec shape after packing to int32. Check D and input dtype.")

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    if D_32 == 1:
        query = tl.reshape(query_vec_32, *query_vec_32.shape[:-1])
        hash_val = _scalar_hash_32bit(query)
    else:
        query = query_vec_32
        hash_val = _vec_hash_32bit(query_vec_32, D=D_32)
    query_tag = hash_val & TAG_BIT_MASK

    is_active = tl.broadcast_to(mask, hash_val.shape)
    found_idx = tl.full(hash_val.shape, -1, tl.int32)

    # Probing loop. The N-probe bound below is enough to guarantee that an
    # existing key is found; if the map is full and the key is absent we
    # would otherwise spin forever. A single scalar counter suffices because
    # every lane in this program runs the same number of iterations.
    curr_slot = hash_val & SLOT_BIT_MASK
    probes = 0
    while _reduce_any(is_active) & (probes < hashmap_size):
        # Compute current slot to probe
        stored_val = tl.load(hashmap_ptr + curr_slot, mask=is_active, other=-1)

        # Drop queries that hit empty slots
        is_active &= (stored_val >= 0)
        
        # Extract stored index & tag
        stored_idx = stored_val & SLOT_BIT_MASK
        stored_tag = stored_val & TAG_BIT_MASK
        # First compare tags
        is_match = is_active & (stored_tag == query_tag)
        # Then load and compare full keys
        if D_32 == 1:
            stored_key = tl.load(keys_ptr_32 + stored_idx, mask=is_match, other=0)
            is_match &= (stored_key == query)
        else:
            key_vec = _vec_load(keys_ptr_32 + stored_idx * D_32, mask=is_match, D=D_32)
            is_match &= _reduce_all(key_vec == query, axis=-1)

        # Update found indices
        success = is_match & is_active
        found_idx = tl.where(success, stored_idx, found_idx)
        is_active &= ~success
        
        # Update current slot
        curr_slot = (curr_slot + 1) & SLOT_BIT_MASK
        probes += 1
    # Sanity check: any lane that is still active after ``hashmap_size``
    # probes means the table is full and we cannot conclusively decide
    # whether the key is present. Trip a device-side assert so the host
    # sees an error instead of a silently mis-returned -1.
    tl.device_assert(tl.max(is_active) == 0, "hashmap_lookup: hashmap full -- caller mis-sized the table")
    return found_idx
    




@triton.jit
def _hashmap_unique_kernel_32bit(
    hashmap_ptr: tl.pointer_type,
    hashmap_size: int,
    keys_ptr: tl.const,
    results_ptr: tl.pointer_type,
    is_canonical_ptr: tl.pointer_type,
    n_keys: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused build + self-lookup kernel for unique.

    For each key, probe the hashmap. On CAS success the key is inserted and
    its own index is the canonical one. On CAS failure, compare the existing
    slot's stored key against ours (tag first, then full key); if it matches
    record the existing key's index, otherwise advance to the next slot.

    Also writes a per-key boolean (uint8) into ``is_canonical_ptr`` indicating
    whether this lane's key is the canonical (first inserted) occurrence,
    which saves a separate comparison kernel on the host side.
    """
    tl.static_assert(D * keys_ptr.dtype.element_ty.itemsize % 4 == 0, "keys byte width must be divisible by 4")
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32, const=True))
    D_32: tl.constexpr = D * keys_ptr.dtype.element_ty.itemsize // 4

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_keys

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    # Compute hash and per-lane stored value.
    if D_32 == 1:
        # Scalar fast path -- mirrors the build/lookup kernels.
        key_vec = tl.load(keys_ptr_32 + idx, mask=mask, other=0)
        hash_val = _scalar_hash_32bit(key_vec)
    else:
        key_vec = _vec_load(keys_ptr_32 + idx * D_32, mask=mask, D=D_32)
        hash_val = _vec_hash_32bit(key_vec, D=D_32)
    my_tag = hash_val & TAG_BIT_MASK
    store_val = my_tag | idx

    found_idx = idx
    active = mask
    target_slot = hash_val & SLOT_BIT_MASK
    probes = 0
    while _reduce_any(active) & (probes < hashmap_size):
        # Inactive lanes feed ``expected = -2`` so their CAS is a memory-level
        # no-op.
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(active, -1, -2), store_val)

        # CAS succeeded (prev == -1): our key now owns this slot and the
        # pre-initialized ``found_idx = idx`` is already the canonical one.
        # CAS failed (prev >= 0): keep this lane active so we can check
        # whether the existing entry is a duplicate of ours.
        active &= (prev >= 0)

        # Compare the existing entry against our key (tag first, then full key).
        prev_tag = prev & TAG_BIT_MASK
        prev_idx = prev & SLOT_BIT_MASK
        tag_match = active & (prev_tag == my_tag)
        if D_32 == 1:
            existing_key = tl.load(keys_ptr_32 + prev_idx, mask=tag_match, other=0)
            full_match = tag_match & (existing_key == key_vec)
        else:
            existing_key = _vec_load(keys_ptr_32 + prev_idx * D_32, mask=tag_match, D=D_32)
            full_match = tag_match & _reduce_all(existing_key == key_vec, axis=-1)
        found_idx = tl.where(full_match, prev_idx, found_idx)
        active &= ~full_match

        # Linear-probe advance. Unconditional ``+ 1`` is fine: settled lanes
        # will just do a no-op CAS on the next slot.
        target_slot = (target_slot + 1) & SLOT_BIT_MASK
        probes += 1

    # Sanity check: every key must have either claimed a slot or matched an
    # existing duplicate. Surfaces a device-side assert if a future caller
    # bypasses the wrapper and supplies an undersized table.
    tl.device_assert(tl.max(active) == 0, "hashmap_unique: hashmap full -- caller mis-sized the table")

    tl.store(results_ptr + idx, found_idx, mask=mask)
    tl.store(is_canonical_ptr + idx, (found_idx == idx).to(tl.int8), mask=mask)


def hashmap_build(keys: Tensor) -> Tensor:
    """
    Build a hash map from the given keys using Triton.
    
    Args:
        keys (Tensor): A tensor of shape `(n_keys, D)` representing the keys.

    Returns:
        Tensor: A 1D tensor representing the hash map.

    Notes
    -----
        The hash map stores a combination of a hash tag and the index of each key.
        See `hashmap_lookup` for querying the hash map.
        Use `hashmap_build_lookup` for a combined build and lookup operation.
    """
    # Determine hash map size
    n_keys = keys.shape[0]
    hashmap_size = triton.next_power_of_2(int(n_keys / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), "Hash map size exceeds 2^30, which is the limit for our 32-bit implementation."

    # Pad keys to a byte width that is a power of two in int32 words.
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes)
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)

    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE), )
    
    _hashmap_build_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return hashmap






def hashmap_unique(
    keys: Tensor, 
    return_index: bool = False,
    return_inverse: bool = False, 
) -> Tensor | tuple[Tensor, ...]:
    """
    Hashmap-based unique operation to find unique keys and optionally return inverse indices.

    NOTE: this function is like `torch.unique` but much faster at the cost of non-deterministic order of the unique keys. 
    The result order is not even consistent for the same input due to the race condition in hashmap.
    NOTE: all returned indices are int32
    
    Args:
        keys (Tensor): A tensor of shape `(n_keys, *key_dims)` representing the keys.
        return_inverse (bool): Whether to return the inverse indices.
        return_counts (bool): Whether to return the counts of each unique key.

    Returns:
        unique_keys (Tensor): A tensor of shape `(n_unique_keys, *key_dims)`
        unique_index (Tensor, optional): A tensor of shape `(n_unique_keys,)` containing the index of one occurrence of each unique key in the original keys. Only returned if `return_index` is True.
        unique_inverse (Tensor, optional): A tensor of shape `(n_keys,)` containing the indices of the original keys in the unique keys. Only returned if `return_inverse` is True.
    """
    # Fused build + self-lookup: each key probes the hashmap; on collision
    # we compare keys instead of skipping, so duplicates resolve to a single
    # canonical index in O(1) probes regardless of duplicate count.
    n_keys = keys.shape[0]
    if n_keys == 0:
        empty_idx = torch.empty((0,), dtype=torch.int64, device=keys.device)
        unique_keys = keys
        returns = (unique_keys,)
        if return_index:
            returns += (empty_idx,)
        if return_inverse:
            returns += (empty_idx,)
        if len(returns) == 1:
            return returns[0]
        return returns

    hashmap_size = triton.next_power_of_2(int(n_keys / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), "Hash map size exceeds 2^30, which is the limit for our 32-bit implementation."

    keys_bytes = keys.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys_bytes.shape[1], 4))
    keys_i32 = pad_to_size_along_dim(keys_bytes, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)
    indices = torch.empty((n_keys,), dtype=torch.int32, device=keys.device)
    is_canonical = torch.empty((n_keys,), dtype=torch.bool, device=keys.device)

    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE),)
    _hashmap_unique_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        results_ptr=indices,
        is_canonical_ptr=is_canonical,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    unique_indices = is_canonical.nonzero(as_tuple=True)[0].to(torch.int32)
    num_uniques = unique_indices.shape[0]
    unique_keys = keys[unique_indices]

    returns = (unique_keys,)

    if return_index:
        returns += (unique_indices,)

    if return_inverse:
        unique_inverse = torch.empty(n_keys, dtype=torch.int32, device=keys.device)
        unique_inverse[unique_indices] = torch.arange(num_uniques, device=keys.device, dtype=torch.int32)
        unique_inverse = unique_inverse[indices]
        returns += (unique_inverse,)

    if len(returns) == 1:
        return returns[0]
    return returns
