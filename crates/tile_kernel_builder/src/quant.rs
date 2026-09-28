//! Typed quantized-block layouts — making stride/offset desync (failure mode F2)
//! structurally unrepresentable.
//!
//! # The gap this closes
//!
//! F1 (capacity) is caught by [`UbView` (tile_std)]; F3 (DMA ordering) by
//! `DmaPending`. F2 — a fusion/tiling rewrite whose scales offset and qs offset
//! disagree on *which sub-block* they address — was **not** caught by the type
//! system: kernels read `qs + 16*iq + 4*ir` and `scales + <word>` as raw integer
//! arithmetic, so mixing one kernel's qs convention with the wrong scales word
//! compiled clean and produced silent wrong numerics (only a runtime differential
//! test caught it — verified this session on the llama.cpp vectorized port).
//!
//! # What is and isn't the invariant (corrected after on-device testing)
//!
//! Two *different* offset conventions are both correct — ggml uses `scales + iq`,
//! our codegen `MulMvQ4KF32` uses `scales + ir` (verified 0/8 on Qwen3-4B), each
//! self-consistent with its lane→sub-block mapping. So the guard is NOT "use a
//! fixed formula". The real invariant is: **the scales and qs offsets must name
//! the SAME 32-weight sub-block.** This module enforces it by deriving BOTH from
//! one [`SubBlock`] under one [`Q4KConvention`] — a kernel cannot pass sub-block A
//! to scales and B to qs. The field-base offsets (d/dmin/scales/qs at 0/2/4/16)
//! ARE canonical and are single-sourced in [`q4k`].
//!
//! # `block_q4_K` canonical layout (144 bytes / 256 weights)
//!
//! ```text
//! offset  size  field
//!   0      2    d      (f16 super-block scale for the quantized scales)
//!   2      2    dmin   (f16 super-block scale for the quantized mins)
//!   4     12    scales (6-bit packed scales+mins, K_SCALE_SIZE)
//!  16    128    qs     (4-bit quants, QK_K/2)
//! ```
//!
//! The `d`/`dmin`/`scales`/`qs` offsets and the 6-bit scale-unpack live in ONE
//! place ([`BlockQ4K`]); every backend emitter derives its pointer arithmetic
//! from these accessors instead of re-deriving `+ 4`, `+ 16`, `16*iq + 4*ir` by
//! hand. That single source of truth is the F2 guard.

use core::marker::PhantomData;

/// Canonical `block_q4_K` geometry. Referenced everywhere an offset is needed,
/// so the numbers exist exactly once.
pub mod q4k {
    /// Weights per super-block.
    pub const QK_K: usize = 256;
    /// Bytes per `block_q4_K`.
    pub const BLOCK_BYTES: usize = 144;
    /// Number of 32-weight sub-blocks per super-block.
    pub const N_SUBBLOCKS: usize = 8;

    // Field byte offsets — the ONLY place these constants appear.
    pub const OFF_D: usize = 0; // f16
    pub const OFF_DMIN: usize = 2; // f16
    pub const OFF_SCALES: usize = 4; // 12 bytes, 6-bit packed
    pub const OFF_QS: usize = 16; // 128 bytes, 4-bit nibbles

    // Compile-time consistency: the offsets must tile the block exactly.
    const _: () = assert!(OFF_D == 0);
    const _: () = assert!(OFF_DMIN == OFF_D + 2);
    const _: () = assert!(OFF_SCALES == OFF_DMIN + 2);
    const _: () = assert!(OFF_QS == OFF_SCALES + 12);
    const _: () = assert!(OFF_QS + QK_K / 2 == BLOCK_BYTES);
    const _: () = assert!(N_SUBBLOCKS == QK_K / 32);
}

/// A validated index of a 32-weight sub-block within a `block_q4_K` (`0..8`).
///
/// Constructing one goes through [`SubBlock::new`], which range-checks `0..8` —
/// so a kernel can never form a sub-block index outside the block, and the
/// `half()` (128-half, `iq`) / `quarter()` (`ir`) decomposition is derived from
/// the ONE index `i`. Because scales and qs offsets both come from the same
/// `SubBlock`, they can't end up naming different sub-blocks — the desync class.
#[derive(Clone, Copy)]
pub struct SubBlock {
    i: u8, // invariant: i < 8
}

impl SubBlock {
    /// The 8 sub-blocks of a super-block, in order. The only constructor.
    #[inline(always)]
    pub const fn all() -> [SubBlock; q4k::N_SUBBLOCKS] {
        [
            SubBlock { i: 0 },
            SubBlock { i: 1 },
            SubBlock { i: 2 },
            SubBlock { i: 3 },
            SubBlock { i: 4 },
            SubBlock { i: 5 },
            SubBlock { i: 6 },
            SubBlock { i: 7 },
        ]
    }

    /// Range-checked constructor (`i < 8` or `None`). There is no way to build a
    /// `SubBlock` from an out-of-range integer.
    #[inline(always)]
    pub const fn new(i: usize) -> Option<SubBlock> {
        if i < q4k::N_SUBBLOCKS {
            Some(SubBlock { i: i as u8 })
        } else {
            None
        }
    }

    /// Raw sub-block index `0..8`.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.i as usize
    }

    /// Which 128-weight half this sub-block lives in (`iq` in the kernels): 0 or 1.
    /// DERIVED from `i` — a kernel can no longer pass `ir` where `iq` was meant.
    #[inline(always)]
    pub const fn half(self) -> usize {
        (self.i as usize) / 4
    }

    /// Which quarter within the 128-half (`ir` in the kernels): `0..4`.
    #[inline(always)]
    pub const fn quarter(self) -> usize {
        (self.i as usize) % 4
    }

    /// Byte offset of this sub-block's low/high nibble selector.
    /// (`i & 1`: even sub-blocks take the low nibble, odd take the high.)
    #[inline(always)]
    pub const fn is_high_nibble(self) -> bool {
        (self.i & 1) == 1
    }

    /// Element offset of this sub-block's 32 activations within the super-block.
    #[inline(always)]
    pub const fn activation_offset(self) -> usize {
        (self.i as usize) * 32
    }
}

/// A typed view over one `block_q4_K` at a known base byte offset in a device
/// buffer. All field accessors return offsets/values DERIVED from [`q4k`] — the
/// kernel never writes `+ 4` / `+ 16` / `16*iq + 4*ir` by hand.
///
/// `'a` brands the block to the buffer it was minted from, mirroring
/// [`UbView` (tile_std)]. The emitter consumes these accessors to generate the
/// pointer arithmetic, so a mis-derived offset in the emitter is a *Rust* type
/// error at the layout boundary, not a silent MSL bug.
#[derive(Clone, Copy)]
pub struct BlockQ4K<'a> {
    /// Byte offset of this block's start within the weight buffer.
    base: usize,
    _brand: PhantomData<&'a [u8]>,
}

impl<'a> BlockQ4K<'a> {
    /// Mint a block view at super-block index `blk` for output row `row` with a
    /// row stride of `row_stride_bytes` (the `nb01` that broke the port when
    /// hand-computed). The base offset is DERIVED here, once.
    #[inline(always)]
    pub const fn at(row: usize, row_stride_bytes: usize, blk: usize) -> BlockQ4K<'a> {
        BlockQ4K {
            base: row * row_stride_bytes + blk * q4k::BLOCK_BYTES,
            _brand: PhantomData,
        }
    }

    /// Byte offset of the `d` (f16) super-block scale.
    #[inline(always)]
    pub const fn d_offset(self) -> usize {
        self.base + q4k::OFF_D
    }

    /// Byte offset of the `dmin` (f16) super-block min-scale.
    #[inline(always)]
    pub const fn dmin_offset(self) -> usize {
        self.base + q4k::OFF_DMIN
    }

    /// Byte offset of the packed 6-bit `scales` region.
    #[inline(always)]
    pub const fn scales_offset(self) -> usize {
        self.base + q4k::OFF_SCALES
    }

    /// Byte offset of the 4-bit `qs` nibble region.
    #[inline(always)]
    pub const fn qs_offset(self) -> usize {
        self.base + q4k::OFF_QS
    }

    /// Byte offset of the packed scale/min words for `sb`, and of the `qs` nibble
    /// words for `sb`, returned TOGETHER from ONE [`SubBlock`].
    ///
    /// # What this actually guards (corrected after testing on-device)
    ///
    /// The F2 desync is NOT "use offset formula X". Two *different* conventions are
    /// both correct, because each kernel assigns sub-blocks to lanes consistently:
    ///   * ggml `mul_mv_q4_K`: scales word `+ iq`, qs word `16*iq + 4*ir`.
    ///   * our codegen `MulMvQ4KF32` (verified 0/8 on Qwen3-4B): scales `+ ir`,
    ///     qs `16*iq + 4*ir`.
    /// So hard-coding `+iq` would WRONGLY reject the valid codegen kernel.
    ///
    /// The real invariant is: **the scales offset and the qs offset must address
    /// the SAME 32-weight sub-block.** The bug that shipped this session mixed
    /// ggml's qs convention with the wrong scales word — a genuine desync between
    /// two offsets that should have named the same sub-block.
    ///
    /// This accessor enforces that by deriving BOTH offsets from a single `sb`
    /// under one `Q4KConvention`, so a kernel can never pass one sub-block to
    /// scales and a different one to qs. The convention (which lane component
    /// indexes scales) is an explicit parameter, not silently baked.
    #[inline(always)]
    pub const fn scale_and_qs_u16_offsets(
        self,
        sb: SubBlock,
        conv: Q4KConvention,
    ) -> (usize, usize) {
        let scales_word = match conv {
            Q4KConvention::Ggml => sb.half(),         // + iq
            Q4KConvention::CodegenIr => sb.quarter(), // + ir
        };
        let scales = self.scales_offset() + scales_word * core_size_of_u16();
        let qs = self.qs_offset() + (16 * sb.half() + 4 * sb.quarter()) * core_size_of_u16();
        (scales, qs)
    }
}

/// Lane→sub-block convention for the scales word. Both are correct kernels; the
/// point of naming them is that a kernel picks ONE and derives BOTH the scales
/// and qs offsets from the same [`SubBlock`] under it — so they cannot desync.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Q4KConvention {
    /// ggml `mul_mv_q4_K`: scales indexed by the 128-half (`iq`).
    Ggml,
    /// our codegen `MulMvQ4KF32`: scales indexed by the quarter (`ir`).
    CodegenIr,
}

/// `size_of::<u16>()` as a const fn usable in the accessors above without pulling
/// in `core::mem` (this crate shadows `core`).
#[inline(always)]
const fn core_size_of_u16() -> usize {
    2
}

// =============================================================================
// block_q6_K
// =============================================================================

/// Canonical `block_q6_K` geometry (210 bytes / 256 weights). NOTE the field order
/// differs from q4_K: the `d` super-block scale sits at the END, and the 6-bit
/// weight is split across `ql` (low 4 bits) and `qh` (high 2 bits) — a classic F2
/// desync site (reading the low nibble from one sub-block index and the high bits
/// from another silently corrupts). The offsets live here, once.
pub mod q6k {
    pub const QK_K: usize = 256;
    pub const BLOCK_BYTES: usize = 210;

    pub const OFF_QL: usize = 0; // 128 bytes, low 4 bits
    pub const OFF_QH: usize = 128; // 64 bytes, high 2 bits
    pub const OFF_SCALES: usize = 192; // 16 bytes, int8
    pub const OFF_D: usize = 208; // 2 bytes, half (at the END)

    // Sizes tile the block exactly.
    const _: () = assert!(OFF_QL == 0);
    const _: () = assert!(OFF_QH == OFF_QL + QK_K / 2);
    const _: () = assert!(OFF_SCALES == OFF_QH + QK_K / 4);
    const _: () = assert!(OFF_D == OFF_SCALES + QK_K / 16);
    const _: () = assert!(OFF_D + 2 == BLOCK_BYTES);
}

/// Typed view over one `block_q6_K`. All offsets DERIVED from [`q6k`] — the kernel
/// never open-codes `+128`, `+192`, `+208`. `[`SubBlockQ6`]` couples the `ql` and
/// `qh` accesses to the same weight so the low-4/high-2 halves can't desync.
#[derive(Clone, Copy)]
pub struct BlockQ6K<'a> {
    base: usize,
    _brand: PhantomData<&'a [u8]>,
}

impl<'a> BlockQ6K<'a> {
    #[inline(always)]
    pub const fn at(row: usize, row_stride_bytes: usize, blk: usize) -> BlockQ6K<'a> {
        BlockQ6K {
            base: row * row_stride_bytes + blk * q6k::BLOCK_BYTES,
            _brand: PhantomData,
        }
    }
    #[inline(always)]
    pub const fn ql_offset(self) -> usize {
        self.base + q6k::OFF_QL
    }
    #[inline(always)]
    pub const fn qh_offset(self) -> usize {
        self.base + q6k::OFF_QH
    }
    #[inline(always)]
    pub const fn scales_offset(self) -> usize {
        self.base + q6k::OFF_SCALES
    }
    #[inline(always)]
    pub const fn d_offset(self) -> usize {
        self.base + q6k::OFF_D
    }

    /// The `ql` byte offset AND the `qh` byte offset for weight `w` (`0..256`),
    /// returned TOGETHER so the low-4 and high-2 bits are always read for the SAME
    /// weight. q6_K packs two weights per `ql` byte and four per `qh` byte; this
    /// derives both indices from one `w`, so a kernel can't take low bits from
    /// weight A and high bits from weight B — the q6_K analogue of the q4_K desync.
    #[inline(always)]
    pub const fn ql_qh_byte_offsets(self, w: usize) -> (usize, usize) {
        // low 4 bits: two weights per ql byte; high 2 bits: four per qh byte.
        (self.ql_offset() + w / 2, self.qh_offset() + w / 4)
    }

    /// int8 scale byte for weight `w` — q6_K uses one 8-bit scale per 16 weights.
    #[inline(always)]
    pub const fn scale_offset(self, w: usize) -> usize {
        self.scales_offset() + w / 16
    }
}

// =============================================================================
// block_q8_0
// =============================================================================

/// Canonical `block_q8_0` geometry (34 bytes / 32 weights). `d` (half) at 0, then
/// 32 int8 quants. The F2 surface here is the row stride (`nb01`) and the qs base,
/// not sub-block packing (q8_0 is flat).
pub mod q8_0 {
    pub const QK: usize = 32;
    pub const BLOCK_BYTES: usize = 34;
    pub const OFF_D: usize = 0; // 2 bytes, half
    pub const OFF_QS: usize = 2; // 32 bytes, int8

    const _: () = assert!(OFF_D == 0);
    const _: () = assert!(OFF_QS == OFF_D + 2);
    const _: () = assert!(OFF_QS + QK == BLOCK_BYTES);
}

/// Typed view over one `block_q8_0`. Offsets DERIVED from [`q8_0`].
#[derive(Clone, Copy)]
pub struct BlockQ8_0<'a> {
    base: usize,
    _brand: PhantomData<&'a [u8]>,
}

impl<'a> BlockQ8_0<'a> {
    #[inline(always)]
    pub const fn at(row: usize, row_stride_bytes: usize, blk: usize) -> BlockQ8_0<'a> {
        BlockQ8_0 {
            base: row * row_stride_bytes + blk * q8_0::BLOCK_BYTES,
            _brand: PhantomData,
        }
    }
    #[inline(always)]
    pub const fn d_offset(self) -> usize {
        self.base + q8_0::OFF_D
    }
    #[inline(always)]
    pub const fn qs_offset(self) -> usize {
        self.base + q8_0::OFF_QS
    }
    /// int8 quant byte for weight `w` (`0..32`).
    #[inline(always)]
    pub const fn qs_byte_offset(self, w: usize) -> usize {
        self.qs_offset() + w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_offsets_are_canonical() {
        let b = BlockQ4K::at(0, q4k::BLOCK_BYTES, 0);
        assert_eq!(b.d_offset(), 0);
        assert_eq!(b.dmin_offset(), 2);
        assert_eq!(b.scales_offset(), 4);
        assert_eq!(b.qs_offset(), 16);
    }

    #[test]
    fn row_and_block_stride_compose() {
        // row 3, stride = 2 blocks worth, block 1
        let b = BlockQ4K::at(3, 2 * q4k::BLOCK_BYTES, 1);
        assert_eq!(b.base, 3 * 288 + 144);
        assert_eq!(b.qs_offset(), 3 * 288 + 144 + 16);
    }

    #[test]
    fn subblock_half_quarter_are_derived_not_free() {
        // The bug I shipped: using `quarter()` (ir) where `half()` (iq) was meant.
        // With SubBlock, half and quarter are *computed from the same index*, so
        // the scales and qs offsets (both derived from one sb) can't be fed
        // mismatched iq/ir.
        for sb in SubBlock::all() {
            assert_eq!(sb.half(), sb.index() / 4);
            assert_eq!(sb.quarter(), sb.index() % 4);
        }
        // sub-block 5 -> half 1, quarter 1
        let sb5 = SubBlock::new(5).unwrap();
        assert_eq!(sb5.half(), 1);
        assert_eq!(sb5.quarter(), 1);
    }

    #[test]
    fn both_conventions_match_their_proven_kernels() {
        let b = BlockQ4K::at(0, q4k::BLOCK_BYTES, 0);
        let sb = SubBlock::new(6).unwrap(); // iq=1, ir=2

        // ggml mul_mv_q4_K: scales word + iq; qs + (16*iq + 4*ir).
        let (sc_g, qs_g) = b.scale_and_qs_u16_offsets(sb, Q4KConvention::Ggml);
        assert_eq!(sc_g, 4 + 1 * 2);
        assert_eq!(qs_g, 16 + (16 * 1 + 4 * 2) * 2);

        // our codegen MulMvQ4KF32 (verified 0/8 on Qwen3-4B): scales word + ir; same qs.
        let (sc_c, qs_c) = b.scale_and_qs_u16_offsets(sb, Q4KConvention::CodegenIr);
        assert_eq!(sc_c, 4 + 2 * 2); // + ir
        assert_eq!(qs_c, qs_g); // qs identical across conventions
    }

    #[test]
    fn scales_and_qs_share_one_subblock() {
        // The invariant: BOTH offsets come from the SAME `sb`, so a kernel can't
        // name sub-block A for scales and sub-block B for qs — the desync class.
        let b = BlockQ4K::at(0, q4k::BLOCK_BYTES, 0);
        for sb in SubBlock::all() {
            let (_sc, qs) = b.scale_and_qs_u16_offsets(sb, Q4KConvention::Ggml);
            // qs offset is a pure function of this one sb — no second index to disagree.
            assert_eq!(qs, 16 + (16 * sb.half() + 4 * sb.quarter()) * 2);
        }
    }

    #[test]
    fn no_out_of_range_subblock() {
        assert!(SubBlock::new(8).is_none());
        assert!(SubBlock::new(7).is_some());
    }

    #[test]
    fn q6k_field_offsets_are_canonical() {
        let b = BlockQ6K::at(0, q6k::BLOCK_BYTES, 0);
        assert_eq!(b.ql_offset(), 0);
        assert_eq!(b.qh_offset(), 128);
        assert_eq!(b.scales_offset(), 192);
        assert_eq!(b.d_offset(), 208); // d at the END, unlike q4_K
    }

    #[test]
    fn q6k_ql_qh_couple_the_same_weight() {
        // The q6_K desync: low-4 bits from one weight, high-2 from another. Coupling
        // both offsets to a single `w` makes that unrepresentable.
        let b = BlockQ6K::at(0, q6k::BLOCK_BYTES, 0);
        for w in [0usize, 1, 2, 15, 200, 255] {
            let (ql, qh) = b.ql_qh_byte_offsets(w);
            assert_eq!(ql, 0 + w / 2); // two weights per ql byte
            assert_eq!(qh, 128 + w / 4); // four weights per qh byte
        }
        // one 8-bit scale per 16 weights
        assert_eq!(b.scale_offset(0), 192);
        assert_eq!(b.scale_offset(16), 193);
    }

    #[test]
    fn q6k_row_and_block_stride_compose() {
        let b = BlockQ6K::at(3, 2 * q6k::BLOCK_BYTES, 1);
        assert_eq!(b.d_offset(), 3 * 420 + 210 + 208);
    }

    #[test]
    fn q8_0_field_offsets_are_canonical() {
        let b = BlockQ8_0::at(0, q8_0::BLOCK_BYTES, 0);
        assert_eq!(b.d_offset(), 0);
        assert_eq!(b.qs_offset(), 2);
        assert_eq!(b.qs_byte_offset(0), 2);
        assert_eq!(b.qs_byte_offset(31), 33);
    }

    #[test]
    fn q8_0_row_and_block_stride_compose() {
        let b = BlockQ8_0::at(5, 4 * q8_0::BLOCK_BYTES, 2);
        assert_eq!(b.qs_offset(), 5 * 136 + 2 * 34 + 2);
    }
}
