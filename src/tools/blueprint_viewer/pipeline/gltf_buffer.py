"""Low-level glTF binary buffer / accessor packing helper.

Wraps the bookkeeping pygltflib leaves to the caller: building the single embedded
GLB binary blob, creating BufferView + Accessor entries that point into it at the
correct byte offsets, and -- critically -- keeping every accessor's byteOffset
4-byte aligned (the glTF 2.0 spec requires bufferView.byteOffset AND accessor
byteOffset to be a multiple of the component size, and this pipeline's contract
explicitly calls out "4-byte accessor alignment").

One ``BufferPacker`` instance accumulates all accessors for the whole file (mesh
attributes AND animation samplers alike), then ``finalize()`` returns the single
concatenated ``bytes`` blob plus the list of ``BufferView``/``Accessor`` objects
already appended to the target ``GLTF2`` document.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pygltflib as gltf

_COMPONENT_SIZE = {
    gltf.BYTE: 1, gltf.UNSIGNED_BYTE: 1,
    gltf.SHORT: 2, gltf.UNSIGNED_SHORT: 2,
    gltf.UNSIGNED_INT: 4, gltf.FLOAT: 4,
}
_TYPE_COMPONENT_COUNT = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16,
}


def _pad_to_4(data: bytearray) -> None:
    while len(data) % 4 != 0:
        data.append(0)


@dataclass
class BufferPacker:
    document: gltf.GLTF2
    _blob: bytearray = field(default_factory=bytearray)

    def _append_bufferview(self, raw: bytes, target: Optional[int] = None) -> int:
        _pad_to_4(self._blob)  # keep every bufferView 4-byte aligned, not just accessors
        offset = len(self._blob)
        self._blob.extend(raw)
        bv = gltf.BufferView(buffer=0, byteOffset=offset, byteLength=len(raw), target=target)
        self.document.bufferViews.append(bv)
        return len(self.document.bufferViews) - 1

    def add_accessor(
        self,
        values: Sequence[Sequence[float]] | Sequence[float],
        component_type: int,
        accessor_type: str,
        *,
        target: Optional[int] = None,
        normalized: bool = False,
        compute_minmax: bool = True,
    ) -> int:
        """Pack ``values`` (a flat list of scalars, or a list of tuples for
        VEC2/VEC3/VEC4) into a new bufferView + accessor. Returns the accessor index.

        NaN/Inf guard: raises ValueError if any packed float is non-finite -- the
        contract requires "min/max sane (no NaN)", so this is enforced at the point of
        creation rather than left to the downstream validator to catch.
        """
        n_comp = _TYPE_COMPONENT_COUNT[accessor_type]
        is_float = component_type == gltf.FLOAT
        fmt_char = {gltf.FLOAT: "f", gltf.UNSIGNED_SHORT: "H", gltf.UNSIGNED_INT: "I",
                    gltf.UNSIGNED_BYTE: "B", gltf.SHORT: "h", gltf.BYTE: "b"}[component_type]

        flat: List[float] = []
        if n_comp == 1:
            flat = [float(v) if is_float else int(v) for v in values]  # type: ignore[arg-type]
            count = len(flat)
        else:
            count = len(values)
            for tup in values:  # type: ignore[assignment]
                if len(tup) != n_comp:
                    raise ValueError(f"expected {n_comp}-tuples for {accessor_type}, got {tup!r}")
                flat.extend(float(x) if is_float else int(x) for x in tup)

        if is_float:
            for v in flat:
                if v != v or v in (float("inf"), float("-inf")):  # NaN check (v!=v) + Inf
                    raise ValueError(f"non-finite value {v} in accessor data (type={accessor_type})")

        raw = struct.pack(f"<{len(flat)}{fmt_char}", *flat)
        bv_index = self._append_bufferview(raw, target=target)

        acc = gltf.Accessor(
            bufferView=bv_index, byteOffset=0, componentType=component_type,
            count=count, type=accessor_type, normalized=normalized,
        )
        if compute_minmax and count > 0:
            if n_comp == 1:
                acc.min = [min(flat)]
                acc.max = [max(flat)]
            else:
                cols = [flat[i::n_comp] for i in range(n_comp)]
                acc.min = [min(c) for c in cols]
                acc.max = [max(c) for c in cols]
        self.document.accessors.append(acc)
        return len(self.document.accessors) - 1

    def finalize(self) -> bytes:
        """Pad the blob to 4 bytes and return it. Call once, after all accessors have
        been added. Also wires ``document.buffers[0].byteLength``."""
        _pad_to_4(self._blob)
        return bytes(self._blob)
