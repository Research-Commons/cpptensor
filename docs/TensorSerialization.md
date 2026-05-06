# Tensor serialization and checkpoint I/O

cpptensor now exposes a versioned binary checkpoint API:

- `Tensor::save(const std::string& path) const`
- `Tensor::load(const std::string& path)`

## Format (v1)

Each checkpoint stores:

1. Magic bytes (`CPTENSR\0`)
2. Format version (`uint16`, currently `1`)
3. Reserved flags (`uint16`, currently `0`)
4. Dtype code (`uint8`, currently `1` = `float32`)
5. Device code (`uint8`, currently `0` = CPU, `1` = CUDA)
6. Reserved padding (`uint16`)
7. Rank (`uint64`)
8. Numel (`uint64`)
9. Shape dims (`rank × uint64`)
10. Tensor values (`numel × float32`, little-endian row-major)

## View behavior

View tensors (slice/transpose/permute/non-contiguous layouts) are **materialized**
on save using their logical row-major values.

- Saved values always represent the tensor’s logical contents.
- Loaded tensors are contiguous.
- Shape and device metadata are preserved.

## Error handling

`Tensor::load()` throws `std::runtime_error` when:

- the file is not a cpptensor checkpoint
- checkpoint version is unsupported
- dtype/device metadata is unsupported
- shape/numel metadata is inconsistent
- the file is truncated or has unexpected trailing bytes
