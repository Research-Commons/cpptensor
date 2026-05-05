# Linear Algebra Decomposition Notes

This project's LAPACK-backed decomposition paths now document and test the
following semantics:

## SVD

- Backend: `LAPACKE_sgesdd` (divide-and-conquer SVD).
- Ordering: singular values are returned in descending order.
- Validation target: representative tests check
  `A ≈ U * diag(S) * Vt` and orthogonality of `U`/`Vt`.
- Failure semantics:
  - negative `info` → illegal LAPACK argument
  - positive `info` → divide-and-conquer bidiagonal solver did not converge

## Symmetric eigen decomposition

- Backend: `LAPACKE_ssyevd` (divide-and-conquer symmetric eigensolver).
- Ordering: eigenvalues are returned in ascending order.
- Validation target: tests check `A * v ≈ λ * v` and orthogonality of the
  eigenvector basis.
- Failure semantics:
  - negative `info` → illegal LAPACK argument
  - positive `info` → off-diagonal reduction failed to converge

## General eigen decomposition

- Backend: `LAPACKE_sgeev`.
- Ordering: no additional sorting is applied; the order is whatever LAPACK
  returns.
- Complex pairs: conjugate eigenvalue pairs remain adjacent and use LAPACK's
  packed real-eigenvector storage.
- Validation target: tests decode packed vectors and check `A * v ≈ λ * v`
  for real and complex eigenpairs.
- Failure semantics:
  - negative `info` → illegal LAPACK argument
  - positive `info` → QR iteration failed to compute all eigenvalues

## Layout and CPU prerequisites

- All decomposition entry points currently require `DeviceType::CPU`.
- Inputs are interpreted in their logical tensor order.
- Non-contiguous CPU views (for example `transpose()` or sliced 2D views) are
  materialized into an internal contiguous row-major workspace before LAPACK is
  called.
