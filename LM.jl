using Parameters
using LinearAlgebra
using BlockArrays

# =============================================================================
# Single-point local Chern marker (PBC, Γ-point only) in the LCPAO basis
#
#   N. Baù and A. Marrazzo, Phys. Rev. B 109, 014206 (2024)
#     Eq. (3)   C        = -1/π  Im Σ_n <ũ_n(b1)|ũ_n(b2)>
#     Eq. (4)   |ũ_n(b)> = Σ_m S⁻¹_mn(b) e^{-ib·r} |u_mΓ>,
#               S_nm(b)  = <u_nΓ| e^{-ib·r} |u_mΓ>
#     Eq. (9)   C(r)     = -1/(2π) Im <r| [P_b1, P_b2] P_Γ |r>
#     Eq. (12)  C^(sym)  = -1/(4π) Im Σ_n (<ũ_n(b1)|-<ũ_n(-b1)|)(|ũ_n(b2)>-|ũ_n(-b2)>)
#     Eq. (15)  C^(sym)(r) = -1/(8π) Im <r| ([P_b1,P_b2]+[P_-b1,P_-b2]
#                                          -[P_-b1,P_b2]-[P_b1,P_-b2]) P_Γ |r>
#
# LCPAO (OpenMX) form, with |u_nΓ> = Σ_μ c_μn |φ_μ>  (c†Sc = 1):
#     T(b)_μν = <φ_μ| e^{-ib·r} |φ_ν>          (T_band / set_expOLP of lcpao.jl)
#     S(b)    = c† T(b) c                       (N_occ × N_occ)
#     D(b)    = c S(b)⁻¹                        -> |ũ_n(b)> = e^{-ib·r} Σ_μ D(b)_μn |φ_μ>
#     P_b     = e^{-ib·r} |φ> D(b) D(b)† <φ| e^{+ib·r}
#   Using the duality <ũ_n(b)|u_mΓ> = δ_nm:
#     P_ba P_bb P_Γ = e^{-i ba·r} |φ> K <φ|,   K = D(ba) [D(ba)† T(bb-ba) D(bb)] c†
#     Tr{P_ba P_bb P_Γ} = tr[K T(ba)] = tr[D(ba)† T(bb-ba) D(bb)] = Σ_n <ũ_n(ba)|ũ_n(bb)>
#   <r|...|r> is resolved on the (non-orthogonal) orbitals by a Mulliken partition
#     w_μ(ba,bb) = 1/2 { [K T(ba)]_μμ + [T(ba) K]_μμ },   Σ_μ w_μ = Tr{P_ba P_bb P_Γ},
#   so that Σ_μ C(μ) is exactly the single-point Chern number.
# =============================================================================

# <φ_μ| e^{-i b·r} |φ_ν> at Γ, b = dk[1] b1 + dk[2] b2 + dk[3] b3, spin-doubled (2N × 2N)
function expibr_matrix(M, dk)
    t = T_band(M, set_expOLP(M, dk), [0 0 0], dk)
    Matrix(mortar(reshape([t, zero(t), zero(t), t], 2, 2)))
end

"""
    local_marker(M, numBand)

Single-point local Chern marker for the lowest `numBand` states at Γ.
Returns a NamedTuple with the orbital- and atom-resolved markers (asymmetric and
symmetric formulas), their sums (= single-point Chern numbers) and diagnostics.
"""
function local_marker(M, numBand)
    @expand M
    SpinP_switch == 3 || error("local_marker: only non-collinear (SpinP_switch=3) scfout files are supported.")

    k = [0 0 0]
    valvec = eigen_band(M, k, "eigen")
    c = valvec[1].vectors[:, 1:numBand]
    evals = valvec[1].values
    gap = (evals[numBand+1] - evals[numBand]) * hartree2ev

    # T(b) for all b = σ1 b1 + σ2 b2 that are needed
    Tb = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for s1 in -1:1, s2 in -1:1
        (s1, s2) == (0, 0) && continue
        Tb[(s1, s2)] = expibr_matrix(M, [s1, s2, 0])
    end
    # hermiticity check: T(-b) should equal T(b)†
    herm_err = maximum(norm(Tb[(s1, s2)] - Tb[(-s1, -s2)]') / norm(Tb[(s1, s2)]) for (s1, s2) in keys(Tb))

    # dual states D(b) = c S(b)^-1
    D = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    detS = Dict{Tuple{Int,Int},ComplexF64}()
    for b in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        Sb = c' * Tb[b] * c
        detS[b] = det(Sb)
        D[b] = c / Sb
    end

    # orbital-resolved <μ| P_ba P_bb P_Γ |μ> (Mulliken), sum = Σ_n <ũ_n(ba)|ũ_n(bb)>
    function w(ba, bb)
        X = Tb[(bb[1] - ba[1], bb[2] - ba[2])]          # <φ|e^{+i ba·r} e^{-i bb·r}|φ>
        A = D[ba] * (D[ba]' * X * D[bb])                # K = A c†
        Ta = Tb[ba]
        kt = vec(sum(A .* transpose(c' * Ta), dims=2))  # diag(K T(ba))
        tk = vec(sum((Ta * A) .* conj.(c), dims=2))     # diag(T(ba) K)
        0.5 .* (kt .+ tk)
    end
    comm(ba, bb) = w(ba, bb) .- w(bb, ba)               # <μ| [P_ba, P_bb] P_Γ |μ>

    p1, m1, p2, m2 = (1, 0), (-1, 0), (0, 1), (0, -1)
    orb_asym = -imag.(comm(p1, p2)) ./ (2π)
    orb_sym = -imag.(comm(p1, p2) .+ comm(m1, m2) .- comm(m1, p2) .- comm(p1, m2)) ./ (8π)

    # orbital -> atom (both spin blocks)
    N = sum(Total_NumOrbs)
    atom_asym = zeros(atomnum)
    atom_sym = zeros(atomnum)
    off = 0
    for ia in 1:atomnum
        for i in 1:Total_NumOrbs[ia], s in 0:1
            atom_asym[ia] += orb_asym[off+i+s*N]
            atom_sym[ia] += orb_sym[off+i+s*N]
        end
        off += Total_NumOrbs[ia]
    end

    # independent evaluation of the single-point Chern numbers, Eqs. (3) and (12)
    ov(ba, bb) = tr(D[ba]' * Tb[(bb[1] - ba[1], bb[2] - ba[2])] * D[bb])
    C_asym = -imag(ov(p1, p2)) / π
    C_sym = -imag(ov(p1, p2) - ov(m1, p2) - ov(p1, m2) + ov(m1, m2)) / (4π)

    (; atom_asym, atom_sym, orb_asym, orb_sym, C_asym, C_sym, gap, herm_err, detS,
        xyz=[Gxyz[i] .* bohr2angstroam for i in 1:atomnum])
end
