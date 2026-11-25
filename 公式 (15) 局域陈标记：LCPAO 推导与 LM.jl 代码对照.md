# 公式 (15) 局域陈标记：LCPAO 推导与 LM.jl 代码对照

Nov 25, 2026 · @Zhang

只算论文 Eq. (15) 时，需要的全部量是 Γ 点占据态系数 c 和八个方向的 T(b) 矩阵；代码里对应的结果是 `orb_sym`（按轨道）和 `atom_sym`（按原子）。下面按“公式 → 矩阵 → 代码”的顺序整理。

## 1. 目标公式与可用输入

要算的是论文 Eq. (15)，即对称版的周期边界局域陈标记：

```latex
C^{(\mathrm{sym})}(\mathbf r) = -\frac{1}{8\pi}\,\mathrm{Im}\,\langle \mathbf r|\Big([P_{\mathbf b_1},P_{\mathbf b_2}] + [P_{-\mathbf b_1},P_{-\mathbf b_2}] - [P_{-\mathbf b_1},P_{\mathbf b_2}] - [P_{\mathbf b_1},P_{-\mathbf b_2}]\Big) P_\Gamma |\mathbf r\rangle
```

其中用到的定义是论文 Eq. (4) 和 Eq. (7) 上下的两行：

```latex
\begin{aligned}
S_{nm}(\mathbf b) &= \langle u_{n\Gamma}| e^{-i\mathbf b\cdot\mathbf r} |u_{m\Gamma}\rangle \\
|\tilde u_n(\mathbf b)\rangle &= \sum_m S^{-1}_{mn}(\mathbf b)\, e^{-i\mathbf b\cdot\mathbf r}\, |u_{m\Gamma}\rangle \\
P_{\mathbf b} &= \sum_n |\tilde u_n(\mathbf b)\rangle\langle \tilde u_n(\mathbf b)|, \qquad P_\Gamma = \sum_n |u_{n\Gamma}\rangle\langle u_{n\Gamma}|
\end{aligned}
```

这些都是算符形式。OpenMX 的 `.scfout` 能给的只有矩阵，所以整个推导就是把上面的算符翻译成下面三样东西：

| 量 | 定义 | 大小 | 代码来源 |
| --- | --- | --- | --- |
| c | Γ 点占据态在原子轨道上的展开系数，每列一个态 | 2N × n | `eigen_band` 取前 n 列 |
| S | 轨道重叠，轨道不正交，所以 c†Sc = 1 而不是 c†c = 1 | 2N × 2N | 已包含在本征问题里 |
| T(b) | 轨道间的相位矩阵元，见下式 | 2N × 2N | `set_expOLP` + `T_band` |

```latex
|u_{n\Gamma}\rangle = \sum_\mu c_{\mu n}\,|\varphi_\mu\rangle, \qquad T(\mathbf b)_{\mu\nu} = \langle\varphi_\mu| e^{-i\mathbf b\cdot\mathbf r} |\varphi_\nu\rangle
```

μ 是打包的轨道指标（原子 i，轨道 α，自旋 σ），N 是不含自旋的轨道总数，n 是占据态数目。Γ 点 k = 0，所以布洛赫函数和它的周期部分相同；对晶格矢量的求和已吸收进 |φ\_μ⟩ 的定义。

后面反复用到一个缩写：|φ⟩ 表示所有轨道排成的一行，⟨φ| 是对应的一列。于是 Σ\_μν |φ\_μ⟩ M\_μν ⟨φ\_ν| 写成 |φ⟩ M ⟨φ|，而 ⟨φ|Ô|φ⟩ 是一个数字矩阵。

## 2. 第一步：S(b)、D(b) 与对偶关系

把 LCPAO 展开代入 S(b) 的定义，重叠矩阵变成一个 n × n 的小矩阵：

```latex
S_{nm}(\mathbf b) = \sum_{\mu\nu} c^*_{\mu n}\,\langle\varphi_\mu|e^{-i\mathbf b\cdot\mathbf r}|\varphi_\nu\rangle\, c_{\nu m} \;\;\Longrightarrow\;\; S(\mathbf b) = c^\dagger\, T(\mathbf b)\, c
```

把展开代入对偶态。S⁻¹ 和 c 都是数，可以合并；e^{-ib·r} 是算符，不依赖 m，留在求和号外面：

```latex
|\tilde u_n(\mathbf b)\rangle = e^{-i\mathbf b\cdot\mathbf r}\sum_\mu |\varphi_\mu\rangle \Big[\sum_m c_{\mu m} S^{-1}_{mn}(\mathbf b)\Big] = e^{-i\mathbf b\cdot\mathbf r}\sum_\mu D(\mathbf b)_{\mu n}\,|\varphi_\mu\rangle, \qquad D(\mathbf b) = c\,S(\mathbf b)^{-1}
```

指数因子没有丢：它不能并进系数 D，要等到做内积时才变成 T 矩阵。

对偶关系 ⟨ũ\_n(b)|u\_mΓ⟩ = δ\_nm 的矩阵形式是后面化简的关键。左矢取共轭后指数变号，而正号的矩阵元等于 T(b)†：

```latex
\langle\varphi_\mu|e^{+i\mathbf b\cdot\mathbf r}|\varphi_\nu\rangle = \big(\langle\varphi_\nu|e^{-i\mathbf b\cdot\mathbf r}|\varphi_\mu\rangle\big)^* = [T(\mathbf b)^\dagger]_{\mu\nu}
```

```latex
\langle\tilde u_n(\mathbf b)|u_{m\Gamma}\rangle = [D^\dagger T^\dagger c]_{nm}, \qquad D^\dagger T^\dagger c = (c^\dagger T D)^\dagger = (c^\dagger T c\, S^{-1})^\dagger = (S S^{-1})^\dagger = \mathbb 1
```

所以有两条等价的恒等式，记作 (★)：D(b)† T(b)† c = 1，以及 c† T(b) D(b) = 1。这不是巧合：S⁻¹ 放进对偶态定义里，目的就是让这个内积变成 δ。

这一步用到 T(−b) = T(b)†。代码里用 `herm_err` 检查，记录显示五个文件都是 0。

## 3. 第二步：三重乘积 P\_a P\_b P\_Γ

Eq. (15) 里四个对易子展开后，每一项都是 P\_a P\_b P\_Γ 的形式（a、b 取 ±b1、±b2），所以只需要推这一个乘积。

先把每个投影算符单独写成“轨道 × 矩阵 × 轨道”。对 n 的求和藏在矩阵乘积 c c†、D D† 的中间指标里：

```latex
P_\Gamma = |\varphi\rangle\, c\,c^\dagger \langle\varphi|, \qquad P_{\mathbf b} = e^{-i\mathbf b\cdot\mathbf r}\,|\varphi\rangle\, D_{\mathbf b} D_{\mathbf b}^\dagger \langle\varphi|\, e^{+i\mathbf b\cdot\mathbf r}
```

三个首尾相接，出现两个“接缝”，左矢遇到右矢就变成数字矩阵：

```latex
P_a P_b P_\Gamma = e^{-i\mathbf a\cdot\mathbf r}|\varphi\rangle\, D_a D_a^\dagger\, \underbrace{\langle\varphi|e^{+i\mathbf a\cdot\mathbf r}e^{-i\mathbf b\cdot\mathbf r}|\varphi\rangle}_{\text{①}\; =\; T(\mathbf b-\mathbf a)}\, D_b D_b^\dagger\, \underbrace{\langle\varphi|e^{+i\mathbf b\cdot\mathbf r}|\varphi\rangle}_{\text{②}\; =\; T(\mathbf b)^\dagger}\, c\,c^\dagger\langle\varphi|
```

- 接缝 ①：两个相位都只是 r 的函数，指数直接合并，得到 T(b−a)。这就是除了 ±b1、±b2 以外还要算 ±b1±b2 方向 T 矩阵的原因。
- 接缝 ②：等于 T(b)†，它和两侧的 D\_b†、c 合起来正好是恒等式 (★)，D\_b† T(b)† c = 1，直接划掉。

结果：

```latex
P_a P_b P_\Gamma = e^{-i\mathbf a\cdot\mathbf r}\,|\varphi\rangle\, K \,\langle\varphi|, \qquad K = D_a\, M\, c^\dagger, \qquad M = D_a^\dagger\, T(\mathbf b-\mathbf a)\, D_b
```

M 是 n × n 的小矩阵，它的 (l, m) 元素就是两个方向对偶态的内积 ⟨ũ\_l(a)|ũ\_m(b)⟩。K 是 2N × 2N，但代码不显式构造它，只保存 A = D\_a M（2N × n）。

最左边还剩一个 e^{-ia·r} 没有配对。求迹或取对角元时，最右边的 ⟨φ| 绕回来和它相遇，形成第三个接缝 ⟨φ|e^{-ia·r}|φ⟩ = T(a)。

## 4. 第三步：从迹到每个轨道的份额 w\_μ

局域标记就是“还没加起来的迹”：陈数是算符的迹，C(r) 是迹里每个位置的那一份。所以先看迹怎么由各部分加起来。

对上一节的结果求迹，用 Tr(|A⟩⟨B|) = ⟨B|A⟩：

```latex
\mathrm{Tr}\{P_a P_b P_\Gamma\} = \sum_{\mu\nu} K_{\mu\nu}\,\langle\varphi_\nu|e^{-i\mathbf a\cdot\mathbf r}|\varphi_\mu\rangle = \mathrm{tr}[K\,T(\mathbf a)] = \sum_\mu [K\,T(\mathbf a)]_{\mu\mu}
```

右边已经是对轨道 μ 的求和。LCPAO 里没有实空间网格，“位置 r”只能换成“轨道 μ”，再合并到原子。不做求和，留下的每一项就是轨道 μ 的份额。

但轨道不正交，每一项 K\_μν T(a)\_νμ 同时涉及两个轨道。这一对的贡献可以全记给行指标（得到 \[K T\]\_μμ），也可以全记给列指标（得到 \[T K\]\_μμ）。两种记法总和相同，没有理由偏向哪边，所以各取一半，和算原子电荷的 Mulliken 布居分析是同一个做法：

```latex
w_\mu(\mathbf a,\mathbf b) = \tfrac12\Big\{[K\,T(\mathbf a)]_{\mu\mu} + [T(\mathbf a)\,K]_{\mu\mu}\Big\} \;\;\leftrightarrow\;\; \langle\mu|P_a P_b P_\Gamma|\mu\rangle
```

这样分配保证总和严格等于迹。再用轮换不变性和恒等式 (★) 的第二种形式 c†T(a)D\_a = 1，总和化简成论文 Eq. (3) 的求和项：

```latex
\sum_\mu w_\mu(\mathbf a,\mathbf b) = \mathrm{tr}[D_a M c^\dagger T(\mathbf a)] = \mathrm{tr}[M\, c^\dagger T(\mathbf a) D_a] = \mathrm{tr}[M] = \sum_n \langle\tilde u_n(\mathbf a)|\tilde u_n(\mathbf b)\rangle
```

这条等式是第 7 节数值检验的依据。

计算对角元时不需要乘出 2N × 2N 的 K。利用 K = A c†：

```latex
[K\,T(\mathbf a)]_{\mu\mu} = \sum_n A_{\mu n}\,(c^\dagger T(\mathbf a))_{n\mu}, \qquad [T(\mathbf a)\,K]_{\mu\mu} = \sum_n (T(\mathbf a)A)_{\mu n}\; c^*_{\mu n}
```

即“两个 2N × n 矩阵逐元素相乘，再按行求和”。

## 5. 第四步：拼成 Eq. (15)

对易子的份额是两个三重乘积份额之差：

```latex
\mathrm{comm}_\mu(\mathbf a,\mathbf b) = w_\mu(\mathbf a,\mathbf b) - w_\mu(\mathbf b,\mathbf a) \;\;\leftrightarrow\;\; \langle\mu|[P_a,P_b]P_\Gamma|\mu\rangle
```

Eq. (15) 按轨道分解后的最终形式：

```latex
C^{(\mathrm{sym})}_\mu = -\frac{1}{8\pi}\,\mathrm{Im}\Big[\mathrm{comm}_\mu(\mathbf b_1,\mathbf b_2) + \mathrm{comm}_\mu(-\mathbf b_1,-\mathbf b_2) - \mathrm{comm}_\mu(-\mathbf b_1,\mathbf b_2) - \mathrm{comm}_\mu(\mathbf b_1,-\mathbf b_2)\Big]
```

四个对易子共调用 8 次 w。每次用到的 T 矩阵如下，键 (s1, s2) 表示 s1·b1 + s2·b2：

| 对易子 | 符号 | w(a, b) | T(b−a) 的键 | T(a) 的键 |
| --- | --- | --- | --- | --- |
| \[P\_b1, P\_b2\] | + | w(+b1, +b2) | (−1, 1) | (1, 0) |
|  |  | w(+b2, +b1) | (1, −1) | (0, 1) |
| \[P\_−b1, P\_−b2\] | + | w(−b1, −b2) | (1, −1) | (−1, 0) |
|  |  | w(−b2, −b1) | (−1, 1) | (0, −1) |
| \[P\_−b1, P\_b2\] | − | w(−b1, +b2) | (1, 1) | (−1, 0) |
|  |  | w(+b2, −b1) | (−1, −1) | (0, 1) |
| \[P\_b1, P\_−b2\] | − | w(+b1, −b2) | (−1, −1) | (1, 0) |
|  |  | w(−b2, +b1) | (1, 1) | (0, −1) |

所以八个方向的 T(b) 全部用到：四个轴向的用于 S(b)、D(b) 和 T(a)，四个对角方向的用于接缝 T(b−a)。

四项和符号的来源是中心差分。论文 Eq. (12) 用 (⟨ũ(b1)| − ⟨ũ(−b1)|)(|ũ(b2)⟩ − |ũ(−b2)⟩) 代替单侧差分，乘开后同号为正、异号为负。两个方向各带一个 1/2，前因子从非对称公式 Eq. (9) 的 1/(2π) 变成 1/(8π)。

最后把轨道合并到原子。轨道排列是先自旋↑块（第 1 到 N 个），再自旋↓块（第 N+1 到 2N 个），每块内按原子顺序：

```latex
C^{(\mathrm{sym})}(\text{原子 } i) = \sum_{\mu\in i,\ \sigma=\uparrow\downarrow} C^{(\mathrm{sym})}_\mu
```

## 6. 代码对照（LM.jl）

只算 Eq. (15) 时，`LM.jl` 里用到的行和公式的对应关系如下。

| 步骤 | 公式 | 代码 | 行号 | 大小 |
| --- | --- | --- | --- | --- |
| 占据态系数 | c，满足 c†Sc = 1 | `c = valvec[1].vectors[:, 1:numBand]` | 47–49 | 2N × n |
| 相位矩阵 | T(b)，八个方向 | `Tb[(s1, s2)] = expibr_matrix(M, [s1, s2, 0])` | 30–34、54–58 | 2N × 2N |
| 重叠矩阵 | S(b) = c† T(b) c | `Sb = c' * Tb[b] * c` | 66 | n × n |
| 对偶态系数 | D(b) = c S(b)⁻¹ | `D[b] = c / Sb` | 68 | 2N × n |
| 接缝 ① | T(b−a) | `X = Tb[(bb[1] - ba[1], bb[2] - ba[2])]` | 73 | 2N × 2N |
| 小矩阵与 A | M = D\_a† T(b−a) D\_b，A = D\_a M | `A = D[ba] * (D[ba]' * X * D[bb])` | 74 | 2N × n |
| 对角元一 | \[K T(a)\]\_μμ | `kt = vec(sum(A .* transpose(c' * Ta), dims=2))` | 76 | 长度 2N |
| 对角元二 | \[T(a) K\]\_μμ | `tk = vec(sum((Ta * A) .* conj.(c), dims=2))` | 77 | 长度 2N |
| 份额 | w\_μ = ½(两者之和) | `0.5 .* (kt .+ tk)` | 78 | 长度 2N |
| 对易子 | w(a,b) − w(b,a) | `comm(ba, bb) = w(ba, bb) .- w(bb, ba)` | 80 | 长度 2N |
| **Eq. (15)** | 四个对易子，除以 8π | `orb_sym = -imag.(comm(p1, p2) .+ comm(m1, m2) .- comm(m1, p2) .- comm(p1, m2)) ./ (8π)` | 84 | 长度 2N |
| 合并到原子 | 对同一原子的轨道和两个自旋求和 | `atom_sym[ia] += orb_sym[off+i+s*N]` | 87–97 | 长度 atomnum |
| 独立检验 | Eq. (12) 的总数 | `C_sym = -imag(ov(p1, p2) - ov(m1, p2) - ov(p1, m2) + ov(m1, m2)) / (4π)` | 100–102 | 1 个数 |

几点读代码时容易卡住的地方：

- `expibr_matrix`：`T_band` 给的是不含自旋的 N × N 矩阵 t。e^{-ib·r} 不改变自旋，所以用 `mortar` 拼成分块对角的 \[t 0; 0 t\]。
- `c / Sb`：Julia 里 A / B 就是 A 乘 B 的逆，比显式求逆稳定。`'` 是共轭转置 †。
- 第 76 行：`c' * Ta` 是 n × 2N，转置后 (μ, n) 元素是 (c†T(a))\_nμ，和 A 逐元素相乘再按行求和，就是第 4 节最后的第一个式子。
- 第 77 行：c† 的转置是 `conj.(c)`，对应第二个式子。
- 第 93 行的索引 `off+i+s*N`：`off` 是当前原子之前的轨道数，`s=1` 时加 N 跳到自旋↓块。
- `p1, m1, p2, m2` 分别是 +b1、−b1、+b2、−b2 的键。

只算 Eq. (15) 时可以删掉的部分：`orb_asym`、`atom_asym`、`C_asym`（非对称公式 Eq. 9 和 Eq. 3），以及诊断量 `gap`、`herm_err`、`detS`。下面是按这个思路删减的版本；它只是从原函数删行得到的，我这边没有 lcpao.jl 和 `.scfout` 文件，没有实际运行过。

```julia
# 只计算论文 Eq. (15)；依赖 LM.jl 里的 expibr_matrix
function local_marker_sym(M, numBand)
    @expand M
    SpinP_switch == 3 || error("only non-collinear (SpinP_switch=3) is supported")

    # Γ 点占据态系数 c (2N × n)
    c = eigen_band(M, [0 0 0], "eigen")[1].vectors[:, 1:numBand]

    # 八个方向的 T(b)
    Tb = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for s1 in -1:1, s2 in -1:1
        (s1, s2) == (0, 0) && continue
        Tb[(s1, s2)] = expibr_matrix(M, [s1, s2, 0])
    end

    # D(b) = c S(b)^-1,  S(b) = c† T(b) c
    D = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for b in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        D[b] = c / (c' * Tb[b] * c)
    end

    # w_μ(a,b) ↔ <μ| P_a P_b P_Γ |μ>（Mulliken）
    function w(ba, bb)
        X = Tb[(bb[1] - ba[1], bb[2] - ba[2])]          # T(b−a)
        A = D[ba] * (D[ba]' * X * D[bb])                # K = A c†
        Ta = Tb[ba]
        kt = vec(sum(A .* transpose(c' * Ta), dims=2))  # diag(K T(a))
        tk = vec(sum((Ta * A) .* conj.(c), dims=2))     # diag(T(a) K)
        0.5 .* (kt .+ tk)
    end
    comm(ba, bb) = w(ba, bb) .- w(bb, ba)

    # Eq. (15)，按轨道
    p1, m1, p2, m2 = (1, 0), (-1, 0), (0, 1), (0, -1)
    orb_sym = -imag.(comm(p1, p2) .+ comm(m1, m2) .- comm(m1, p2) .- comm(p1, m2)) ./ (8π)

    # 轨道 → 原子（两个自旋块）
    N = sum(Total_NumOrbs)
    atom_sym = zeros(atomnum)
    off = 0
    for ia in 1:atomnum
        for i in 1:Total_NumOrbs[ia], s in 0:1
            atom_sym[ia] += orb_sym[off+i+s*N]
        end
        off += Total_NumOrbs[ia]
    end

    # 检验：sum(atom_sym) 应等于 Eq. (12) 的 C_sym
    ov(ba, bb) = tr(D[ba]' * Tb[(bb[1] - ba[1], bb[2] - ba[2])] * D[bb])
    C_sym = -imag(ov(p1, p2) - ov(m1, p2) - ov(p1, m2) + ov(m1, m2)) / (4π)

    (; atom_sym, orb_sym, C_sym, xyz=[Gxyz[i] .* bohr2angstroam for i in 1:atomnum])
end
```

## 7. 检验与注意事项

**求和检验。** 把第 4 节的求和等式代入 Eq. (15)，并用 ⟨x|y⟩ = ⟨y|x⟩\* 使每个对易子的虚部加倍，所有轨道的总和正好是论文 Eq. (12) 的对称 single-point 陈数：

```latex
\sum_\mu C^{(\mathrm{sym})}_\mu = -\frac{1}{4\pi}\,\mathrm{Im}\sum_n \big(\langle\tilde u_n(\mathbf b_1)| - \langle\tilde u_n(-\mathbf b_1)|\big)\big(|\tilde u_n(\mathbf b_2)\rangle - |\tilde u_n(-\mathbf b_2)\rangle\big) = C^{(\mathrm{sym})}
```

代码里 `sum(atom_sym)` 走的是 w 的路径，`C_sym` 走的是 `ov` 的路径，两者应逐位相同。这是判断实现有没有写错的第一道检查。

**归一化。** `atom_sym` 是原始值，所有原子加起来等于 C，单个原子只有约 C/原子数。论文的局域陈数是宏观平均（单位面积的迹），所以画图用的是 `nat .* atom_sym`，平均值等于 C。这一步在 `run_LocalMarker.jl` 里，不在 `LM.jl` 里。

**和海报写法不同的三处。**

| 位置 | 海报写法 | 这里的写法 | 原因 |
| --- | --- | --- | --- |
| 对偶态（海报 Eq. 18） | 没有 e^{-ib·r} | 保留，与论文 Eq. (4) 一致 | 去掉后 P\_b 和 P\_Γ 在同一个 n 维空间，对易子的迹恒为零 |
| 对偶态内积（海报 Eq. 19） | S⁻¹(b1)† S⁻¹(b2) | 中间夹 c† T(b2−b1) c | 两个相位因子合并后不是单位算符 |
| 局域分解（海报 Eq. 17） | 直接取 ⟨φ\_μ…φ\_μ⟩ | Mulliken 分配 | 非正交基下前者求和不等于迹 |

**还没有确认的地方。**

- T(b) 是一阶近似：库里把 e^{-ib·(r−τ)} 只展开到一阶。Eq. (15) 用到的对角方向矢量比 b 长，误差更大。记录里的估计是 SKX\_2by2 丢掉的二阶项约 0.2，没有做数值验证。
- Mulliken 不是唯一分法：换成 Löwdin 正交化，总和不变，但每个原子上的数值会变。
- S(b) 接近奇异时公式失效：原函数返回的 `detS` 可以用来看这一点，精简版里删掉了。
- SKX\_N4\_3by3 没有对上（标记平均 −0.36，Bott index 为 0，能隙也和另两个文件不同），可能是 n = 9 的填充不对应。
- 记录里 SKX\_2by2 的 −1.116 既被写成 `sum C_asym`，又被写成“marker 平均”。SKX 文件的对称值 C\_sym 是多少没有单独记下来，建议直接看 `.dat` 第 8 列的平均或 `C_sym`。记录里明确的对称值只有 kagome：3by3 为 0.999，4by4 为 1.010，Bott index 都是 1。

## 来源

- N. Baù, A. Marrazzo, “Local Chern Marker for Periodic Systems”, arXiv:2310.15783v2（Phys. Rev. B 109, 014206）：公式编号 Eq. (3)、(4)、(8)、(9)、(12)、(15) 已对照原文核对。
- 上传的《LocalMarker.jl 推导问答记录》：推导步骤、数值结果和海报公式的描述。
- 上传的 `LM.jl`：行号以该文件为准。
