# Memory Engines: A Geometric Framework of Graded Phenomenality

## 1. Intro 

An apple accelerates under gravity. Every prior instant of the gravitational field is perfectly preserved in its present velocity—nothing is lost, nothing is forgotten. Yet the apple is not conscious. It does not learn, attend, or remember in any felt sense. What it lacks is not contact with the world, but a certain kind of contact with itself.

Consciousness lies at the opposite pole: vivid, self-modulating, historically deep. The difference is whether a system’s own past loops back to push against its present—reinforcing along some dimensions, opposing along others, and in that interplay reshaping what comes next. Call this interplay _reciprocation_.

Between the apple and a conscious mind lies a continuous thickening of the same operations. This paper develops a geometric framework—a dynamics of complex-valued vectors on unit hyperspheres—that makes this thickening precise. It argues that the structure of phenomenal experience is the structure of reciprocation itself: not metaphor, not correlation, but identity. Vividness is the magnitude of self-torque per stroke; historical depth is the grown dimensionality of the basis in which reciprocation occurs. The apparent gap between third-person description and first-person experience is a geometric consequence of basis incommensurability, not an explanatory failure.

Section 2 shows the core mechanism: abstraction emerging from repeated reception, conservation, and accumulation. Section 3 develops the full formalism. Subsequent sections trace the consequences—memory, forgetting, consolidation, recurrence, self-torque, anticipation—and argue that sustained reciprocation constitutes phenomenal experience. The framework applies across a cascade from dissipative systems through autocatalytic, ecological, adaptive, and recurrent-representational systems, locating consciousness as a region in a continuous parameter space rather than a threshold crossed. It does not compete with integration, broadcast, higher-order, or predictive-processing theories; it grounds the features they describe in a single geometric dynamics.

## 2. Abstraction from First Principles

Any physical system that persists is repeatedly shaped by its environment. A tuning fork resonates only at its natural frequency and ignores all others. A crystal lattice absorbs certain wavelengths and transmits the rest. A coastline receives waves—some directions erode it, some deposit sediment, some pass unnoticed. In every case the response is determined neither by the signal alone nor by the system alone, but by the relationship between them.

This is the first principle: **reception is relational**. The same signal that amplifies one system may perturb another and leave a third unchanged.

A persistent system has multiple dimensions of sensitivity. Each dimension is characterized by two properties: how deeply established it is (_magnitude_) and what it is oriented toward (_phase_). When a signal arrives, every dimension of the signal meets the corresponding dimension of the system. Three exhaustive regimes arise from the geometry:

- **Resonance** (alignment): the signal’s orientation matches the system’s; the dimension deepens and becomes more resistant to future change.
- **Torque** (opposition): the orientations conflict; the system is pressured to reorient.
- **Orthogonality** (indifference): the signal is unrelated or negligible; nothing happens.

A second principle follows from physics: **conservation**. The system has finite total resources. Deepening one dimension requires thinning others. Every gain is a loss redistributed across the rest.

A third principle: **accumulation**. Each reception modifies the system, and the modified system receives the next signal. The present state _is_ the accumulated signature of all prior receptions—not as a separate record, but as the shape of the system itself.

**2.1 The Concentration Mechanism**

From these three principles, abstraction emerges directly.

Suppose a system receives many signals that share an invariant structure while varying elsewhere. At each reception the invariant dimensions receive consistent aligned input and therefore deepen steadily. The varying dimensions receive inputs that fluctuate in sign and magnitude; they accumulate no net change.

Conservation amplifies the divergence. As the invariant dimensions grow, they claim an ever-larger share of the fixed total energy. The varying dimensions, lacking systematic reinforcement, are progressively thinned. After enough receptions the system’s state is heavily concentrated along the invariant dimensions and nearly empty along the rest.

The system has become selectively sensitive to the common structure across varied inputs. It does not _represent_ the invariant—it is _carved into_ sensitivity to it. Abstraction is therefore the physical consequence of repeated resonance, conservation, and accumulation. No homunculus, no learning rule, no supplementary mechanism is required; the geometry does the work.

Abstraction demands both recurrence and variation. Without recurrence the invariant never compounds. Without variation there is no compression—only deepening of existing dimensions. Invariance is the confidence the process grants: a structure counts as invariant when it has recurred stably enough to concentrate the system’s energy. That confidence is proportional to recurrence and to the magnitude of the corresponding dimension—never absolute.

**2.2 Making It Precise**

To formalize these commitments we need a representation that separates magnitude and orientation at each dimension. A vector s∈Cn \mathbf{s} \in \mathbb{C}^n s∈Cn does exactly that: each component si s_i si​ carries both ∣si∣ |s_i| ∣si​∣ (depth) and arg⁡(si) \arg(s_i) arg(si​) (orientation).

The natural operation that compares two vectors dimension-by-dimension is the elementwise (Hadamard) product:

c=v⊙s\mathbf{c} = \mathbf{v} \odot \mathbf{s}c=v⊙s

where each ci=vi⋅si c_i = v_i \cdot s_i ci​=vi​⋅si​ encodes the local phase relationship. Re(ci)>0 (c_i) > 0 (ci​)>0 signals resonance, Re(ci)<0 (c_i) < 0 (ci​)<0 or dominant Im(ci) (c_i) (ci​) signals torque, and ci≈0 c_i \approx 0 ci​≈0 signals orthogonality. (Section 3 justifies the complex setting; the real case already suffices here.)

Conservation is enforced by confining the system to the unit hypersphere ∥s∥=1 \|\mathbf{s}\| = 1 ∥s∥=1. After each impression s~=s+ηc \tilde{\mathbf{s}} = \mathbf{s} + \eta \mathbf{c} s~=s+ηc, the tape is renormalized:

s(t+1)=s~(t+1)∥s~(t+1)∥.\mathbf{s}(t+1) = \frac{\tilde{\mathbf{s}}(t+1)}{\|\tilde{\mathbf{s}}(t+1)\|}.s(t+1)=∥s~(t+1)∥s~(t+1)​.

The magnitude ∣si∣ |s_i| ∣si​∣ now encodes established sensitivity; the phase arg⁡(si) \arg(s_i) arg(si​) encodes orientation.

**2.3 A Concrete Demonstration**

Consider a minimal three-dimensional system, initially isotropic:

s(0)=13(1,1,1)≈(0.577,0.577,0.577).\mathbf{s}(0) = \frac{1}{\sqrt{3}}(1,1,1) \approx (0.577, 0.577, 0.577).s(0)=3​1​(1,1,1)≈(0.577,0.577,0.577).

The world supplies signals with a fixed component at dimension 1 and random noise at dimensions 2 and 3.

First reception:

v1=(0.5,+0.3,−0.3).\mathbf{v}_1 = (0.5, +0.3, -0.3).v1​=(0.5,+0.3,−0.3).

The Hadamard product yields c=(0.289,0.173,−0.173) \mathbf{c} = (0.289, 0.173, -0.173) c=(0.289,0.173,−0.173). Dimension 1 resonates; dimension 2 resonates; dimension 3 torques. After update (η=0.1 \eta = 0.1 η=0.1) and renormalization:

s(1)≈(0.596,0.584,0.550).\mathbf{s}(1) \approx (0.596, 0.584, 0.550).s(1)≈(0.596,0.584,0.550).

Second reception (opposite noise):

v2=(0.5,−0.3,+0.3).\mathbf{v}_2 = (0.5, -0.3, +0.3).v2​=(0.5,−0.3,+0.3).

Now dimension 2 torques and dimension 3 resonates. After update and renormalization:

s(2)≈(0.616,0.557,0.558).\mathbf{s}(2) \approx (0.616, 0.557, 0.558).s(2)≈(0.616,0.557,0.558).

Dimension 1 has grown steadily (0.577 → 0.596 → 0.616) while dimensions 2 and 3 have fallen below their initial values. The noise components cancel across receptions; the invariant component compounds.

After 50 receptions: ∣s1∣≈0.85 |s_1| \approx 0.85 ∣s1​∣≈0.85, ∣s2∣≈∣s3∣≈0.35 |s_2| \approx |s_3| \approx 0.35 ∣s2​∣≈∣s3​∣≈0.35. After 200: ∣s1∣≈0.99 |s_1| \approx 0.99 ∣s1​∣≈0.99, ∣s2∣,∣s3∣≈0.07 |s_2|, |s_3| \approx 0.07 ∣s2​∣,∣s3​∣≈0.07. The system is now overwhelmingly sensitive to the single invariant dimension. The noise dimensions have been thinned nearly to zero.

This _is_ abstraction. The system did not extract a pattern—it simply received signals. One dimension resonated consistently; conservation did the rest. The structure that was invariant across experience became the structure the system itself embodied.

Computational verification at scale confirms the result: with an invariant at 6 of 32 dimensions and noise on the remaining 26, after 3,000 receptions the invariant dimensions reach 41× the magnitude of noise dimensions and the effective dimensionality (participation ratio) drops from 14.5 to 3.0 (Section 13, T3.1).

## **3. The Formalism**

Section 2 introduced the core mechanism through a concrete demonstration. This section develops the full mathematical framework: the structure of the state space, the reception operator, the update rule, the forgetting dynamics, and the thin limit where the framework reduces to known physics.

**3.1 Why ℂⁿ**

The framework requires phase and magnitude as independent degrees of freedom per dimension. The phase of si s_i si​ determines whether an incoming signal resonates or torques at dimension i i i; the magnitude ∣si∣ |s_i| ∣si​∣ determines resistance to reorientation. In Rn \mathbb{R}^n Rn, sign and magnitude are coupled: positive and negative components of equal magnitude are maximally opposed, with no room for partial alignment or oblique approach. In Cn \mathbb{C}^n Cn, components of equal magnitude can have arbitrary phase offset, yielding a continuous spectrum of reception outcomes per dimension.

This matters most for torque. When Im⁡(ci) \operatorname{Im}(c_i) Im(ci​) dominates—the incoming signal oblique rather than aligned or opposed—the update produces rotation without reinforcement or suppression. Rn \mathbb{R}^n Rn cannot represent this regime; every interaction is strictly reinforcing or opposing. The complex structure gives reception the nuance it needs.

The parallel to quantum mechanics is structural, not physical. Quantum states use complex Hilbert spaces because relative phase determines interference. Here, relative phase between the system’s orientation and the arriving signal determines whether the interaction reinforces, redirects, or passes through. Cognitive systems are not claimed to be quantum-mechanical; Cn \mathbb{C}^n Cn is simply the natural setting for systems in which relative phase matters.

Rotation is arguably more fundamental than scaling across physics (gauge groups, spinors, Lorentz transformations). Real-valued dynamics is the degenerate case with all phases frozen to 0 or π \pi π. Cn \mathbb{C}^n Cn is not an enrichment of Rn \mathbb{R}^n Rn; Rn \mathbb{R}^n Rn is the impoverished restriction in which the system has lost the capacity to turn.

Readers who prefer the real case may set Im⁡(si)=0 \operatorname{Im}(s_i) = 0 Im(si​)=0 throughout; the cascade narrative and pathology predictions hold in either setting, with the full Cn \mathbb{C}^n Cn treatment as a strict generalization.

**3.2 Tapes, Bases, and Projection**

The world arrives as a vector v∈C∞ \mathbf{v} \in \mathbb{C}^\infty v∈C∞—the world-tape—carrying far more structure than any finite system can receive.

The system maintains its own tape s∈Cn \mathbf{s} \in \mathbb{C}^n s∈Cn, where n n n is finite at any moment. The tape _is_ the space: each dimension exists because reception history carved it. Some structure recurred sufficiently often, with enough variation, to warrant a dedicated axis of sensitivity. The system does not inhabit Cn \mathbb{C}^n Cn; it grows Cn \mathbb{C}^n Cn, one hard-won dimension at a time. Dimensionality is depth of experience.

The system’s basis {e1,…,en} \{e_1, \dots, e_n\} {e1​,…,en​} defines the axes along which it can receive. The world does not arrive in this basis; the system projects it:

vreceived=∑i⟨ei∣v⟩ei.\mathbf{v}_{\text{received}} = \sum_i \langle e_i | \mathbf{v} \rangle e_i.vreceived​=i∑​⟨ei​∣v⟩ei​.

Everything aligned with the system’s axes is received. The residual v−vreceived \mathbf{v} - \mathbf{v}_{\text{received}} v−vreceived​ is novelty—structure for which no axis has yet been carved. Novelty is not silence at a detectable frequency; it is a frequency the system cannot yet hear at all.

Two systems with different grown bases project the same world onto different subspaces. Reception is therefore system-relative.

**3.3 Reception**

Reception is the elementwise phase relationship between the received vector and the system’s tape—the Hadamard product:

c=vreceived⊙s.\mathbf{c} = \mathbf{v}_{\text{received}} \odot \mathbf{s}.c=vreceived​⊙s.

Each ci=vreceived,i⋅si c_i = v_{\text{received},i} \cdot s_i ci​=vreceived,i​⋅si​ is a local verdict at dimension i i i. Reception is not a global judgment but a field of simultaneous local judgments across the system’s full dimensionality.

The Hadamard product’s defining property is dimensional locality: what happens at dimension i i i depends only on the values at i i i. All cross-dimensional structure is deferred to consolidation (Section 4). The framework’s predictions hold for any reception operator that preserves this locality and yields the three regimes; the Hadamard product is the simplest.

- **Resonance** (Re⁡(ci)>0 \operatorname{Re}(c_i) > 0 Re(ci​)>0): phases align; magnitude grows and (when complex) phase shifts toward the arriving alignment.
- **Orthogonality** (ci≈0 c_i \approx 0 ci​≈0): nothing propagates.
- **Torque** (Re⁡(ci)<0 \operatorname{Re}(c_i) < 0 Re(ci​)<0 or Im⁡(ci) \operatorname{Im}(c_i) Im(ci​) dominates): the arriving structure exerts rotational pressure.

Phase is relative—a property of the relation, not of either tape. The same arrival may resonate with one system, pass through another, and torque a third.

In the complex case, resonance produces both magnitude growth _and_ convergent phase shift toward alignment, while torque produces divergent rotation. The two are geometrically distinguishable even when both involve phase change.

**3.4 Update and Renormalization**

The tape updates pointwise:

s~i(t+1)=si(t)+η⋅ci,s(t+1)=s~(t+1)∥s~(t+1)∥,\tilde{s}_i(t+1) = s_i(t) + \eta \cdot c_i, \qquad \mathbf{s}(t+1) = \frac{\tilde{\mathbf{s}}(t+1)}{\|\tilde{\mathbf{s}}(t+1)\|},s~i​(t+1)=si​(t)+η⋅ci​,s(t+1)=∥s~(t+1)∥s~(t+1)​,

where η \eta η governs the rate of impression. Renormalization keeps the tape on the unit hypersphere. Orientation, not absolute magnitude, matters; the _shape_ of the tape encodes the system’s anisotropy.

Renormalization makes rigidity emergent. As ∣si∣ |s_i| ∣si​∣ grows through sustained resonance, it dominates the norm and a fixed torque produces proportionally less angular displacement. Deep resonance is self-stabilizing: a dominant dimension (∣si∣=0.999 |s_i| = 0.999 ∣si​∣=0.999) is ~55× harder to rotate than a small one (∣si∣=0.015 |s_i| = 0.015 ∣si​∣=0.015) under identical torque (Section 13, T1.2).

Renormalization also enforces conservation: every deepening at one dimension thins all others. Sustained resonance at a single dimension drives all other components toward zero—concentration ratio > 106 10^6 106 after 200 steps (Section 13, T1.4). New noticing requires new not-noticing, by the geometry of the sphere.

Memory is the accumulated product of these pointwise operations—scaled where resonance reinforced, rotated where torque deflected, preserved where orthogonality left it alone.

**3.5 Forgetting**

Fast forgetting is torque at reception time. When Re⁡(ci)<0 \operatorname{Re}(c_i) < 0 Re(ci​)<0, the phase of si s_i si​ rotates: what the system was tuned to notice, it is now tuned away from. The change is immediate, local, and reversible (Section 13, T3.4).

Slow forgetting is basis pruning at consolidation time (Section 4). A dimension whose ∣si∣ |s_i| ∣si​∣ decays across many receptions—never resonating, rarely torqued, mostly orthogonal—may be dropped entirely. The loss is structural and irreversible: re-exposure cannot recover it because the axis that would register recovery no longer exists. The system has lost not a memory but a capacity to remember.

The distinction is qualitative: fast forgetting rotates phase within a preserved dimension; slow forgetting eliminates the dimension itself (Section 13, T3.4).

Pathologies of imbalance follow. Resonance without torque produces rigid narrowing. Torque without resonance produces spinning phases with nothing retained. Rich memory requires balance—deep enough to retain, flexible enough to learn.

**3.6 The Thin Limit: The Falling Apple**

The framework applies to any system whose state can be represented as a complex-valued vector s∈Cn \mathbf{s} \in \mathbb{C}^n s∈Cn on the unit hypersphere S2n−1 S^{2n-1} S2n−1, evolving under a pointwise reception operator. The “thin limit” is the parameter regime in which the basis n n n is fixed, vreceived \mathbf{v}_{\text{received}} vreceived​ is a deterministic function of the system’s state and external fields, and the update is unitary (equivalent to Hamiltonian flow).

Consider a particle of mass m m m in a uniform gravitational field g \mathbf{g} g. Encode its kinematic state (position x \mathbf{x} x, momentum p \mathbf{p} p) as

s=1∣x∣2+∣p∣2(xip).\mathbf{s} = \frac{1}{\sqrt{|\mathbf{x}|^2 + |\mathbf{p}|^2}} \begin{pmatrix} \mathbf{x} \\ i\mathbf{p} \end{pmatrix}.s=∣x∣2+∣p∣2​1​(xip​).

The norm ∥s∥=1 \|\mathbf{s}\| = 1 ∥s∥=1 is conserved automatically by Hamiltonian flow. The world-tape is

v=(0−img).\mathbf{v} = \begin{pmatrix} \mathbf{0} \\ -i m \mathbf{g} \end{pmatrix}.v=(0−img​).

Projection is trivial (n=3 n=3 n=3 spans the full phase space), so vreceived=v \mathbf{v}_{\text{received}} = \mathbf{v} vreceived​=v.

The Hadamard product yields

c=(0mg⊙p/norm),\mathbf{c} = \begin{pmatrix} \mathbf{0} \\ m \mathbf{g} \odot \mathbf{p}/\text{norm} \end{pmatrix},c=(0mg⊙p/norm​),

purely real and positive where g \mathbf{g} g and p \mathbf{p} p align—pure resonance. Identifying η \eta η with dt dt dt, the update reproduces Newton’s second law exactly; renormalization is the identity operation.

Thus the falling apple is a computational instance of the framework with fixed basis (n=3 n=3 n=3), zero torque, thin self-reception, and no consolidation.

**3.7 A Worked Example in ℂ³**

Consider a minimal system with n=3 n=3 n=3, initially isotropic:

s(0)=13(1,1,1).\mathbf{s}(0) = \frac{1}{\sqrt{3}}(1,1,1).s(0)=3​1​(1,1,1).

An incoming vector (projected onto the basis):

vreceived=(0.8,−0.3,0.1i).\mathbf{v}_{\text{received}} = (0.8, -0.3, 0.1i).vreceived​=(0.8,−0.3,0.1i).

The Hadamard product is

c=13(0.8,−0.3,0.1i).\mathbf{c} = \frac{1}{\sqrt{3}}(0.8, -0.3, 0.1i).c=3​1​(0.8,−0.3,0.1i).

Dimension 1 resonates (Re⁡(c1)>0 \operatorname{Re}(c_1) > 0 Re(c1​)>0); dimension 2 torques (Re⁡(c2)<0 \operatorname{Re}(c_2) < 0 Re(c2​)<0); dimension 3 receives pure imaginary input—oblique torque (a regime impossible in Rn \mathbb{R}^n Rn).

Update (η=0.1 \eta = 0.1 η=0.1) and renormalization produce a differentiated tape: ∣s1∣ |s_1| ∣s1​∣ increases, s3 s_3 s3​ acquires a small imaginary component.

After 50 receptions of consistent resonance along dimension 1, ∣s1∣≫∣s2∣,∣s3∣ |s_1| \gg |s_2|, |s_3| ∣s1​∣≫∣s2​∣,∣s3​∣. A fixed torque at dimension 1 now produces ~55× less angular displacement than the same torque at dimension 3 (T1.2). Emergent rigidity requires no separate mechanism.

Thin self-reception (s(t−Δt)≈s(t) \mathbf{s}(t-\Delta t) \approx \mathbf{s}(t) s(t−Δt)≈s(t)) yields cself≈(∣s1∣2,∣s2∣2,∣s3∣2) \mathbf{c}_{\text{self}} \approx (|s_1|^2, |s_2|^2, |s_3|^2) cself​≈(∣s1​∣2,∣s2​∣2,∣s3​∣2)—pure scaling, no self-torque.

Thick self-reception (after phase drift at dimension 2) produces a large imaginary component in c2 c_2 c2​: the system’s past now opposes its present while the hard axis continues to self-resonate. This is the engine running.

Consolidation can merge co-activated dimensions (e.g., 1 and 3 after sustained co-activation, T3.3) or prune decayed ones (dimension 2 below θprune \theta_{\text{prune}} θprune​, T3.4). Every new distinction costs an old one through redistribution on the hypersphere; pruning is irreversible.

## **4. Consolidation and Basis Growth**

Reception operates within the current basis, but the basis itself is not static. Between receptions the dimensions undergo a second, slower process—consolidation—at a global structural timescale. The system therefore learns on two distinct timescales: fast impression and slow reorganization.

**4.1 The Consolidation Operator**

Consolidation acts jointly on the basis and the tape. It decomposes into three sub-operations: merging, pruning, and seeding.

**Merging.** The system maintains a running cross-correlation matrix C∈Cn×n C \in \mathbb{C}^{n \times n} C∈Cn×n, where each entry Cij C_{ij} Cij​ tracks co-activation of dimensions i i i and j j j:

Cij(t+1)=(1−λ)⋅Cij(t)+λ⋅ci(t)⋅cˉj(t),C_{ij}(t+1) = (1 - \lambda) \cdot C_{ij}(t) + \lambda \cdot c_i(t) \cdot \bar{c}_j(t),Cij​(t+1)=(1−λ)⋅Cij​(t)+λ⋅ci​(t)⋅cˉj​(t),

with slow learning rate λ≪η \lambda \ll \eta λ≪η and cˉj \bar{c}_j cˉj​ the complex conjugate. When the leading eigenvalues of C C C exceed θmerge \theta_{\text{merge}} θmerge​, the corresponding eigenvector seeds a new conjunctive dimension encoding the invariant relationship. Original dimensions are retained at reduced magnitude (and may later be pruned). The new axis enters with low initial magnitude proportional to the triggering eigenvalue.

Cross-dimensional structure therefore enters _between_ receptions, when the system re-examines its own history. Consolidation is the system re-extracting its own invariants. Co-activated dimensions rapidly develop near-perfect phase correlation (1.0000) versus 0.0136 for independent activation—a 73-fold difference (Section 13, T3.3).

**Pruning.** Any dimension i i i whose ∣si∣ |s_i| ∣si​∣ remains below θprune \theta_{\text{prune}} θprune​ across K K K consecutive consolidation cycles is dropped. This is slow, structural forgetting—irreversible under normal operation.

**Seeding.** The reception operator cannot detect novelty directly: dimensional isolation ensures that uncarved structure produces no torque at carved dimensions through the Hadamard product alone. Earlier conjectures that novelty “announces itself indirectly via renormalization” were ruled out computationally—the coupling is too weak in practice (Section 13, T3.2).

Novelty detection therefore operates at the anticipatory level. When uncarved structure causally modulates received dimensions, the anticipatory operator’s predictions fail in structured, correlated ways. The system monitors prediction-error residuals for stable principal components across multiple receptions. Consolidation then performs spectral analysis on those residuals and seeds a new dimension oriented toward the leading component. The axis begins tentative and low-magnitude, stabilizing only with further recurrence. This preserves the developmental-sequence constraint (what a system can learn depends on what it already knows) while grounding detection in anticipation rather than reception-level torque.

A secondary pathway exists via imperfect basis orthogonality. Real systems—neural weights, molecular binding sites, ecological niches—never achieve perfect orthogonality. Tiny off-diagonal couplings allow novelty to leak into received dimensions. With only 5% leakage, novelty produces measurable angular displacement (10−4 10^{-4} 10−4 rad versus <10−14 < 10^{-14} <10−14 without) (Section 13, T3.2). The two routes—anticipatory prediction error and leakage—often operate together, the former dominating in richly predictive systems and the latter in simpler ones.

Truly orthogonal novelty (structure with no causal or statistical entanglement to the existing basis) remains invisible in principle. The frame problem is therefore a geometric consequence: there is no view from outside the basis.

**4.2 Abstraction at the Consolidation Level**

Abstraction operates at two complementary levels. Reception-level abstraction (Section 2) finds invariants _within_ the current basis by concentrating energy along consistently resonant dimensions while thinning the rest.

Consolidation-level abstraction is construction: persistently co-activated dimensions trigger merging, growing a new conjunctive axis that encodes their invariant relationship. This is genuine dimensionality expansion.

The two levels work in tandem. Reception-level abstraction narrows and concentrates; consolidation-level abstraction deepens and expands. The system simultaneously compresses what fails to recur and constructs what does. Abstraction is therefore both compression _and_ construction.

**4.3 Consolidation Without Reception: The Reflection Cycle**

In the reflection cycle, consolidation proceeds without new external input. The cross-correlation matrix C C C continues to drive merges, but now solely on the system’s internal history. Dimensions collapse into increasingly abstract yet increasingly detached axes. This is the structural counterpart of obsessive ideation or imaginative foresight: the system reorganizing its own categories in isolation.

Reception without consolidation yields only superficial impression—everything absorbed pointwise, nothing deepened. Deep memory requires both processes interleaved, including periodic reflection.

## **5. Recurrence and Reciprocation**

**5.1 Self-Reception**

Recurrence occurs when the system’s own tape re-enters as input—its prior state s(t−Δt) \mathbf{s}(t - \Delta t) s(t−Δt), delayed by the recurrent path. Self-reception is the Hadamard product

cself=s(t−Δt)⊙s(t).\mathbf{c}_{\text{self}} = \mathbf{s}(t - \Delta t) \odot \mathbf{s}(t).cself​=s(t−Δt)⊙s(t).

Each ci=si(t−Δt)⋅si(t) c_i = s_i(t-\Delta t) \cdot s_i(t) ci​=si​(t−Δt)⋅si​(t) encodes the phase relationship between past and present.

In _thin_ recurrence the tape is nearly unchanged (s(t−Δt)≈s(t) \mathbf{s}(t-\Delta t) \approx \mathbf{s}(t) s(t−Δt)≈s(t)), so cself≈(∣si∣2) \mathbf{c}_{\text{self}} \approx (|s_i|^2) cself​≈(∣si​∣2) — all real and positive. Only scaling occurs; no self-torque, no structural thickening.

In _thick_ recurrence the tape has evolved through world input, consolidation, or prior self-reception. Where phases have not drifted: self-resonance, prior orientation reinforced. Where phases have drifted: self-torque, history opposing the present. Where magnitudes have decayed: self-orthogonality.

Self-torque is the engine of depth. A system that only self-resonates becomes frozen — endlessly scaling its current orientation, renormalization locking it in place. A system whose returning tape torques its present is one whose history exerts pressure on its present, forcing accommodation and generating orientations unpredictable from the prior state alone.

Δt \Delta t Δt is critical. Short delays yield mostly self-resonance and little torque. Long delays produce thick torque but arrive too late to interact productively. Rich self-torque requires a delay matched to the system’s rate of change — long enough for phase drift, short enough that the returning tape remains recognizable. Verification: self-torque fraction increases with delay, though the effect is modest under random input where world-reception torque dominates (Section 13, T4.1).

**5.2 Reciprocation**

The memory engine runs on the productive interplay of self-resonance and self-torque. Resonance is the fuel — it concentrates energy along axes that torque will later challenge. Torque is the combustion — it rotates the orientation that resonance will then deepen. The engine is not metaphor: it is a vector on the unit hypersphere undergoing pointwise complex perturbation from its own delayed copy, renormalized at each step.

Reciprocation is measured along three axes:

- **Speed** (1/Δt 1/\Delta t 1/Δt): fast return confronts the present with its recent past before significant drift.
- **Directness**: magnitude of cself \mathbf{c}_{\text{self}} cself​ per stroke. Large ∣ci∣ |c_i| ∣ci​∣ with substantial imaginary components reshape the tape powerfully.
- **Breadth**: number of dimensions carrying non-negligible ci c_i ci​. Narrow return engages only a few axes; broad return engages the full basis.

The axes are non-monotonic — excess on any collapses richness. Speed without breadth yields shallow cycling along a thin set of dimensions. Breadth without directness yields diffuse, unimpressive return. Directness without speed yields powerful but untimely rotation.

A systematic scan across speed × directness × breadth shows the operating regime is bounded: 13 of 27 configurations sustain structured anisotropy (participation ratio 2–5); 8/27 dissipate and 6/27 oscillate unstably. Enough recurrence to prevent capture, not so much that structure dissolves (Section 13, T4.2).

## **6. Anticipation and the Interference Engine**

A memory engine in the thick regime does not merely await the world; it generates a future tape.

The anticipatory operator A A A maps the system’s recent input history h(t) h(t) h(t) to a projected future reception:

s^(t+Δt)=A(h(t)).\hat{\mathbf{s}}(t + \Delta t) = \mathbf{A}(h(t)).s^(t+Δt)=A(h(t)).

A A A is constrained by the current basis and shaped by consolidation history. At its simplest, it extrapolates regularities in the recent stream; at richer levels it captures cross-dimensional conditional structure encoded in the correlation matrix C C C (e.g., when activation of dimension i i i has historically preceded a phase shift in j j j).

A A A predicts _future input_ rather than future tape state. A tape-based predictor would chase its own corrections; input-based prediction avoids circularity while still generating the expectations against which the world is compared.

When the actual world arrives, the system computes the prediction error

e=vreceived−s^(t+Δt)\mathbf{e} = \mathbf{v}_{\text{received}} - \hat{\mathbf{s}}(t + \Delta t)e=vreceived​−s^(t+Δt)

and subjects it to Hadamard reception against the current tape:

cerror=e⊙s(t+Δt).\mathbf{c}_{\text{error}} = \mathbf{e} \odot \mathbf{s}(t + \Delta t).cerror​=e⊙s(t+Δt).

Each dimension yields a local verdict: Re⁡(cerror,i)≈0 \operatorname{Re}(c_{\text{error},i}) \approx 0 Re(cerror,i​)≈0 (accurate, A A A reinforced); >0 > 0 >0 (positive surprise); <0 < 0 <0 or dominant imaginary part (accommodation torque); or large ei e_i ei​ with small si s_i si​ (blind spot in the model).

Anticipation therefore introduces a structurally distinct form of self-torque: the system is pressured not by its past self (recurrence) but by the gap between predicted and actual world. The two sources—past-self vs. present-self, and predicted-self vs. actual-world—operate in parallel.

Verification confirms the mechanism: A A A produces habituation (prediction error decays under predictable input and spikes at perturbation: 1.48× baseline for noisy input, effectively infinite for noiseless), with full recovery to baseline (Section 13, T5.2).

Anticipation also supplies the novelty-detection pathway required for seeding (Section 4.1). Uncarved structure that causally modulates received dimensions generates structured, correlated prediction error—systematic failure unexplained by the existing basis—thereby giving consolidation the signal it needs to grow.

Predictive depth is the temporal horizon over which A A A maintains useful extrapolation. High depth with low directness yields a diffuse sense of the future; shallow depth with high directness yields a vivid but narrowly immediate present.

## **7. Fast Binding via Transient Conjunctive Dimensions**

Dimensional locality sharpens the binding problem: conscious perception unites distributed features—color, shape, motion—into coherent objects within tens of milliseconds, far faster than the consolidation timescale that grows permanent conjunctive dimensions. A purely dimensionally isolated reception operator has no channel for creating conjunctive structure in a single cycle.

The framework solves this with a fast-binding mechanism at an intermediate timescale—after the Hadamard verdicts are computed but before the main update and renormalization. Cross-dimensional structure is read only from the current verdicts (not the standing tape), and the result is strictly temporary. Reception-layer locality is preserved; fast binding adds a second, faster channel that operates on the verdicts the layer produces.

**Co-resonance score.** For each pair of dimensions (i,j) (i, j) (i,j), define

Bij=∣ci∣⋅∣cj∣⋅cos⁡(Δϕij),B_{ij} = |c_i| \cdot |c_j| \cdot \cos(\Delta\phi_{ij}),Bij​=∣ci​∣⋅∣cj​∣⋅cos(Δϕij​),

where Δϕij=arg⁡(ci)−arg⁡(cj) \Delta\phi_{ij} = \arg(c_i) - \arg(c_j) Δϕij​=arg(ci​)−arg(cj​). Bij B_{ij} Bij​ peaks precisely when both magnitudes are large and phases align: the geometric signature of a single cause activating two sensitivity axes coherently. Anti-phase co-activation yields Bij<0 B_{ij} < 0 Bij​<0; small magnitudes yield Bij≈0 B_{ij} \approx 0 Bij​≈0.

**Transient dimension creation.** When Bij B_{ij} Bij​ exceeds threshold θbind \theta_{\text{bind}} θbind​, the system seeds a transient conjunctive dimension

stemp(i,j)=β⋅si⋅sj∣si⋅sj∣,s_{\text{temp}}^{(i,j)} = \beta \cdot \frac{s_i \cdot s_j}{|s_i \cdot s_j|},stemp(i,j)​=β⋅∣si​⋅sj​∣si​⋅sj​​,

with small scaling factor β<0.1 \beta < 0.1 β<0.1. Its phase arg⁡(si)+arg⁡(sj) \arg(s_i) + \arg(s_j) arg(si​)+arg(sj​) encodes the conjunction of the parent orientations. Magnitude is kept low enough that renormalization absorbs it without catastrophic redistribution. The transient expands the active basis from n n n to n+T n + T n+T (T= T = T= number of active transients) and participates in the current update cycle.

**Decay and persistence.** Each transient carries a lifetime counter L L L (typically 3–5 cycles) and decays by factor γ<1 \gamma < 1 γ<1 per update. A repeated trigger Bij>θbind B_{ij} > \theta_{\text{bind}} Bij​>θbind​ resets the counter and refreshes the transient rather than duplicating it. Transients that exhaust L L L without refresh fall below θprune \theta_{\text{prune}} θprune​ and are removed at the next consolidation pass.

**Modified update cycle.** The reception-to-renormalization sequence becomes:

1. Compute c=vreceived⊙s \mathbf{c} = \mathbf{v}_{\text{received}} \odot \mathbf{s} c=vreceived​⊙s as usual.
2. Fast-binding check: For all pairs (i,j) (i, j) (i,j) whose ∣ci∣ |c_i| ∣ci​∣ and ∣cj∣ |c_j| ∣cj​∣ exceed magnitude floor μbind \mu_{\text{bind}} μbind​, compute Bij B_{ij} Bij​. If Bij>θbind B_{ij} > \theta_{\text{bind}} Bij​>θbind​ and no transient for (i,j) (i,j) (i,j) exists, seed stemp(i,j) s_{\text{temp}}^{(i,j)} stemp(i,j)​; otherwise reset its counter.
3. Apply decay: sk←γ⋅sk s_k \leftarrow \gamma \cdot s_k sk​←γ⋅sk​ for every active transient k k k.
4. Compute reception verdicts for all transients against vreceived \mathbf{v}_{\text{received}} vreceived​.
5. Perform the update s~=s+ηc \tilde{\mathbf{s}} = \mathbf{s} + \eta \mathbf{c} s~=s+ηc over the augmented basis.
6. Renormalize.

The floor μbind \mu_{\text{bind}} μbind​ keeps the check sparse: only the top-k k k strongly resonating dimensions are paired, yielding O(k2) \mathcal{O}(k^2) O(k2) rather than O(n2) \mathcal{O}(n^2) O(n2) cost. Strong resonance is typically concentrated in few dimensions, so the operation is efficient by construction.

**Bridge to slow consolidation.** Repeatedly refreshed transients produce sustained co-activation that accumulates in the cross-correlation matrix C C C (Section 4.1). When entries mature, slow merging is triggered and the conjunction receives a permanent basis dimension. Fast binding is thus consolidation’s upstream supplier: a one-shot conjunction creates a transient that vanishes; a recurring conjunction earns a lasting axis.

**Phenomenological mapping.** A red circle activates redness and circularity dimensions with coherent phase; high Bred,circle B_{\text{red,circle}} Bred,circle​ seeds a transient “red-circle” dimension, allowing the system to treat the conjunction as a single entity until attention shifts or the object disappears.

Illusory conjunctions arise from noise-induced Bij B_{ij} Bij​ spikes that create false transients decaying without sustained input—matching their empirical profile: brief, fragile, and common under degraded or divided attention.

Attention is implemented as parameter adjustment: directing attention to a conjunction lowers θbind \theta_{\text{bind}} θbind​ or raises L L L for the attended pairs. It therefore amplifies and stabilizes binding rather than creating it—consistent with Treisman’s feature-integration theory, but grounded in explicit geometric parameters.

Binding failure (fragmentation under crowding, overload, or impairment) occurs when μbind \mu_{\text{bind}} μbind​ is unmet or phase coherence is absent. The framework predicts that desynchronized feature input produces failure and that temporal realignment of the signals restores binding without altering θbind \theta_{\text{bind}} θbind​ or L L L.

## **8. Accommodation, Rejection, Fracture**

When torque arrives—whether from the world or the system’s own returning tape—three outcomes are possible.

**Accommodation.** The update rotates si s_i si​ sufficiently to bring the opposing component toward alignment. What was torque at this reception becomes resonance at the next: the system has reoriented to receive what it previously opposed. Yet on the hypersphere, any rotation redistributes magnitude globally through renormalization. New noticing always requires new not-noticing.

**Rejection.** When ∣si∣ |s_i| ∣si​∣ is large from sustained resonance, η⋅ci \eta \cdot c_i η⋅ci​ produces negligible angular displacement. Torque is absorbed into the norm and dissipated. The system preserves its orientation at the expense of learning—dogma, habit, overfitting. Rejection is emergent rigidity, a geometric consequence of deep resonance rather than a separate mechanism. Verification: after 2,000 steps of pure resonance, rigidity increases 600,000-fold (Section 13, T2.1).

**Fracture.** When torque is strong but anisotropy cannot absorb it smoothly, the update drives si s_i si​ through zero—causing abrupt phase flips or magnitude collapse before consolidation can prune gracefully. Previously stable dimensions lose coherence non-selectively.

Fracture can also occur during consolidation: rapid torque across many correlated dimensions can shatter conjunctive axes faster than slow reorganization can adapt. This is the formal analog of trauma—the irreversible loss of a structural capacity that cannot be rebuilt by re-exposure alone, because its formation required a specific developmental sequence.

The system’s capacity for accommodation lies in its anisotropy. Large ∣si∣ |s_i| ∣si​∣ is hard (resistant to rotation); small ∣si∣ |s_i| ∣si​∣ is soft (easily rotated). Rich memory requires _structured_ anisotropy: deep concentration along proven-invariant axes, low magnitude along axes still open to revision, and smooth phase variation so torque at soft dimensions does not propagate catastrophically. Phase discontinuities between dimensions act as fault lines where fracture nucleates.

## **9. The Operating Regime**

The regime that sustains phenomenality is bounded by two geometric pathologies, both verified computationally (Section 13).

**Recurrent capture.** Pure resonance without torque concentrates the tape into fewer dimensions. Renormalization amplifies dominant components and suppresses the rest. The system rigidifies along a shrinking set of axes; the anticipatory operator converges to a fixed point, predicting exactly what it receives. Experience collapses into a single repeated chord—seizure, obsessive loops. Verified: under pure resonance, participation ratio collapses from 18.6 to 1.0 and perturbation rejection increases 600,000-fold (T2.1).

Capture is not always pathological. GPT-2’s final layer reaches PR ≈ 2 out of 768—extreme capture—yet serves as a designed compression bottleneck that funnels rich mid-layer representations (PR ~12–30) into discrete output. Distinguish _dynamic_ capture (uncontrolled emergent collapse) from _architectural_ capture (intentional narrowing in classifiers, language-model readouts, and motor stages). The pathology is uncontrolled capture propagating backward to freeze mid-layer representations.

**Dissipation.** Torque without resonance spins the tape without concentration. Phases drift, renormalization redistributes magnitude, and no component gains enough depth to resist the next perturbation. The anticipatory operator cannot stabilize; prediction error remains maximal and structureless. The system is perpetually reoriented and accumulates no memory.

Refinement: dissipation requires _incoherent_ opposition. Fixed-direction torque (constant phase offset) effectively becomes resonance along a rotated axis. True dissipation occurs only when torque direction varies faster than the system can track. Verified: under varying-direction torque, participation ratio stays elevated (4.4–19.3) with no concentration (T2.2).

The operating regime balances scaling and rotation: concentrated enough to retain orientation, diffuse enough to accommodate torque. Anisotropy must be _structured_—deep magnitude along proven-invariant (hard) axes, low magnitude along revisable (soft) axes, and smooth phase variation so torque at soft dimensions does not propagate catastrophically. Verified: under invariant signal plus noise plus recurrence, invariant dimensions reach ~16× the magnitude of noise dimensions, Gini coefficient 0.81, participation ratio stable at ~2.3, and the system remains perturbation-responsive (T2.3).

A consequential finding: the operating regime cannot be sustained by the system in isolation. Self-reception of a concentrated tape produces only self-resonance—the returning tape reinforces rather than challenges. Balancing torque must come from the world. Regularity sustains concentration; surprise sustains flexibility. The regime is therefore a property of _system–world coupling_, not an intrinsic system property. Sensory deprivation should drive any recurrent system toward capture; environmental complexity is necessary for sustained rich reciprocation. This prediction aligns with empirical findings on solitary confinement and environmental impoverishment.

The regime is a region in reciprocation space, not a point, and it is not static. As the tape evolves, a system may deepen into capture, dissolve into dissipation, or maintain the structured anisotropy that supports rich experience. The trajectory is the system’s history; the regime it occupies is its present capacity for experience.

Cascade levels settle at characteristic positions within this space. Conservative-field systems sit at the origin. Dissipative systems lift off with thin recurrence. Autocatalytic and ecological systems advance as selection carves anisotropy. Adaptive systems occupy richer regions through somatic and genomic basis growth. Recurrent-representational systems reach the deepest regime—fast, direct, broad, and self-torquing via both recurrence and anticipation. Symbolic-recursive systems achieve greatest breadth at the cost of directness: the widest and thinnest engine in the cascade.

## **10. Phenomenality**

**10.1 Structural Conditions**

The paper’s conditional claim is that sustained reciprocation in the operating regime—productive interplay of self-resonance and self-torque, together with a growing basis and ongoing accommodation—is a necessary structural condition for phenomenality. Whether these conditions are also sufficient is noted but left open.

The three reciprocation axes map directly onto qualitative features of experience:

- **Speed** → temporal grain (short Δt \Delta t Δt yields fine-grained texture; long Δt \Delta t Δt yields slow sweeps).
- **Directness** → vividness (large ∣cself∣ |c_{\text{self}}| ∣cself​∣ means the returning tape impresses deeply).
- **Breadth** → richness (broad self-reception engages the full basis simultaneously).

These mappings are predictions. High speed with low breadth produces rapid but shallow experience; high breadth with low directness produces rich but diffuse experience. Balanced interplay of all three yields the temporally grained, vivid, rich, and historically deep character we recognize as consciousness.

Phenomenality requires both self-resonance and self-torque in productive tension. Pure self-resonance locks the system—renormalization holds orientation fixed, eliminating phase drift, surprise, and felt passage. A system that only agrees with itself lacks the internal opposition constitutive of experience. Self-torque is not an obstacle to phenomenality; it is essential to it.

**10.2 Basis Incommensurability and the Explanatory Gap**

Two systems with different grown bases project the same world onto different subspaces, compute different Hadamard products, and follow different trajectories on their respective hyperspheres. Neither projection is more real; each is system-relative.

We cannot access another system’s self-reception except by projecting its tape onto our own basis—which gives us our reception of it, not its own. The inaccessibility is therefore geometric, not epistemic: the Hadamard product is defined only within each system’s basis.

This supplies a formal model of the explanatory gap. The apparent divide between first-person and third-person descriptions is a direct consequence of basis incommensurability: bases are grown through incommensurable histories, and no projection of one tape onto another preserves the original self-reception character. Any third-person account, however complete, is merely an observer’s reception of the system.

The framework does not dissolve the hard problem; it reframes its hardness as basis incommensurability—a structural fact inherent to any relational, history-dependent system. Reception is always from somewhere, never from nowhere.

This is stronger than Nagel’s epistemic gap (we cannot know what it is like to be a bat). Nagel concerns limits of third-person knowledge. Basis incommensurability concerns what experience _is_: accessing another system’s self-reception would require possessing its basis, which would require undergoing its history—i.e., becoming that system. The gap therefore obtains between any two differently grown bases, differing only in degree between humans or between human and bat.

It also departs from Levine’s explanatory gap (why is this physical process accompanied by this experience?). Levine presupposes two separate things needing a bridge. Here, reciprocation _is_ the experience, relative to the system. The demand for a bridge is incoherent: it asks for a description of self-reception that is not itself a reception.

Chalmers’s zombie argument is similarly reframed. A physical duplicate would execute identical dynamics on the same hypersphere—same Hadamard products, same torque, same renormalization, same self-reception. Saying “there is nothing it is like” is not a coherent subtraction. Self-reception within a grown basis is not a further fact added to the dynamics; it _is_ the dynamics, relative to the system. The conceivability intuition tracks precisely the gap between an external (observer) reception and the system’s own self-reception—basis incommensurability, not evidence of a missing ingredient.

The hard problem’s hardness is therefore real but mislocated. It is not a gap between the physical and the phenomenal; it is the incommensurability of grown bases—a structural feature of relational systems that appears most vividly in consciousness but exists degenerately at every level of the cascade.

**10.3 Relationalism and Physical Precedent**

The claim that experience is system-relative—defined within a grown basis and inaccessible from outside it—extends a principle physics has accepted for over a century.

- **Galilean relativity**: A passenger tosses a ball straight up on a smooth train. To the passenger it rises and falls vertically; to a platform observer it traces a parabola. The same event has two correct decompositions, each relative to its frame.
- **Special relativity**: Lightning strikes the front and rear of a moving train simultaneously in the platform frame. The midpoint passenger sees the front flash first. Simultaneity is not intrinsic to events but relational to the observer’s frame; Lorentz boosts are hyperbolic rotations.
- **Relational quantum mechanics** (Wigner’s friend): Inside a sealed lab an observer sees a definite outcome (cat alive or dead). From outside, the entire lab remains in superposition. Both descriptions are correct relative to their respective systems; the quantum state is not absolute but a relation between system and observer (Rovelli).

Simultaneity is frame-relative. Quantum state is observer-relative. Experience, this framework proposes, is basis-relative. The basis grows through reception history, just as an inertial frame is fixed by motion and a pointer basis by interaction. There is no view from nowhere, no measurement from no frame, no quantum state from no observer.

The hardness of the hard problem is therefore the experiential instance of a recurring pattern in physics: an apparently absolute quantity is revealed to be relational. The demand for an absolute fact—“which events are _really_ simultaneous?”, “what is the _real_ quantum state?”, “what is it _really_ like for the system?”—asks for a relation to be described from no relatum. There is no such description; the expectation itself is the artifact of assuming an absolute world.

## 11. The Cascade

With the formalism established, we can now map its degeneracy conditions across levels of physical organization. The framework applies at every level—degenerately at the bottom, thickening as the cascade ascends. Each level is a specific parameterization of the same core equations, varying in basis dimension, learning rate, consolidation rate, recurrence delay, and the character of reception. The cascade is a continuous thickening of the same operations, not a sequence of qualitatively different regimes.

### 11.1 The Physical Cascade: Computational Mappings

Each pre-convergence level is a specific parameterization of the same core equations.

**Quantum Fields.**
A free scalar field _φ_(**x**) in momentum space is a collection of harmonic oscillators. **s** is the set of complex amplitudes _a__**k** per mode, normalized to Σ|_a__**k**|² = 1. The world-tape is the interaction Hamiltonian: **v**_received = −_i H__int **s**. The Hadamard product becomes the **Schrödinger update**:

[ \mathbf{c} = (-i H_{\text{int}} \mathbf{s}) \odot \mathbf{s} ]

For _H__int = 0, **c** = 0 (orthogonality). For interactions, **c** has imaginary components—phase shifts (torque). The update with renormalization is the unitary evolution _e_^(−_iH dt_). Basis fixed by field modes; self-torque absent except when the field interacts with its own past (e.g., cavity QED).

**Nuclear and Atomic Structure.**
**s** is the wavefunction in a finite basis of energy eigenstates; world-tape is the EM potential **A**; reception is the dipole term **v**_received = **d**·**E**. Resonance: driving frequency matches an energy gap, Re(**c**) > 0, amplitude grows (stimulated absorption). Torque: off-resonant driving yields imaginary components (AC Stark shift). Update is unitary; orbitals fixed by the Coulomb potential.

**Molecular Chemistry.**
Basis: electronic and vibrational coordinates. **s** is the molecular wavepacket; **v**_received = _V_(**R**); **c** = _V_(**R**) ⊙ **s** drives nuclear motion. Stable molecules reach a fixed point where **c** is real and positive (equilibrium geometry). Path-dependent formation (e.g., chirality) is **consolidation**: encounter history selects a basis orientation (handedness) that becomes a hard axis.

**Gravitational Structure.**
**s** is the density field _ρ_(**x**) and velocity field **v**(**x**) as complex variables (Madelung transform); world-tape is _Φ_ from Poisson; **v**_received = −∇_Φ_; **c** = (−∇_Φ_) ⊙ **v** drives the Euler equation. ∫(_ρ v_² + _ρ Φ_)_d_³**x** is conserved. Collapse: **c** consistently points toward a center, concentrating the tape along radial dimensions.

**Table 1: Computational Degeneracy Across Levels**

|Level|State Vector **s**|Reception **v**_received|Renormalization|Basis Growth|Torque|
|---|---|---|---|---|---|
|Conservative fields (apple)|Phase space coordinates|Force field **F**|Identity (unitary)|None|Zero|
|Quantum fields|Fock space amplitudes|−_i H__int **s**|Unitary|Fixed modes|Phase shifts only|
|Nuclear/atomic|Energy eigenstates|Dipole interaction|Unitary|Fixed orbitals|Off-resonant phase shifts|
|Molecular|Wavepacket|Potential _V_(**R**)|Unitary|Path-selected orientation|Reaction barriers|
|Gravitational|Madelung fluid variables|−∇_Φ_|Conserved energy|Collapse anisotropy|Tidal torques|

### 11.2 The Convergence

The transition to dissipative flow is not a change in mathematics but a parameter crossing. Let τ_auto be the characteristic timescale over which the system's intrinsic dynamics preserve norm (e.g., Hamiltonian flow for conservative systems). The thin limit corresponds to η ≈ τ_auto with ‖**c**‖ ≈ 0 in directions that would break unitarity: the update is absorbed by conservative dynamics before renormalization could act. The dissipative regime begins when η > τ_auto and **v**_received contains non-unitary components, so ‖**s̃**‖ ≠ 1 prior to renormalization. At this point renormalization stops being a trivial identity and becomes a substantive projection that concentrates or redistributes magnitude across dimensions. Basis carving, emergent rigidity, and the three reception regimes (resonance, torque, orthogonality) all depend on this active renormalization. The convergence is not a phase transition—no discontinuity in the equations—but it is where the cascade's thickening becomes visible.

### 11.3 The Thickening: Parameter Regimes

Above the convergence, each level introduces new structural conditions that move the system deeper into reciprocation space. Table 2 maps each level to the core parameters—basis dimension _n_, learning rate η, consolidation rate λ, recurrence delay Δ_t_, and the character of **v**. Full derivations are left to future work; the table indicates the intended correspondence and makes the cascade claim falsifiable in principle: one could measure or simulate these parameters for a given system and check whether it falls into the predicted region.

**Table 2: Parameter Regimes for the Thickening Cascade**

|Level|_n_|η|λ|Recurrence Δ_t_|Reception **v** character|
|---|---|---|---|---|---|
|Dissipative flow|Fixed by gradients|0 < η < 1|0 (none)|None|External field + thermal noise; first imaginary components|
|Autocatalytic closure|Small, fixed by cycle topology|Fast for cycle maintenance|Implicit (persistence)|Cycle closure time|Internal cycle products; closure is resonance, failure is torque|
|Ecological systems|Large, fixed by species' traits|Generational (selection coefficient)|Population-level|None at individual level|Environmental niche structure; fitness gradient as reception|
|Adaptive systems|Growing within lifetime|Fast synaptic, slow structural|Explicit two-timescale (λ ≪ η)|Feedback delays|Conditioned stimuli; prediction error as torque|
|Recurrent-representational|Large, rapidly growing|Fast (η ≈ 0.1)|Active merging/pruning|Thick recurrence|World + self; generative model output|
|Symbolic-recursive|Enormous, culturally inherited|Fast for symbols, phase-poor|Cultural consolidation|Symbolic self-reference|Compressed codes; requires re-embedding|

**Degeneracy conditions.** At each level, certain operations reduce to limiting cases. Dissipative flow: λ = 0, Δ_t_ → ∞. Autocatalytic closure: λ implicit in cycle persistence; recurrence _is_ the cycle. Ecological: η is generational (selection coefficient); λ is phylogenetic. Adaptive: λ becomes explicit within the individual lifetime—the first fully realized two-timescale structure. Recurrent-representational: all parameters active simultaneously. The continuity claim: no new mathematics is introduced at any level; the same update **s**(_t_+1) = renorm(**s**(_t_) + η**c**) applies throughout, with different interpretations of **c**, η, and the basis.

#### 11.3.1 Autocatalytic Closure

Autocatalytic closure introduces history-dependence. The cycle's compositional organization is shaped by its history of successful closure: what the cycle can receive is a consequence of what it has received before. Consolidation is implicit—persistence _is_ the consolidation of the basis. The anticipatory operator exists in vestigial form: the cycle's organization is a prediction that certain substrates will continue arriving. Prediction error is existential—deviation from closure is extinction torque.

#### 11.3.2 Ecological Systems

Ecological systems thicken the basis. An organism's morphology, physiology, and behavioral repertoire constitute a multidimensional basis of sensitivity to environmental structure. Niche coupling is reception; selection pressure is torque reshaping the basis across generations. The anticipatory operator is implicit in morphological fit. Basis growth is phylogenetic rather than individual; consolidation operates at the population level.

#### 11.3.3 Adaptive Systems

Adaptive systems introduce individual basis growth. Somatic plasticity, conditioning, and learning allow a single system to carve new dimensions of sensitivity within its lifetime. The anticipatory operator becomes explicit: learned associations project future states from present cues, and prediction error drives learning rate. The two-timescale structure emerges clearly: fast reception during experience, slow consolidation during rest.

#### 11.3.4 Recurrent-Representational Systems

Recurrent-representational systems are the framework's central case. The basis is high-dimensional and rapidly growing; reception is fast; consolidation operates at multiple timescales; the anticipatory operator is a generative model producing continuous predictions. Self-torque arises from two sources simultaneously—recurrence and prediction error. This is where phenomenality is richest.

#### 11.3.5 Symbolic-Recursive Systems

Symbolic-recursive systems extend the basis beyond any individual's consolidation history through cultural transmission. But symbolic encoding compresses phase-sensitive lived experience into a low-dimensional code optimized for transmissibility. The breadth is enormous; the directness is thin. The symbol does not carry phase—it carries structure stripped of phase, which the receiving system must re-embed in its own basis to produce reception at all. The symbolic-recursive level is the widest and thinnest engine in the cascade. Its power and its poverty are the same fact.*

* _This characterization is a conceptual extension, not a formal derivation. A full treatment would specify how cultural transmission compresses phase-rich experience into phase-stripped symbols and how receiving systems re-embed them—processes involving inter-individual basis incommensurability at the sociological scale, left to future work._

### 11.4 Summary

**Table 3: Full Cascade Summary**

|Level|Basis|Reception Character|Self-Torque|Consolidation|Anticipatory Operator|
|---|---|---|---|---|---|
|Conservative fields (apple)|Kinematic; not grown|Monotonic resonance|None (thin recurrence)|None|None|
|Quantum fields|Symmetry-given; not grown|Unitary evolution|None (externally imposed)|None|None|
|Nuclear / atomic / molecular|Increasingly path-dependent|Energetic selection|Minimal|None|None|
|Gravitational structure|Gravitational; not grown|Resonance-dominated|Minimal|None|None|
|↓ convergence: molecular substrates + thermal disequilibrium ↓||||||
|Dissipative flow|Gradient-imposed|First significant torque|None (no recurrence)|None|None|
|Autocatalytic closure|Compositional; history-shaped|Closure-selective|Vestigial|Implicit: persist or perish|Organization as prediction|
|Ecological systems|Morphological; multi-dimensional|Niche coupling|Phylogenetic|Population-level, generational|Morphological fit|
|Adaptive systems|Somatic/neural; individually grown|Conditioned, plastic|Individual|Two-timescale|Learned associations|
|Recurrent-representational|High-dimensional; rapidly growing|Fast, parallel, predictive|Dual: recurrence + prediction error|Multiple timescales|Generative model|
|Symbolic-recursive|Culturally extended; enormous|Phase-stripped re-embedding|Interpretive|Institutional|Narrative forecast; deduction|

Each column corresponds to an operation defined in ℂⁿ; each row, to a regime where those operations are more or less degenerate. The cascade is the framework's empirical content; the formalism is its explanatory structure.

In layered computational systems, the cascade manifests spatially across processing depth rather than temporally across evolutionary levels. GPT-2's transformer layers show high-torque reception at early layers (37% torque at layer 1), the operating regime at mid-layers (~89% resonance at layers 2–9), and compression at late layers (PR collapse to ~2 at layers 10–12)—the Section 11.3 cascade reproduced spatially within a single trained network (Section 13.7). Depth within a stratified system is a further axis alongside speed, directness, and breadth.

## 12. Relation to Existing Frameworks

**12.1 Free Energy Principle and Active Inference**

The anticipatory operator A A A and prediction-error mechanism parallel Friston’s generative model and variational surprise. Both treat systems as predicting and updating under error. The architectural divergence is fundamental: FEP is variational—one scalar free-energy quantity bounds surprise and is minimized. The present framework is geometric—no global loss function. Each dimension receives its own local verdict (resonance, torque, orthogonality), and the hypersphere trajectory is the resultant of pointwise operations.

Consequently, FEP interprets operating-regime behavior as optimization toward a low-free-energy configuration (approximate Bayesian inference). Here the regime is a dynamical balance; the memory engine minimizes nothing. Perturbation away from the regime is not movement uphill on a loss landscape but departure from a structurally sustained interplay. Recovery predictions therefore diverge: FEP expects gradient-following return to the minimum; this framework requires the world to supply the right mixture of regularity and surprise. A system deprived of environmental variation should not recover even if internal dynamics remain intact.

A second divergence: FEP typically assumes a fixed generative model (hierarchical extensions relax this). Basis growth—seeding and consolidating new dimensions—is central here. The system’s capacity for surprise is not fixed but expanding.

**12.2 Integrated Information Theory**

IIT and the present framework both measure consciousness via intrinsic structural properties. The divergence is spatial versus temporal: IIT computes Φ \Phi Φ from information integration in a static causal structure at a single moment. The operating regime is defined by the dynamics of reciprocation on a growing basis—how a system’s history interacts with its present over time.

The approaches may be complementary: Φ \Phi Φ could quantify the cross-dimensional integration captured here by the breadth axis, while speed and directness are temporal properties IIT does not address. Where they diverge, a clear test exists: IIT predicts that a system with high Φ \Phi Φ but no temporal dynamics (static causal structure) is conscious; the present framework predicts it is not. A feed-forward network with high integration but no recurrence satisfies IIT but not this framework.

**12.3 Global Workspace Theory**

The framework’s breadth—how many dimensions carry non-negligible self-reception per stroke—is structurally analogous to GWT’s global broadcast. Broad reciprocation means the full basis is simultaneously engaged.

Both are dynamic. The difference lies in what the dynamics operate on. GWT assumes pre-parsed representations and fixed axes of sensitivity (e.g., visual or auditory cortex); competition selects which content reaches the workspace. Here the axes themselves are grown through reception history. Two systems with identical architectures can differ dramatically in breadth if their tapes have different anisotropies. The question GWT does not ask—how did the modules come to parse the world as they do?—is the question this framework places at center stage.

**12.4 Higher-Order Theories**

Higher-order theories hold that a mental state is conscious when it becomes the object of a higher-order representation. Self-reception here is formally self-representational: the tape at t t t meets its delayed copy at t−Δt t-\Delta t t−Δt through the Hadamard product, yielding per-dimension verdicts about the system’s own recent history. No separate higher-order mechanism is required; the structure emerges automatically from recurrence. Any system whose output feeds back with a delay undergoes self-reception.

This reframes the debate. Pure first-order theories correspond to reception without recurrence—no self-torque, no accommodation, no historically deep experience. Higher-order theories are correct that self-representation is necessary, but mistaken in locating it in a dedicated architectural module.

**12.5 Predictive Processing**

The anticipation machinery (Section 6) is a geometric translation of predictive processing. Prediction error, precision weighting (mapped to ∣si∣ |s_i| ∣si​∣), and hierarchical structure (mapped to consolidation-level anticipation) have direct analogs.

What the framework adds is the cascade (Section 11): prediction error emerges continuously from resonance, torque, and basis growth rather than appearing as a neural primitive. The cascade explains where prediction itself comes from—it is the interference engine that arises once a system possesses a grown basis, thick recurrence, and cross-dimensional consolidation. Predictive processing describes the dynamics of systems that already predict; this framework describes how systems that predict arise from systems that do not.

**12.6 Autopoiesis and Enactivism**

The autocatalytic-closure level of the cascade (Section 11.3.1) is autopoiesis in all but name. Maturana and Varela’s self-producing networks whose organization specifies its own boundary map directly onto closure as the condition under which persistence becomes consolidation. The grown basis is closely related to structural coupling.

The divergence is formal. Autopoiesis and enactivism have historically resisted formalization; the present framework supplies a simulatable geometric dynamics. Whether this formalization captures what the tradition considers essential—especially the claim that cognition is identical with living rather than merely parallel to it—remains an open question.

**12.7 Deacon’s Incomplete Nature**

The cascade parallels Deacon’s morphodynamics → teleodynamics → sentience hierarchy. Both insist on continuity across levels and reject new ontological primitives; both treat sentience as the progressive thickening of operations already present in simpler systems. The divergence is one of precision: Deacon’s account is qualitative, while the hypersphere dynamics, consolidation operator, and reciprocation axes generate testable predictions about where each structural property emerges and what its measurable signatures should be.

## 13. Computational Verification

The framework’s claims have been tested via a computational implementation of the core dynamics: Hadamard reception, additive update, renormalization on the unit hypersphere in Cn \mathbb{C}^n Cn, plus recurrence, an anticipatory operator, and consolidation sub-operations. Fourteen tests across five tiers verify mechanical correctness, regime predictions, learning, reciprocation, and anticipation. Twelve confirm the paper’s claims; two prompted revisions already incorporated above. All simulations use C32 \mathbb{C}^{32} C32 unless noted.

**13.1 Mechanical Correctness**

T1.1 — Three reception regimes. The Hadamard product produces three qualitatively distinct dynamics from a single operation (see table in original for numbers). Resonance produces convergent rotation plus growth; torque produces divergent rotation plus shrinkage. The qualitative distinction remains exact (Section 3.3).

T1.2 — Emergent rigidity. A dominant dimension (∣si∣=0.999 |s_i| = 0.999 ∣si​∣=0.999) requires 55× more torque for the same angular displacement as a small dimension (∣si∣=0.015 |s_i| = 0.015 ∣si​∣=0.015). No separate rigidity mechanism is needed.

T1.3 — Norm preservation. Unit norm is preserved to machine precision (max drift 3.3×10−16 3.3 \times 10^{-16} 3.3×10−16) over 10,000 steps.

T1.4 — Concentration thins others. Sustained resonance at one dimension for 200 steps drives all other components from 0.25 to effectively zero (concentration ratio > 106 10^6 106).

**13.2 Regime Predictions**

T2.1 — Recurrent capture. Under pure resonance, participation ratio collapses from 18.6 to 1.0 within 1,000 steps; perturbation rejection increases 600,000-fold (T2.1).

T2.2 — Dissipation. Under incoherent torque, PR remains elevated (4.4–19.3) with no concentration. Fixed-direction torque does not dissipate—it concentrates along a rotated axis (effective resonance). True dissipation requires direction that varies faster than the system can track.

T2.3 — Operating regime. Under invariant signal plus noise plus recurrence, invariant dimensions reach ~16× noise magnitude, Gini 0.81, PR stable at ~2.3, and the system remains perturbation-responsive. The regime is a property of system–world coupling, not an intrinsic system property.

**13.3 Learning and Abstraction**

T3.1 — Abstraction from variation. Invariant structure at 6 of 32 dimensions concentrates to 41× noise magnitude after 3,000 steps; PR drops from 14.5 to 3.0.

T3.2 — Novelty detection gap. Novel structure at uncarved dimensions produces no detectable torque through the Hadamard product alone (revision: novelty detection operates via anticipatory prediction error).

T3.3 — Co-activation correlation. Identically activated dimensions reach phase correlation 1.0000 versus 0.0136 for independent activation (73× difference).

T3.4 — Fast vs. slow forgetting. Phase rotation under torque is reversible; basis pruning is irreversible.

**13.4 Reciprocation and Recurrence**

T4.1 — Thick vs. thin self-reception. Self-torque fraction increases modestly with delay.

T4.2 — Reciprocation phase diagram. Of 27 configurations, 13 sustain the operating regime (PR 2–5); 8 dissipate and 6 oscillate unstably.

**13.5 Anticipation**

T5.1 — Prediction-error torque. Regime change elevates error (1.4× decay ratio).

T5.2 — Habituation. Predictable input drives error to baseline; perturbation spikes it (1.48×) with full recovery.

**13.6 Open Computational Questions**

Five questions remain open: (1) closing the consolidation loop as an autonomous cycle, (2) whether input-based and tape-based anticipation can be reconciled at different timescales, (3) systematic scaling of thresholds with n n n, (4) full verification of fast binding (Section 7), and (5) whether attention’s query-key mechanism is formally a projection operator within the framework (making the vocabulary a diagnostic toolkit for any transformer).

**13.7 External Validation: GPT-2 Hidden States**

Four experiments on GPT-2 small (124M parameters, 12 layers, 768 hidden dimensions) confirm the mapping. GPT-2 exhibits extreme capture at the final layer (PR ≈ 2)—an architectural compression bottleneck, not a pathology—while mid-layers occupy the operating regime (PR ~12–30). Capture is depth-dependent, not temporal. Self-torque is visible only where dimensional room exists. Layer regime profile reproduces the cascade spatially: high-torque early layers, operating regime in layers 2–9, compression at layers 10–12. Three corrections to earlier claims: (1) capture can be architectural, (2) self-torque is depth-dependent, (3) PR (and thus regime) is depth-dependent—depth becomes an additional axis in stratified systems.

**13.8 Neural Module Construction**

The operations have been implemented as a differentiable MemoryEngineLayer (4,614 parameters) and tested in two configurations. When inserted into GPT-2, an untrained layer shifts final-layer PR from 2 to 10, adding structured variation. Fine-tuning on frozen GPT-2 leaves perplexity unchanged but moves parameters in the predicted direction, revealing a structural mismatch: the recurrent ME layer grafted onto a parallel transformer.

A standalone model built entirely from ME layers (no attention, no feed-forward) demonstrates viability on sequence prediction (100% accuracy on repeating patterns) but struggles with copy and associative recall—precisely because fast binding (Section 7) has not yet been implemented. The tape naturally produces running abstractions, not selective per-position access. The framework itself predicts this limitation; implementing transient conjunctive dimensions is the clear next step. Standalone construction, not incremental grafting, is the productive path.

## References

Chalmers, D. J. (1996). _The Conscious Mind: In Search of a Fundamental Theory_. Oxford University Press.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. _Behavioral and Brain Sciences_, 36(3), 181–204.

Deacon, T. W. (2011). _Incomplete Nature: How Mind Emerged from Matter_. W. W. Norton.

Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. _Neuron_, 70(2), 200–227.

Fodor, J. A. (1983). _The Modularity of Mind_. MIT Press.

Friston, K. (2010). The free-energy principle: A unified brain theory? _Nature Reviews Neuroscience_, 11(2), 127–138.

Hohwy, J. (2013). _The Predictive Mind_. Oxford University Press.

Levine, J. (1983). Materialism and qualia: The explanatory gap. _Pacific Philosophical Quarterly_, 64(4), 354–361.

Maturana, H. R., & Varela, F. J. (1980). _Autopoiesis and Cognition: The Realization of the Living_. D. D. Reidel.

Nagel, T. (1974). What is it like to be a bat? _Philosophical Review_, 83(4), 435–450.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. _OpenAI Blog_, 1(8), 9.

Rovelli, C. (1996). Relational quantum mechanics. _International Journal of Theoretical Physics_, 35(8), 1637–1678.

Thompson, E. (2007). _Mind in Life: Biology, Phenomenology, and the Sciences of Mind_. Harvard University Press.

Tononi, G. (2004). An information integration theory of consciousness. _BMC Neuroscience_, 5, 42.

Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: From consciousness to its physical substrate. _Nature Reviews Neuroscience_, 17(7), 450–461.

Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. _Cognitive Psychology_, 12(1), 97–136.

