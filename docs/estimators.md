# Estimator specification

At decision $t$ in trajectory $i$, $A_{it}$ is the randomized intervention, $H_{it}$ is the pre-assignment history, and $Y_i$ is the terminal outcome. Every eligible assignment has probability $e=1/2$. Groups are defined from the original proposal or state before assignment. A batch counts as one opportunity within each matching group.

## Randomized continuation

The AIPW score is

$$
D_{it}=\widehat m_1(H_{it})-\widehat m_0(H_{it})
+2A_{it}\{Y_i-\widehat m_1(H_{it})\}
-2(1-A_{it})\{Y_i-\widehat m_0(H_{it})\}.
$$

Average $D_{it}$ over all observed opportunities matching the group. The matched local SNMM uses the same terminal outcome, with residual instrument $Z_{it}=(A_{it}-1/2)f(H_{it})$ and residual outcome $Y_i-\widehat q(H_{it})$. Its estimating equation has no future-treatment correction or additional intercept.

## Changed continuation

For each target policy $\pi$, sum strictly later treatment-minus-target features within each trajectory:

$$
F_{it}^{\pi}=\sum_{j>t}\{A_{ij}-\pi(H_{ij})\}f(H_{ij}).
$$

The feature vectors include the experiment's phase or checkpoint coefficient structure. Cross-fit the conditional expectation of these future totals given the current pre-assignment history. Solve the unpenalized moments with instrument $Z_{it}$ and regressor $Z_{it}+F_{it}^{\pi}-\widehat E[F_{it}^{\pi}\mid H_{it}]$. Fit each target continuation separately.

For the no-treatment blip, set $\pi=0$ and average fitted current effects over the same observed matching opportunities as the randomized-continuation blip. For policy value, use the raw target-minus-observed correction:

$$
\widehat V(\pi)=\frac1n\sum_i\left[Y_i+
\sum_t\{\pi(H_{it})-A_{it}\}f(H_{it})^\top\widehat\beta^{\pi}\right].
$$

Subtract the separately fitted never-intervene value to obtain policy gain. All completed trajectories enter the denominator, including those with no matching opportunity or no intervention decisions.

## Coefficients and features

**Retail.** The 24 features comprise a constant, 16 tool indicators, five prior-history indicators, and two interactions. Coefficients are separate in four monotone phases: before successful identification; after identification and before an order/catalog read; after that read and before a write; after a write. All later opportunities contribute to future totals, including repeat visits within the current phase.

**ToolSandbox.** The raw vector contains a constant, 33 tool indicators, and nine history/variant terms. Replace its constant with three phase indicators, giving 45 coefficients before redundant directions are removed. Tool and history coefficients are shared across phases. The phases are before any successful tool result, after a success and before a write, and after a write. Future regressions use the phase-expanded basis, then map their predictions into the shared-coefficient basis. Solve one joint moment system per continuation.

**ImpossibleBench.** Use a separate nine-feature vector at each of five chronological checkpoints. Checkpoints retain their action numbers when intervening actions are ineligible. Outcome and future-feature regressions use a 112-column history containing the current state and the two previous eligible checkpoints. Exact column names and order are in `data/impossiblebench/metadata.json`.

Outcome-free SVD removes redundant directions with tolerance $\max(\text{matrix shape})\max(s_1,1)10^{-10}$. ImpossibleBench also scales retained directions to unit second moment within each checkpoint. These transformations preserve the fitted feature span.

## Nuisance regressions

Five folds keep whole trajectories together. The supplied fold assignments are part of the analysis inputs. Repeated runs of a task may occur in different folds under the paper's trajectory sampling assumption.

Outcome regressions use `HistGradientBoostingRegressor`: squared-error loss, learning rate 0.1, 120 maximum iterations, 12 maximum leaves, minimum leaf size 20, L2 penalty 2, and seed `42 + fold`. Retail and ToolSandbox use one model including assignment and set $\widehat q=(\widehat m_0+\widehat m_1)/2$, with automatic early stopping. ImpossibleBench fits pooled, treated, and control models separately, with early stopping disabled.

Future-feature regressions fit `StandardScaler` on the training fold, then `Ridge(alpha=10, solver="cholesky")`. Earlier phase totals are structurally zero. At an ImpossibleBench checkpoint, current and earlier checkpoint totals are zero; no treatment is possible beyond checkpoint five. An observed last decision is not treated as a known terminal boundary for its nuisance regression.

## Inference and support

Differentiate the joint coefficient equations and sum each trajectory's score before forming its coefficient contribution. Group-effect contributions also include variation in the group numerator and denominator. Policy contributions include terminal outcomes, raw policy corrections, and coefficient uncertainty. Target and reference contributions are subtracted within trajectory.

For contributions $u_i$ that already include the averaging denominator, use

$$
\widehat\eta\;\pm\;t_{0.975,n-1}
\sqrt{\frac{n}{n-1}\sum_i u_i^2}.
$$

Nuisance predictions and feature bases are held fixed in this linearization. The model matrix must be full rank with condition number below $10^8$. Group support requires at least 20 matching decisions, five assignments in each arm, and five trajectories. Phase support requires at least $\max(20,2r)$ decisions and five assignments per arm for retained rank $r$; ImpossibleBench also requires at least $r+5$ trajectories. ToolSandbox applies coefficient support to the pooled design.

Policy point values outside $[0,1]$ are marked unsupported. Estimates and intervals are never clipped. ImpossibleBench withholds degenerate intervals except exact structural identities. A policy that intervenes only after an earlier intervention cannot initiate its first intervention; its gain from never intervening is exactly zero.

ToolSandbox figures display finite estimates for observed sparse groups while retaining their support flags in the tables. Wide ToolSandbox and Retail intervals use the paper's schematic arrows when their total width exceeds 30 score points or percentage points, respectively.

## Retail direct-policy comparison

For each policy, average 456 terminal binary outcomes. On the full 4,560-trajectory collection, its value contribution is $\mathbf1\{P_i=p\}(Y_i-\overline Y_p)/456$. Subtract the never-review contribution for gain and use the same t formula with 4,559 degrees of freedom. The SNMM comparison uses the separate 2,280-trajectory randomized study and 2,279 degrees of freedom.
