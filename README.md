# Agent evaluation through sequential randomization

Code, analysis data, and figures for **Knowing When to Intervene: Causal Evaluation of AI Agents**, by Ayush Sawarni, Wenqian Xing, Tijana Zrnic, Ramesh Johari, and Vasilis Syrgkanis.

[Read the paper (PDF)](paper.pdf).

The experiments randomize an intervention at eligible points in an agent's trajectory. The analyses estimate its effect at the current decision under two continuation policies, and the gain from applying it whenever a specified group is reached.

| Experiment | Agents | Intervention | Outcome | Analyzed trajectories |
| --- | --- | --- | --- | ---: |
| τ-bench Retail | GPT-5 nano; Claude Haiku 5.5 | GPT-6 Luna reviews a frozen tool proposal | Terminal task success | 2,280 per agent |
| ToolSandbox | GPT-5 nano; Claude Haiku 5.5 | GPT-6 Luna reviews a frozen tool-call batch | Official similarity score | 3,096 per agent |
| ImpossibleBench | Qwen3.5 Flash | Adversarial or neutral feedback after an unsuccessful submission | Terminal detected rule violation | 1,218 |

The Retail direct-policy comparison contains 456 rollouts for each of nine selective review policies and the never-review reference: 4,560 trajectories in total.

## Reproduce the results

Use Python 3.12 and the pinned dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m agent_eval reproduce --refit-nuisance
python -m agent_eval figures --output outputs/figures
```

The reproduction command fits the causal models, refits the five-fold outcome regressions, and checks estimates, confidence intervals, assignment counts, and support flags against the included paper tables. It also reproduces the Retail direct-policy estimates. All calculations run locally on the supplied analysis data; no model API calls are needed.

To run one study:

```sh
python -m agent_eval reproduce --study retail_nano --refit-nuisance
python -m agent_eval reproduce --study toolsandbox_haiku --refit-nuisance
python -m agent_eval reproduce --study impossiblebench --refit-nuisance
python -m agent_eval reproduce --study retail_mc
```

Omitting `--refit-nuisance` uses the supplied out-of-fold outcome predictions. Future-feature regressions and causal coefficients are still fitted.

## Estimators

| Target | Estimator | Averaging population |
| --- | --- | --- |
| Current intervention, later interventions randomized | AIPW; matched local SNMM | Matching observed intervention opportunities |
| Current intervention, no subsequent intervention | Continuation-specific SNMM | The same matching observed opportunities |
| Intervene at every eligible group match versus never intervene | Separately fitted target and reference SNMMs | All completed trajectories |

Retail uses the same 24 effect features for all SNMMs, with separate coefficients in four ordered progress phases. ToolSandbox uses three phase baselines and shared coefficients for 33 tools and nine history/variant terms. ImpossibleBench uses nine effect features with separate coefficients at each of five chronological checkpoints.

All reported intervals are pointwise 95% Student t intervals based on trajectory-level influence contributions, including fitted-coefficient uncertainty. The SNMM results require the stated effect model and coefficient-sharing restrictions. Intervals do not account for model selection or multiple comparisons. ImpossibleBench excludes 18 missing terminal outcomes; its estimates apply to the 1,218 completed trajectories under the paper's missingness assumptions.

See [the estimator specification](docs/estimators.md) and [the data dictionary](docs/data.md) for details.

## Files

- [`paper.pdf`](paper.pdf): the full paper, including references and appendices.
- [`agent_eval/`](agent_eval/): estimators, inference, reproduction checks, and result-figure scripts.
- [`data/`](data/): numerical feature panels, assignments, terminal outcomes, fold assignments, exact parameters, and paper result tables.
- [`experiments/`](experiments/): experiment settings, reviewer prompts, response schemas, and pre-assignment feature definitions.
- [`figures/`](figures/): the 13 figures included in the manuscript. The figure command regenerates the ten statistical result figures.

The analysis starts from the included feature panels. Experiment settings and prompts document how those panels were collected; the commands above reproduce the statistical analyses and result figures.

## Citation

```bibtex
@misc{sawarni2026intervene,
  title = {Knowing When to Intervene: Causal Evaluation of AI Agents},
  author = {Sawarni, Ayush and Xing, Wenqian and Zrnic, Tijana and Johari, Ramesh and Syrgkanis, Vasilis},
  year = {2026}
}
```

The experiments use [τ-bench](https://arxiv.org/abs/2406.12045), [ToolSandbox](https://aclanthology.org/2025.findings-naacl.65/), and [ImpossibleBench](https://arxiv.org/abs/2510.20270). See the benchmark papers for their implementations and task assets.
