# Federated TinyML Thesis — Current Status Audit

## ✅ What's Done
- **Thesis restructured** into modular LaTeX: `main.tex` + 12 chapter files
- **Simulation completed**: `fl_simulation_results.json` exists (timestamp: 2026-04-13)
- **All 8 figures generated** in `thesis_figures/` (publication-quality PNGs)
- **Table structures** built in all chapters
- **Appendix** with source code excerpts in place
- **References** (`references.bib`) with 7,470 bytes of entries

## 🔴 Remaining `[XX]` Placeholders (38 total across 6 files)

### Tier 1: Auto-fillable from `fl_simulation_results.json` (22 placeholders)

| File | Line(s) | What's needed | Source value |
|------|---------|---------------|-------------|
| [abstract.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/abstract.tex) | 26–28 | Centralized R², RMSE; Federated R², rounds | R²=1.0000, RMSE=0.0007; R²=0.9999, 20 rounds |
| [ch3_related_work.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch3_related_work.tex) | 108 | Our R², RMSE in comparison table | R²=0.9999, RMSE=0.083 dB |
| [ch5_implementation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch5_implementation.tex) | 36 | Final dataset row count | 1,714,379 |
| [ch5_implementation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch5_implementation.tex) | 101–108 | Client sample counts + total | ED0:228,627 ED1:226,727 ED2:229,404 ED3:226,237 ED4:225,399 ED5:235,109 Total:1,371,503 |
| [ch6_evaluation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch6_evaluation.tex) | 82–84 | Three-way table: R², RMSE, MAE | Centralized: 1.0000 / 0.0007 / 0.0001; FL: 0.9999 / 0.083 / 0.002 |
| [ch8_conclusion.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch8_conclusion.tex) | 26, 37 | Summary + Key Findings R² values | 0.9999 vs 1.0000, 0.001% drop |

### Tier 2: Needs YOUR input (cannot auto-fill) — 12 placeholders

| File | Line(s) | What's needed | Notes |
|------|---------|---------------|-------|
| [main.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/main.tex) | 63–68 | Supervisor name, University, Department, Programme, Matrikelnr, Date | Personal info |
| [ch1_introduction.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch1_introduction.tex) | 39–41 | Prior project MLR & RF R²/RMSE; XGBoost RMSE | From your prior project report |
| [ch5_implementation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch5_implementation.tex) | 101–106 | Device **locations** (room names for ED0–ED5) | You know the physical placement |
| [ch6_evaluation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch6_evaluation.tex) | 140–143 | **Hardware measurements**: Flash/SRAM/arena/latency | Requires compiling firmware on MKR WAN 1310 |
| [ch8_conclusion.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch8_conclusion.tex) | 43 | Flash/SRAM utilization percentages | Same as above |

### Tier 3: Sections that are comment-only (need prose writing)

| File | Section | Status |
|------|---------|--------|
| [ch6_evaluation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch6_evaluation.tex) | §6.1 Eval Methodology | Comments only — no prose |
| [ch6_evaluation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch6_evaluation.tex) | §6.2 Centralized Baseline | Comments only — no prose |
| [ch6_evaluation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch6_evaluation.tex) | §6.3 FL Results (convergence, non-IID, hyperparams) | Comments only — no prose |
| [ch6_evaluation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch6_evaluation.tex) | §6.5 Communication Efficiency | Table done, analysis comments only |
| [ch7_discussion.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch7_discussion.tex) | §7.1–7.5 (ALL sections) | Entirely comments — zero prose written |
| [ch8_conclusion.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch8_conclusion.tex) | §8.1 Summary | Comment block only |
| [ch3_related_work.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch3_related_work.tex) | §3.1, §3.2 subsections, §3.3, §3.4 | Mostly comments with some prose |
| [ch5_implementation.tex](file:///c:/Users/prati/Desktop/edge%20AI/FederatedTinyML/thesis/chapters/ch5_implementation.tex) | Most subsections | Comments/outlines only |

## ⚠️ Additional Issues Noticed

1. **Model architecture inconsistency**: The chapter files reference `Dense(5→8→1)` with 57 params, but the simulation used `Dense(8→8→1)` with 81 params (8 features including RSSI, SNR, SF). This needs to be unified throughout.

2. **Communication table inconsistency**: ch6 says 2,404 B/day but JSON says 2,356 B/day.

3. **ch7 Limitations §7.6 line 105**: Says `Dense 5→8→1` — should be `Dense 8→8→1`.

4. **Figure references**: None of the chapter files contain `\includegraphics` commands yet — the 8 figures in `thesis_figures/` aren't referenced.

## 📋 Suggested Priority Order

1. **Fix architecture consistency** (5→8 everywhere should be 8→8)
2. **Auto-fill Tier 1 placeholders** from JSON data
3. **Write prose for Ch6 Evaluation** (most critical — results chapter)
4. **Write prose for Ch7 Discussion** (answers your RQs)
5. **Fill in Ch5 prose** (implementation details)
6. **Add figure references** (`\includegraphics` for all 8 figures)
7. **Fill Tier 2** with your personal/hardware info
8. **Polish Ch8, abstract, Ch3**
