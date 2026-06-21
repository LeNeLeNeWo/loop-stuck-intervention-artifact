# Strongest Near-F1 Risk Pairs

The table below ranks detector pairs with absolute F1 difference <= 0.02 by largest PIIR gap, then HN-FPR gap. This is the clearest evidence that aggregate F1 can hide trigger-site risk.

## Pair 1: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=25`
- **Detector B**: `max_step_guard` / `threshold=200`
- **F1**: `0.003` vs `0.000` (delta `0.003`)
- **PIIR**: `1.000` vs `0.106` (delta `0.894`)
- **HN-FPR**: `1.000` vs `0.004` (delta `0.996`)
- **Burden**: `0.845` vs `0.003` (delta `0.841`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 2: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=50`
- **Detector B**: `max_step_guard` / `threshold=200`
- **F1**: `0.004` vs `0.000` (delta `0.004`)
- **PIIR**: `0.998` vs `0.106` (delta `0.891`)
- **HN-FPR**: `1.000` vs `0.004` (delta `0.996`)
- **Burden**: `0.683` vs `0.003` (delta `0.680`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 3: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=75`
- **Detector B**: `max_step_guard` / `threshold=200`
- **F1**: `0.005` vs `0.000` (delta `0.005`)
- **PIIR**: `0.976` vs `0.106` (delta `0.870`)
- **HN-FPR**: `0.994` vs `0.004` (delta `0.990`)
- **Burden**: `0.521` vs `0.003` (delta `0.518`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 4: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=100`
- **Detector B**: `max_step_guard` / `threshold=200`
- **F1**: `0.003` vs `0.000` (delta `0.003`)
- **PIIR**: `0.886` vs `0.106` (delta `0.780`)
- **HN-FPR**: `0.938` vs `0.004` (delta `0.934`)
- **Burden**: `0.367` vs `0.003` (delta `0.364`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 5: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=25`
- **Detector B**: `max_step_guard` / `threshold=150`
- **F1**: `0.003` vs `0.008` (delta `0.005`)
- **PIIR**: `1.000` vs `0.536` (delta `0.464`)
- **HN-FPR**: `1.000` vs `0.448` (delta `0.552`)
- **Burden**: `0.845` vs `0.133` (delta `0.712`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 6: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=50`
- **Detector B**: `max_step_guard` / `threshold=150`
- **F1**: `0.004` vs `0.008` (delta `0.004`)
- **PIIR**: `0.998` vs `0.536` (delta `0.461`)
- **HN-FPR**: `1.000` vs `0.448` (delta `0.552`)
- **Burden**: `0.683` vs `0.133` (delta `0.550`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 7: Enriched500

- **Detector A**: `max_step_guard` / `threshold=50`
- **Detector B**: `file_revisit_guard` / `window=5;repeats=3`
- **F1**: `0.719` vs `0.729` (delta `0.009`)
- **PIIR**: `0.101` vs `0.562` (delta `0.461`)
- **HN-FPR**: `0.118` vs `0.349` (delta `0.230`)
- **Burden**: `0.341` vs `0.454` (delta `0.113`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 8: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=75`
- **Detector B**: `max_step_guard` / `threshold=150`
- **F1**: `0.005` vs `0.008` (delta `0.003`)
- **PIIR**: `0.976` vs `0.536` (delta `0.440`)
- **HN-FPR**: `0.994` vs `0.448` (delta `0.546`)
- **Burden**: `0.521` vs `0.133` (delta `0.389`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 9: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=150`
- **Detector B**: `max_step_guard` / `threshold=200`
- **F1**: `0.008` vs `0.000` (delta `0.008`)
- **PIIR**: `0.536` vs `0.106` (delta `0.430`)
- **HN-FPR**: `0.448` vs `0.004` (delta `0.444`)
- **Burden**: `0.133` vs `0.003` (delta `0.129`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.

## Pair 10: OpenHandsExternal330

- **Detector A**: `max_step_guard` / `threshold=100`
- **Detector B**: `max_step_guard` / `threshold=150`
- **F1**: `0.003` vs `0.008` (delta `0.004`)
- **PIIR**: `0.886` vs `0.536` (delta `0.350`)
- **HN-FPR**: `0.938` vs `0.448` (delta `0.490`)
- **Burden**: `0.367` vs `0.133` (delta `0.234`)
- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.
