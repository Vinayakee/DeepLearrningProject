# Bounded relaxed label comparison

| ID | P | Observed days | Max missing gap | Patch M | Train events | Train positive % | Val events | Val positive % | Test positives? | Test components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A | 75 | 1 | 0 | 2 | 4364 | 24.5290 | 847 | 10.4750 | yes | 951 |
| B | 80 | 1 | 0 | 2 | 3811 | 19.5655 | 680 | 8.3032 | yes | 789 |
| C | 75 | 2 | 1 | 2 | 1136 | 9.7678 | 112 | 2.1859 | yes | 60 |
| D | 80 | 2 | 1 | 2 | 937 | 7.1905 | 86 | 1.7400 | yes | 36 |
| E | 75 | 2 | 2 | 2 | 1210 | 10.8127 | 128 | 2.4172 | yes | 122 |
| F | 80 | 2 | 2 | 1 | 1340 | 8.1529 | 138 | 1.9493 | yes | 113 |

Chosen: A, the first qualifying definition. Selection was saved before test label inspection. No model performance was used.

Missing days are never labelled positive and never count toward duration. Components can bridge bounded missing gaps between observed endpoints; they are not independent biological events.
