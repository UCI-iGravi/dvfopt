| arm | setup s | iters | solve s | total s | rel resid | peak RSS MB | note |
|---|---|---|---|---|---|---|---|
| sa | 454 | 7 | 139 | 593 | 2.70e-05 | 30194 |  |
| rs_default | 369 | 9 | 181 | 550 | 6.61e-05 | 54264 |  |
| rs_classical | - | - | - | - | - | - | subprocess exit 1 |
| sa_cheap | 310 | 67 | 645 | 956 | 9.96e-05 | 41577 |  |
| gmg | 19 | 41 | 906 | 924 | 8.58e-02 | 11637 | solve capped at 900s budget |

**Winner (lowest total wall at rtol 0.0001): rs_default**

Arms that did not cleanly converge / were capped: gmg
