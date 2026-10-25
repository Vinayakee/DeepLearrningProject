# Final integration checks — 13 September 2026

The running local API uses the separate final deployment pointer. The real HTTP smoke test passed root/health/dashboard HTML/CSS/JS, invalid-date validation (422), actual inference, all five forecast PNGs, cached normal/demo responses, alerts and evaluation. Its complete response record is saved in `results/improvement/http_smoke.json`.

In the existing in-app browser, the dashboard was reloaded from `http://127.0.0.1:8000/dashboard/` and tested through its controls:

- Historical input ending 2024-02-15 used actual observations from February 11–15 and produced five maps dated February 16–20. Day +1 and Day +5 were visibly verified.
- Coverage displayed 9,841 valid chlorophyll cell-days, a required minimum of 1,421, 34.63% mean coverage, four observed input days and the latest satellite observation date February 14. The two calendar channels and frozen score factors `[1.0000, 1.2840, 0.7788]` appeared.
- Day +1 showed 143 elevated pixels and Day +5 showed 142, with their respective raw model scores. Latest-day chlorophyll was correctly marked unobserved, while actual environmental means remained visible.
- Historical input ending 2024-03-26 had zero chlorophyll observations across its input window. The page explicitly displayed “Insufficient data for reliable prediction”, zero cell-days and “INSUFFICIENT DATA”; the previous map and forecast values were hidden.
- Browser console inspection reported no errors or warnings during these checks. JavaScript syntax and Git diff whitespace checks passed.

These checks verify software behavior, not scientific forecast reliability. The historical demonstration notice remains prominent, and operational warnings remain disabled because regional warning validation failed.
