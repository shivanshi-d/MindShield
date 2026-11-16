## 2026-11-16 · Mon
State: extension code is in the repo and runs when loaded unpacked in Chrome. Both classifiers (ToS clauses, dark patterns) run in-browser with no backend. The parent companion app is not started. No client yet.
Did: committed the existing extension code to the repo; loaded it from the repo as an unpacked extension (chrome://extensions → Developer mode → Load unpacked) and confirmed it detects ToS clauses and dark patterns on test pages
Next: get a hello-world running in the companion app's stack; decide what the extension sends the app (only the single most severe dark-pattern category, never URLs or page content)
Blockers: none
