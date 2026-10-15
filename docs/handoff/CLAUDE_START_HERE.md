# Paste this entire prompt into the new Claude account

I am transferring my **Sports Forecasting Intelligence** project from an earlier conversation. Treat the attached handoff folder as the authoritative product brief, **not** as proof that code already exists. Read all files before proposing implementation.

My goal is to build an exceptionally capable, self-improving, multi-agent sports analytics system, starting with the NFL, with the ability to expand to other leagues. I want forecasts for winners, expected final scores, point differentials, game totals, passing/rushing/receiving yards and other player and team statistical outcomes. Use prior seasons, the current season, advanced analytics, matchup context, injury reports, weather, scheduling, player usage, and every *relevant, trustworthy* factor. The priority is well-calibrated, high-reliability forecasts; the system must explicitly abstain when the data is sparse or uncertainty is too high. It is a no-money research/analytics project, not a promise of risk-free choices.

I want a Claude-based orchestrator coordinating specialized agents for data collection, stats, matchup, player usage, injuries/context, simulation, statistical validation, monitoring, and explanations. However, forecasts and probability estimates must come from reproducible statistical/ML models rather than unsupported LLM guesses. Maintain a persistent archive of every pre-event forecast (with as-of time, inputs, model version), score results after games, diagnose errors, train candidate models, walk-forward backtest, monitor calibration and promote model improvements only if validated.

Start by reading `README.md`, `PRODUCT_REQUIREMENTS.md`, `ARCHITECTURE.md`, `FORECASTING_AND_EVALUATION.md`, `DATA_AND_SCHEMA.md`, `BUILD_ROADMAP.md`, and `PROJECT_STATUS.md`. Then:
1. Summarize the product, verified current status, remaining unknowns and risks.
2. Propose the minimum working NFL implementation and repo tree.
3. Implement a reproducible skeleton incrementally if you're operating in an environment with file/code tools; include tests, a sample synthetic dataset and reproducible baseline forecast evaluations.
4. Do not assert that databases, API integrations, trained models or deployments already exist. Ask for actual repos/datasets/credentials only when required to proceed, never ask me to paste secret keys into chat.
5. Maintain `PROJECT_STATUS.md` as development progresses and distinguish simulated from real-world forecasts.

Preserve the core vision while avoiding hype: powerful agents, robust data and rigorous evaluations—not unverified claims of perfect accuracy.
