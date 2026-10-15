# Sports Forecasting Intelligence — Claude Account Handoff
**Prepared:** 2026-10-15
**Status:** Architecture and requirements documented. No verified sports project source code, trained model, data warehouse, API credentials, or running deployment were found in the accessible material.

## Mission
Build an exceptionally rigorous, self-improving, multi-agent, **no-money** sports forecasting and analytics platform, initially for NFL, then NBA/MLB/NHL and other sports. Forecast outcomes with calibrated uncertainty; prioritize high-reliability findings and abstain when evidence is weak. The system is not a guarantee of winning or a risk-free outcome picker.

## Owner requirements from conversation
- Analyze game winners, predicted scores, point differentials, totals/overs/unders, passing/receiving/rushing yardage, receptions, touchdowns, other player and team statistics.
- Use prior-season and current-season data, advanced analytics, matchups, injuries, schedule, weather, player usage, and other relevant context.
- Compare outcomes by modeled probability, not perceived intuition. Avoid publishing weak, high-uncertainty forecasts; support abstention.
- Log every prediction and compare it to eventual reality; improve through governed retraining and out-of-sample evaluation.
- Use a central Claude orchestrator and specialized agents, but actual probabilities come from vetted statistical/ML models.
- Scale to a powerful agentic platform with automated data ingestion, simulations, analytics, and a dashboard.
- Initially proposed as sports forecasting rather than real-money wagering.

## What was discussed vs built
**Discussed:** proposed architecture, analysis categories, learning-loop design, technology stack, phased roadmap, and reliability controls.
**Not verified as built:** repositories, executable code, trained weights, production databases, historical datasets, live feeds, dashboards, and CI/CD. This is a transfer of plans, not a dump of existing implementation.

## Package map
- `CLAUDE_START_HERE.md`: ready-to-paste prompt for new Claude account.
- `PRODUCT_REQUIREMENTS.md`: scope, success criteria, detailed categories.
- `ARCHITECTURE.md`: components, agents, data, modeling and deployment.
- `FORECASTING_AND_EVALUATION.md`: statistical methodology, calibration, abstention and validation.
- `DATA_AND_SCHEMA.md`: required data sources, point-in-time contract and example event schema.
- `BUILD_ROADMAP.md`: staged deliverables and first implementation tasks.
- `PROJECT_STATUS.md`: verified status and open decisions.
- `config.example.yaml`: illustrative configuration, not verified production settings.

## Non-negotiable engineering rules
1. As-of timestamps: prohibit data leakage from future information.
2. Preserve raw source, extraction timestamp, event timestamp, model version and forecasts.
3. Use walk-forward testing and simple baselines before complex ensembles.
4. Probability calibration matters more than rhetorical confidence.
5. Do not silently self-modify production models; use champion/challenger evaluations.
6. Respect licensed data-provider terms, rate limits, provenance, monitoring and data quality checks.
7. Abstain rather than fabricate precision. Publish forecast interval and qualification status.
