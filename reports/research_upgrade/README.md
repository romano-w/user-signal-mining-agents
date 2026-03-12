# Research Upgrade Reports

Place integration-gate input JSON files here:

- `schema_compatibility.json`
- `retrieval_eval_summary.json`
- `robustness_report.json`
- `domain_transfer_report.json`
- `failure_tags_report.json`

Run gate:

```powershell
uv run usm integration-gate --reports-dir reports/research_upgrade
```
