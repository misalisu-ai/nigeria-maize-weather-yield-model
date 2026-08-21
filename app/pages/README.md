# Dashboard Polish Bundle

Copy the included files into the model repository:

- `app/components/style.py`
- `.streamlit/config.toml`
- `docs/capstone_summary.md`
- `docs/submission_checklist.md`

Then, near the top of each Streamlit page, add:

```python
from components.style import apply_dashboard_style, sidebar_branding
apply_dashboard_style()
sidebar_branding()
```

On the homepage you can additionally use:

```python
from components.style import hero
hero(
    "Nigeria Maize Yield Intelligence",
    "Weather-based machine learning for state-level yield analysis, climate-stress sensitivity and explainable decision support.",
)
```
