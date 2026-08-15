# Attribution — Foorilla Data

Data pulled via this ingestion module comes from the Foorilla API and is
licensed under **CC BY-SA 4.0**:
https://creativecommons.org/licenses/by-sa/4.0/

## What this requires (per Foorilla's attribution guidelines: https://foorilla.com/api/)

1. **Give credit** — reference Foorilla as the data source anywhere this
   data (or a model/analysis derived from it) is published, presented, or
   deployed — including the dashboard.
2. **Link to the license** — include the CC BY-SA 4.0 URL above.
3. **Note changes** — if the data is cleaned, transformed, aggregated, or
   used to train a model, say so; don't imply Foorilla endorses the result.
4. **Share-alike** — adapted/derived datasets built from this data must be
   released under the same CC BY-SA 4.0 license if they're shared publicly.

## Suggested attribution line for the dashboard / report / README

> Salary and job posting data sourced from [Foorilla](https://foorilla.com/api/),
> licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
> Data has been cleaned, transformed, and used to train a predictive model;
> resulting outputs do not imply endorsement by Foorilla.

Every manifest written by `fetch.py` includes this notice under the
`"attribution"` key — don't strip it when handing CSVs off to teammates,
since whoever builds the dashboard/report needs it too.
