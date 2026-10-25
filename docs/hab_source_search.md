# Verified HAB label search — 24 October 2026

## Decision

No suitable supervised HAB training catalogue was obtained in this search. **Zero records were accepted as training labels and zero monitored-negative zone-days were established.** This is a finite search result, not proof that suitable observations do not exist. The environmental period 2021–2025 is retained. No proxy labels, dates, coordinates, negative observations, species or performance results were manufactured.

Local inspection found 13 tabular/document/metadata files under `data`, consisting of sequence indices, coverage reports, environmental dates, satellite provenance and chlorophyll fitting metadata. Neither `data/verified_hab_records.csv` nor `data/hab_observations.csv` existed, and no HAB training run existed. Existing `.keras` checkpoints belong to the chlorophyll experiments.

## Sources actually inspected

| Source | Actual inspection and access | Suitability |
|---|---|---|
| [Kolluru & Nayak, Zenodo 14690930](https://zenodo.org/records/14690930) | Downloaded the actual 45,970-byte workbook and API metadata; verified the published MD5. Open CC-BY-4.0. Contains 218 numbered literature records; six fall in 2021–2025. | Three recent records explicitly name Kerala, and one additional record names the southwest coast. Records provide month/year and locality, not daily dates or coordinates; no monitored negatives. The authors explicitly disclaim authentication of original reports. Leads for follow-up, not accepted labels. |
| [Live HAEDAT India export](https://haedat.iode.org/events/export/filtered/csv/?country=57) | Downloaded unauthenticated CSV: 29 India events. Actual Event Date values range from 1908-01-01 to 2018-09-09. The event-name year and actual Event Date can disagree. | No actual event dates in 2021–2025. Grid centroids and location descriptions sometimes disagree. Fields such as “No high phytoplankton” do not establish monitored HAB absence. Raw export retained locally; no blanket redistribution license inferred from public web access. |
| [HAEDAT via OBIS](https://portal.obis.org/dataset/62ddad25-2a19-485d-9bae-7eb3a40a71c5) | Queried actual public API records for 74–78°E, 8–12°N. One all-years record: Chavakkad, 2018-09-09, with 10 km coordinate uncertainty; zero records for 2021–2025. OBIS dataset page states CC-BY-4.0. | Outside period; not an independent dense monitoring dataset. A trial `country=India` query was ignored by the API and returned other countries, so it was excluded from evidence; geographic results were checked explicitly. |
| [NCCR field study, DOI 10.3389/fmars.2026.1706039](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1706039/full) | Inspected public methods and station table. One November 2022 cruise, 12 stations, grouped into active, decay and non-bloom stations around the Gulf of Mannar/southern Kerala. The authors state that continuous temporal monitoring was not performed. | Useful field evidence; point reference stations cannot establish complete-zone daily absence. One linked bloom episode cannot support four independent chronological splits. Sampling south/east of the project's grid needs exclusion or separate inputs. No raw station-day monitoring series was obtained. |
| [Dias et al., DOI 10.1007/s10661-023-11278-7](https://link.springer.com/article/10.1007/s10661-023-11278-7) | Public abstract corroborates the October 2021 Kannur event using SEM and pigments. Publisher states datasets are available from the corresponding author on reasonable request; full text is subscription content. | Exact station-day data not publicly obtained. No purchase, login bypass or author message was attempted. |
| [INCOIS ABIS](https://incois.gov.in/site/services/hab.jsp) and [data holdings](https://incois.gov.in/site/dataholdings.jsp) | Inspected public service description, products and data-access listing. Bloom products are derived from ocean-colour and related environmental information. | Did not establish a downloadable, verified species/impact-labelled daily Kerala observation series with monitored negatives. Satellite bloom products cannot automatically become independent HAB truth for the same satellite predictors. |
| [NIOT WOSC 2024, abstract 415](https://www.niot.res.in/WOSC2024/documents/WOSC_2024_FINAL_Version_15.03.2024.pdf) | Official conference abstract describes KUFOS monitoring at Thikkodi through 2023, including recurrent blooms. | Strong lead for station sampling logs; an abstract is not the underlying observation catalogue or evidence of absence across a coastal zone. No correspondence sent. |
| [CMFRI/PIB August 2025 field notice](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2160852&lang=2&reg=48) | Government report describes mid-August field observations off Kochi. It distinguishes observed bioluminescence/red tide from demonstrated harm. | No exact daily station catalogue or monitored negatives. Publication date must not be substituted for sampling date. A species name or coloured water alone does not adjudicate the chosen HAB definition. |
| [Northern Indian Ocean hotspot dataset](https://zenodo.org/records/7969500) | Inspected repository description and listed archive contents. Dataset consists of chlorophyll statistics and thresholded chlorophyll groups. | Excluded as proxy environmental evidence, despite “HAB” in the title. Large archive not downloaded because it does not supply verified event labels. |

## Specific recoverable leads

The downloaded literature catalogue identifies these candidates; all remain **unaccepted** pending original-evidence review:

| Workbook row / catalogue ID | Reported month | Locality | Reported organism |
|---|---|---|---|
| 215 / 213 | January 2021 | Kochi estuary | *Akashiwo sanguinea* |
| 216 / 214 | October 2021 | Kannur | *Gonyaulax polygramma* |
| 218 / 216 | November 2021 | Kochi estuary | *Noctiluca scintillans* |
| 219 / 217 | December 2021 | Southwest coast, Arabian Sea | *Noctiluca scintillans* |

Month-long intervals must not be expanded into daily positive labels. Locality names must not be geocoded to falsely precise event points. Missing reports remain unknown. A site tested negative for one species is not necessarily negative for every HAB covered by a broader event definition.

## Exactly what is missing

1. A reviewed event definition identifying the species/abundance, harmful-impact or other adjudication evidence that counts as a positive.
2. Actual observation dates, defensible locations/uncertainty, event identities and supporting field/laboratory/source evidence for occurrences within the existing input grid.
3. Monitoring logs supporting negative observations over the declared target area/time. If observations are only at stations, either obtain zone-representative monitoring or explicitly redesign the target to monitored-site occurrence before fitting; do not silently promote station absence to zone absence.
4. Enough independent events and monitored negatives across training, validation, calibration and test to make the experiment meaningful. One event with many stations or repeated rolling windows is still one event.
5. Observation publication/availability timestamps for persistence and lead-time checks. Date-only literature reports need an explicit interval/uncertainty import policy; no midnight UTC timestamps were invented to satisfy the current strict CSV contract.

No outside messages, paid requests, external notifications or private-data publication occurred.

## Evidence and reproduction

Downloaded public files and retrieval URLs remain in ignored `data/raw/hab_source_audit_20261024/`. The source workbook is unchanged. The machine-readable [source audit](results/hab_submission/source_audit.json) contains hashes, counts and candidate references. Reproduce it without network access:

```bash
.venv/bin/python -m operations.hab_data_audit \
  --source-dir data/raw/hab_source_audit_20261024 \
  --output /tmp/new-hab-source-audit.json
```

The audit does not create a training CSV. A code-only clone does not include the downloaded files; retrieve the exact public versions from the recorded URLs and verify the published checksum before rerunning. No claim is made that source websites will remain unchanged.
