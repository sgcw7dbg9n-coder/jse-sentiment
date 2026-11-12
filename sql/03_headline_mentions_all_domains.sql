WITH companies AS (
  SELECT ticker, pattern
  FROM UNNEST(@tickers) AS ticker WITH OFFSET AS ticker_pos
  JOIN UNNEST(@patterns) AS pattern WITH OFFSET AS pattern_pos
    ON ticker_pos = pattern_pos
),

headlines AS (
  SELECT
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(gkg.DATE AS STRING)) AS published_at,
    gkg.SourceCommonName AS domain,
    gkg.DocumentIdentifier AS url,
    REGEXP_EXTRACT(gkg.Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
    SAFE_CAST(SPLIT(gkg.V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS gdelt_tone
  FROM `gdelt-bq.gdeltv2.gkg_partitioned` AS gkg
  WHERE gkg._PARTITIONTIME >= TIMESTAMP(@start_date)
    AND gkg._PARTITIONTIME < TIMESTAMP_ADD(TIMESTAMP(@end_date), INTERVAL 1 DAY)
)

SELECT
  h.published_at,
  h.domain,
  h.url,
  h.title,
  h.gdelt_tone,
  c.ticker,
  TRUE AS in_title
FROM headlines AS h
JOIN companies AS c
  ON REGEXP_CONTAINS(LOWER(h.title), c.pattern)
-- One combined pattern first, so the per-company join only sees candidate headlines
WHERE REGEXP_CONTAINS(LOWER(h.title), @any_pattern)
