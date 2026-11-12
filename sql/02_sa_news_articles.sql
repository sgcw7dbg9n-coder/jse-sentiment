WITH companies AS (
  SELECT ticker, pattern
  FROM UNNEST(@tickers) AS ticker WITH OFFSET AS ticker_pos
  JOIN UNNEST(@patterns) AS pattern WITH OFFSET AS pattern_pos
    ON ticker_pos = pattern_pos
),

articles AS (
  SELECT
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(gkg.DATE AS STRING)) AS published_at,
    gkg.SourceCommonName AS domain,
    gkg.DocumentIdentifier AS url,
    REGEXP_EXTRACT(gkg.Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
    SAFE_CAST(SPLIT(gkg.V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS gdelt_tone,
    LOWER(IFNULL(gkg.Organizations, '')) AS orgs
  FROM `gdelt-bq.gdeltv2.gkg_partitioned` AS gkg
  WHERE gkg._PARTITIONTIME >= TIMESTAMP(@start_date)
    AND gkg._PARTITIONTIME < TIMESTAMP_ADD(TIMESTAMP(@end_date), INTERVAL 1 DAY)
    AND gkg.SourceCommonName IN UNNEST(@domains)
)

SELECT
  a.published_at,
  a.domain,
  a.url,
  a.title,
  a.gdelt_tone,
  c.ticker,
  REGEXP_CONTAINS(LOWER(IFNULL(a.title, '')), c.pattern) AS in_title
FROM articles AS a
JOIN companies AS c
  -- GDELT's entity extraction misses some names, so a headline match also counts
  ON REGEXP_CONTAINS(a.orgs, c.pattern)
  OR REGEXP_CONTAINS(LOWER(IFNULL(a.title, '')), c.pattern)
