WITH sa_news AS (
  SELECT
    SourceCommonName AS domain,
    LOWER(Organizations) AS orgs
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2024-01-01')
    AND _PARTITIONTIME < TIMESTAMP('2026-07-01')
    AND SourceCommonName IN (
      'moneyweb.co.za',
      'businesslive.co.za',
      'news24.com',
      'fin24.com',
      'iol.co.za',
      'dailymaverick.co.za',
      'businesstech.co.za',
      'citizen.co.za',
      'timeslive.co.za',
      'techcentral.co.za',
      'mybroadband.co.za',
      'miningweekly.com',
      'engineeringnews.co.za',
      'bizcommunity.com'
    )
)
SELECT
  COALESCE(domain, 'TOTAL') AS domain,
  COUNT(*) AS total_articles,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bnaspers\b')) AS naspers,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bprosus\b')) AS prosus,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bmtn\b')) AS mtn,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bvodacom\b')) AS vodacom,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bsasol\b')) AS sasol,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bshoprite\b')) AS shoprite,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bwoolworths\b')) AS woolworths,
  -- FNB is FirstRand's retail brand, so FNB news counts toward FirstRand
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bfirstrand\b|\bfirst national bank\b')) AS firstrand,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bstandard bank\b')) AS standard_bank,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bcapitec\b')) AS capitec,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\babsa\b')) AS absa,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bnedbank\b')) AS nedbank,
  -- Also matches Anglo American Platinum (now Valterra); good enough for a coverage estimate
  COUNTIF(REGEXP_CONTAINS(orgs, r'\banglo american\b')) AS anglo_american,
  COUNTIF(REGEXP_CONTAINS(orgs, r'\bsibanye\b')) AS sibanye
FROM sa_news
GROUP BY ROLLUP(domain)
ORDER BY total_articles DESC;