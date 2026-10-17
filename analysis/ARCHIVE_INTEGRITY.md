# Raw archive integrity

The initial archive transport normalized CSV CRLF line endings to LF while preserving numerical values. The follow-up integrity commit restores the original CSV bytes; checksums.sha256 remains the original run digest. Compare SHA256 of raw file bytes, not a newline-normalizing text reader. .gitattributes disables CSV checkout normalization. Source commits, measurements, manifests and numerical findings are unchanged.
