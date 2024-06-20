SELECT 
    round((julianday(max(datetime)) - julianday(min(datetime))) * 86400, 2) AS difference_in_seconds
FROM 
    mlcheck_results;

