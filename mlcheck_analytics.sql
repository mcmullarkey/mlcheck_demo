with checks_passed as (
select
    file_name,
    group_id,
    avg(case when detected then 100.0 else 0.0 end) AS percentage_passed
from
    mlcheck_results
group by
    file_name,
    group_id
)

select
    round(min(percentage_passed), 2) as min_passed,
    round(avg(percentage_passed), 2) as mean_passed,
    round(max(percentage_passed), 2) as max_passed
from
    checks_passed;