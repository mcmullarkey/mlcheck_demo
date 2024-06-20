# mlcheck Demo

This repo is a brief demo of the <a href="https://github.com/mcmullarkey/mlcheck" target="_blank">`mlcheck` command line tool</a> built by Michael Mullarkey using Rust.

Think of `mlcheck` as a spell-check equivalent for ML best practices.

This repo contains:
- A bash script –`download_py_github.sh`– for downloading code files from Github using the Github CLI tool `gh` to the directory `downloaded_files/`. 
  - The files must end in .py and contain "sklearn"
- The ouput of running `mlcheck` on the `downloaded_files/` directory in the sqlite database `mlcheck_output.db`
- A SQL query –`mlcheck_analytics.sql`– for calculating the minimum, mean, and maximum percentage of best-practice checks passed across all the .py files.

See the <a href="https://github.com/mcmullarkey/mlcheck" target="_blank">`mlcheck` repo</a> for installation and usage instructions.