# System Statuses

Here are the various statuses that are assigned to jobs at different stages.  These are marked in the database.  Setting the status in the `jobs` table should be a reflection of what's going on with that job's chunks in the `chunks` table, but that relationship could be improved upon in the code. 

## Phedf (jobs Table):

| Status | Description |
| :-------- | :-------- |
| pending | initial state when the job is submitteed from the website into the database |
| reading | when the phdef readers start for any chunk |
| fortracc | when FortraCC begins work |
| plotting | when plotting begins |
| complete | when the plotting is done |
| failed | error somewhere a long the way |


## Phedf (chunks Table):

| Status | Description |
| :-------- | :-------- |
| pending | initial state after the job is submitted from the website and the chunker chunks the jobs in the DB |
| reading | when the readers begin reading the chunk's data |
| complete | when the reading is done, ending chunked operations for PhDef |
| failed | error somewhere along the way |


## Curation (jobs Table):

| Status | Description |
| :-------- | :-------- |
| pending | initial state when the job is submitted from the website to the database |
| running | when the curator starts running for any chunk |
| complete | when the stitched interpolated file is done |
| failed | error somewhere along the way |


## Curation (chunks Table):

| Status | Description |
| :-------- | :-------- |
| pending | initial state after the job is submitted from the website and the chunker chunks the jobs in the DB |
| subsetting | when the subsetters begin work on the chunk's data |
| interpolating | when interpolation begins on a chunk's data |
| complete | when the stitched interpolated file is done |
| failed | error somewhere along the way |