# Submission

## Batch submission

To run them separately (recommended)
```
sbatch run_case.sh pre D0pp MBvspt_ntrkl
sbatch run_case.sh train D0pp MBvspt_ntrkl
sbatch run_case.sh apply D0pp MBvspt_ntrkl
```

To run everything in one go (not recommended)
```
./run_all.sh D0pp MBvspt_ntrkl
```

To run analysis (everything in one go)
```
./run_analysis.sh D0pp
```
