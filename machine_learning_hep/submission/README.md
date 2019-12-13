# Submission

## Batch submission

To run them seperately (recommended)
```
sbatch run_case.sh pre D0pp
sbatch run_case.sh train D0pp
sbatch run_case.sh apply D0pp
```

To run everything in one go (not recommended)
```
./run_all.sh D0pp
```

To run analysis (everything in one go)
```
./run_analysis.sh D0pp
```
