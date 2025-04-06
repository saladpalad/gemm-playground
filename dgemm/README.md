`Usage: python3 test_dgemm.py`
- v1 only splits one matrix among 4 GPUs uses all-gather
- v2 splits 2 matrix among 4 GPUs uses all-reduce
