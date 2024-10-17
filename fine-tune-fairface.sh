#!/bin/sh

# DATASET: Name of dataset from Dassl configs
# CFG: Config file stating the backbone model
# CTP: Class Token Position (end or middle)
# NCTX: Number of context tokens
# SHOTS: Number of shots (1, 2, 4, 8, 16)
# CSC: Class-specific context (False or True)

# bash scripts/coop/main.sh <DATASET> <CFG> <CTP> <NCTX> <SHOTS> <CSC>

bash scripts/coop/main.sh fairface rn50_ep50 middle 16 1 False
