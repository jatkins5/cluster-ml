#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J nettest
#SBATCH -o logs/nettest_%j.out
curl -sS -o /dev/null -w "compute node -> lofar-surveys.org: HTTP %{http_code}\n" \
  --max-time 30 "https://lofar-surveys.org/" || echo "NO OUTBOUND NETWORK"
