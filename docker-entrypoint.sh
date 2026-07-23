#!/bin/bash
set -e

# Run your original setup
cd /opt/K-Radar/ops && python setup.py develop
cd /opt/K-Radar/utils/Rotated_IoU/cuda_op && python setup.py install
cd /opt/K-Radar

exec /bin/bash -c "while true; do sleep 1; done"
