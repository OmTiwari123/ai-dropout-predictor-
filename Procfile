web: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 gunicorn api.app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2
