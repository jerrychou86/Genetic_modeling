import hail as hl
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

# Import the genoprep
import sys
# input where genoprep.py locate
sys.path.append('/space/chen-syn01/1/data/cchunju/Geno_pipeline/Function')
from genoprep import genoqc, mt_2_sample

# hail initiate (can change due to hardware)
hl.init(spark_conf={
    'spark.driver.memory': '64g',
    'spark.executor.memory': '64g',
    'spark.driver.cores': '5',
    'spark.executor.cores': '5'
})

# Start mt process
mt_path = '/space/chen-syn01/1/data/cchunju/dbgap/genomatrix/dbgap_chr1.mt'
processor = genoqc(mt_path)
mt = processor.load_mt()

# QC step
mt_qc = processor.apply_qc(mt)

# Annotate the MatrixTable with external pheno data
#file = '/space/chen-syn01/1/data/cchunju/Geno_pipeline/igsr_samples.txt'
#sample_column = 'Sample name'
#mt_annotated = processor.annotate_samples(mt_qc, file, sample_column)

# Check mt_annotated
#processor.print_summary(mt_annotated)

# Output
output_path = '/space/chen-syn01/1/data/cchunju/dbgap/genomatrix/dbgap_chr1_qced.mt'
processor.output_mt(mt_qc, output_path, overwrite=True)

# Sample output function call
mt_qc_path = '/space/chen-syn01/1/data/cchunju/dbgap/genomatrix/dbgap_chr1_qced.mt'
mt = hl.read_matrix_table(mt_qc_path)
transfer = mt_2_sample(mt)
output_dir = '/space/chen-syn01/1/data/cchunju/dbgap/sample_gt'

# Reference dictionary
transfer.output_dict(1, output_dir)

# Sample genotype data
#transfer.output_batches(chr=1, output_dir=output_dir, n_partitions=10)
transfer.export_sample_batches(chr=1, output_dir=output_dir, batch_size=100)

batch_dir = os.path.join(output_dir, "batches")
mt_2_sample.process_samples_from_batches(
    batch_dir=batch_dir,
    output_dir=output_dir,
    chr=1,
    n_jobs=5
)

# Stop Hail session
hl.stop()