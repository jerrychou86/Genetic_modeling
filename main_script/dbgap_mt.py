import hail as hl
# Import the genoprep
import sys
# input where genoprep.py locate
sys.path.append('/space/chen-syn01/1/data/cchunju/dbgap/function')
from genoprep import GP2GTConverter, VCF2MT

# hail initiate
hl.init()
"""
hl.init(spark_conf={
    'spark.driver.memory': '64g',
    'spark.executor.memory': '64g',
    'spark.driver.cores': '5',
    'spark.executor.cores': '5'
})
"""

# define the paths
vcf_path = '/space/chen-syn01/1/data/database/dbGAP/phg000989.v1.MappingHumanConnectome_Marchini.genotype-imputed-data.c1.GRU-IRB-PUB/chr1.filtered.sampid.vcf'
new_vcf_path = '/space/chen-syn01/1/data/database/dbGAP/gt_processed/chr1.filtered.gt.vcf'
mt_path = '/space/chen-syn01/1/data/cchunju/dbgap/genomatrix/dbgap_chr1.mt'

# VCF2MT class initiate (default ref_genome is GRCh37, if not please input)
gp_converter = GP2GTConverter(vcf_path, new_vcf_path)
gp_converter.convert_gp_to_gt()

# Import vcf and output mt
converter = VCF2MT(new_vcf_path, mt_path)
converter.import_and_write_mt(overwrite=True)

# Stop Hail
hl.stop()