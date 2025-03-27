#!/bin/bash

# break down into smaller scripts and use main script to call them

# to run the script checkout: https://laderast.github.io/bash_for_bioinformatics/
# dx run swiss-army-knife -icmd="bash /your_bash_script.sh" --instance-type desired_instance_type
# example: dx run swiss-army-knife    -icmd="dx download project-GxqpVq0Jpp5Py82xVbZV198y:/GWAS_pipeline/tools/GCTA_setup.sh -o GCTA_setup.sh && bash GCTA_setup.sh"    --instance-type mem1_ssd1_v2_x2    --destination /GWAS_pipeline/logs/    --brief

# ENV settings (update: seperate to another script - GCTA_setup.sh)
echo "Setting up gcta & PLINK 2.0..."
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/GWAS_pipeline/tools/tool_setup.sh" -o tool_setup.sh
chmod +x tool_setup.sh
bash tool_setup.sh

# Find the genotype data (use imputed genotype data)
#dx find data --folder "/NeuroGene/Bulk/Geno Calling/" --name "*.bed"
# example: -iin=project-BQbJpBj0bvygyQxgQ1800Jkk:file-FxXZzV0JkF65g2vX9Vx8jkkZ \
# project-GxqpVq0Jpp5Py82xVbZV198y
# file-FxXZzV0JkF65g2vX9Vx8jkkZ to ukb22418_c1_b0_v2.bed
echo "Downloading genotype data..."
mkdir -p /UKB_genotype_data
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/Bulk/Imputation/UKB imputation from genotype/ukb22828_c21_b0_v3.bgen" -o /UKB_genotype_data/partial_chr21.bgen
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/Bulk/Imputation/UKB imputation from genotype/ukb22828_c21_b0_v3.bgen.bgi" -o /UKB_genotype_data/partial_chr21.bgen.bgi
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/Bulk/Imputation/UKB imputation from genotype/ukb22828_c21_b0_v3.mfi.txt" -o /UKB_genotype_data/partial_chr21.mfi.txt
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/Bulk/Imputation/UKB imputation from genotype/ukb22828_c21_b0_v3.sample" -o /UKB_genotype_data/partial_chr21.sample

# Transform bgen to plink format
# filter SNPs with info score >= 0.9
echo "Filtering SNPs with info score >= 0.9..."
awk '$8 >= 0.9 {print $2}' /UKB_genotype_data/partial_chr21.mfi.txt > /UKB_genotype_data/high_info_snps.txt

# Transform bgen to plink format
echo "Transforming bgen to plink format..."
# Just the first 1k samples
tail -n +3 /UKB_genotype_data/partial_chr21.sample | head -n 1000 | awk '{print $1, $2}' > /UKB_genotype_data/subset_1k.txt
head /UKB_genotype_data/subset_1k.txt
wc -l /UKB_genotype_data/subset_1k.txt

plink2 --bgen /UKB_genotype_data/partial_chr21.bgen ref-first \
      --sample /UKB_genotype_data/partial_chr21.sample \
      --extract /UKB_genotype_data/high_info_snps.txt \
      --keep /UKB_genotype_data/subset_1k.txt \
      --make-bed \
      --out /UKB_genotype_data/partial_chr21

# Transforamtion check
echo "Checking PLINK output..."
ls -lh /UKB_genotype_data/partial_chr21.*
head -n 5 /UKB_genotype_data/partial_chr21.bim
head -n 5 /UKB_genotype_data/partial_chr21.fam
wc -l /UKB_genotype_data/partial_chr21.bim
wc -l /UKB_genotype_data/partial_chr21.fam

# Merge all chromosome data
# create merge list.txt
#plink --bfile /chr1 \
#      --merge-list merge_list.txt \
#      --make-bed \
#      --out /path_to_data/UKB_all

# Sample QC
plink2 --bfile /path_to_data/UKB_genotype_data/partial_chr21 \
      --mind 0.1 \
      --geno 0.05 \
      --maf 0.01 \
      --hwe 1e-6 \
      --make-bed \
      --out /path_to_data/UKB_genotype_data/partial_chr21_QCed

# Filter samples (to be updated)
#mkdir -p /UKB_demographics
#dx download "project-BQbJpBj0bvygyQxgQ1800Jkk:/EUR_sampeldata" -o /UKB_demographics/EUR_sampel
# filter execuetion
#plink --bfile partial_chr1 --keep EUR_sampel --out partial_EUR_chr1

# LD_pruned SNP list (could update to 1KG if outdated)
# Here, hapmap3.prune.in, a list of SNPs that passed LD pruning (indep-pairwise 1000 100 0.9), is used.
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/GWAS_pipeline/hapmap3.prune.in" -o hapmap3.prune.in
# Create sparse GRM
echo "Creating sparse GRM..."
mkdir -p /GRM
/tools/gcta-1.94.3-linux-kernel-3-x86_64/gcta64 \
      --bfile /UKB_genotype_data/partial_chr21 \
      --autosome \
      --extract hapmap3.prune.in \
      --make-grm \
      --sparse-cutoff 0.05 \
      --thread-num 10 \
      --out /GRM/test_sprs_grm

dx upload /GRM/test_sprs_grm.grm.bin --destination "project-BQbJpBj0bvygyQxgQ1800Jkk:/Test"
dx upload /GRM/test_sprs_grm.grm.id --destination "project-BQbJpBj0bvygyQxgQ1800Jkk:/Test"
dx upload /GRM/test_sprs_grm.grm.N.bin --destination "project-BQbJpBj0bvygyQxgQ1800Jkk:/Test"
dx upload /GRM/test_sprs_grm.grm.N.id --destination "project-BQbJpBj0bvygyQxgQ1800Jkk:/Test"

# import the phenotype file (file tag if necessary)
#dx download 

# fastGWA for UKB
#for i in `cat UKB42k_Chen_GWAS_DWI_Features_061223.txt`

#do /home/cchunju/bin/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1 \
#--bfile $path/genotype_QCed/UKB42krep_QCed_mind_geno_hwe_maf_041620 \
#--grm-sparse $path/GRM/sparse_UKB42krep_QCed_mind_geno_hwe_maf_041620_GRM \
#--fastGWA-mlm \
#--pheno $path/phenotypes/dwi/UKB42kChen_GWAS_DM_061223_Chen_FA_${i}.txt \
#--threads 10 \
#--out $path/phenotypes/dwi/sumstat/UKB42kChen_GWAS_DM_061223_Chen_FA_${i}_sumstat

#done
