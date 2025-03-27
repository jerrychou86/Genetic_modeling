#!/bin/bash

# Download GCTA and unzip
echo "downloading gcta..."
mkdir -p "/tools"
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/GWAS_pipeline/tools/gcta-1.94.3-linux-kernel-3-x86_64.zip" -o /tools/gcta.zip

echo "unzipping gcta..."
unzip /tools/gcta.zip -d "/tools"
chmod +x "/tools/gcta-1.94.3-linux-kernel-3-x86_64/gcta64"
echo 'export PATH="/tools/gcta-1.94.3-linux-kernel-3-x86_64:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Download PLINK2.0 and unzip
echo "downloading plink2.0..."
dx download "project-GxqpVq0Jpp5Py82xVbZV198y:/GWAS_pipeline/tools/plink2_linux_avx2_20250129.zip" -o /tools/plink2.zip

echo "unzipping plink..."
unzip /tools/plink2.zip -d "/tools"
chmod +x "/tools/plink2/plink2"
echo 'export PATH="/tools/plink2:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify GCTA installation
/tools/gcta-1.94.3-linux-kernel-3-x86_64/gcta64 --help
echo "gcta installation complete"
# Verify PLINK2.0 installation
plink2 --help
echo "plink 2.0 installation complete"