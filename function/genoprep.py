import hail as hl
import pandas as pd
import numpy as np
import os
from collections import Counter
from math import log, isnan
from joblib import Parallel, delayed
# Memory & CPU 
import psutil

class VCF2MT:
    """
    Parameters:
    vcf_path: str
        Path to the input VCF file.
    mt_path: str
        Path where the MatrixTable (.mt) will be saved.
    reference_genome: str, optional (default='GRCh37')
        Reference genome version used for the VCF file.
    overwrite: bool, optional (default=False)
        Whether to overwrite the existing MatrixTable if it already exists at the specified 'mt_path'.
    Example:
    converter = VCFConverter(vcf_path, mt_path, reference_genome='GRCh38')
    converter.import_and_write_mt(overwrite=True)
    """
    def __init__(self, vcf_path, mt_path, reference_genome='GRCh37'):
        self.vcf_path = vcf_path
        self.mt_path = mt_path
        self.reference_genome = reference_genome
    
    def import_and_write_mt(self, overwrite=False):
        """Import VCF and save as Hail MatrixTable."""
        print("Importing VCF and writing to MatrixTable...")
        mt = hl.import_vcf(self.vcf_path, reference_genome=self.reference_genome)
        mt.write(self.mt_path, overwrite=overwrite)
        print(f"MatrixTable saved at {self.mt_path}")

# Remember to add for PLINK and bgen format
# mt2 = hl.import_bgen('/path/to/my.bgen')
# mt3 = hl.import_plink(bed='/path/to/my.bed', bim='/path/to/my.bim', fam='/path/to/my.fam')

class GP2GTConverter:
    """
    Converts genotype probabilities (GP) in a VCF to genotype calls (GT).
    
    Parameters:
    vcf_path: str
        Path to the input VCF file containing GP fields.
    output_vcf_path: str
        Path where the new VCF file with converted GT fields will be saved.
    reference_genome: str, optional (default='GRCh37')
        Reference genome version used for the VCF file.
    """

    def __init__(self, vcf_path, output_vcf_path, reference_genome='GRCh37'):
        self.vcf_path = vcf_path
        self.output_vcf_path = output_vcf_path
        self.reference_genome = reference_genome

    @staticmethod
    def gp_to_gt(gp):
        """Converts genotype probabilities (GP) to genotype calls (GT)."""
        return hl.if_else(
            hl.is_missing(gp),  # check if GP is missing
            hl.null(hl.tcall),  # return NULL if missing (true)
            hl.call(hl.argmax(gp) // 2, hl.argmax(gp) % 2)  # find max index, and diploid phase (else)
        )


    def convert_gp_to_gt(self, remove_gp=True):
        """Reads VCF, converts GP to GT, remove GP to save space (optional)
            and writes a new VCF file."""
        print(f"Importing VCF with GP from {self.vcf_path}...")
        mt = hl.import_vcf(self.vcf_path, reference_genome=self.reference_genome)
        
        # Convert GP to GT
        mt = mt.annotate_entries(GT=GP2GTConverter.gp_to_gt(mt.GP))
        
        # remove gp if ture
        if remove_gp:
            mt = mt.drop(mt.GP)
        
        # Write new VCF with converted GT field
        hl.export_vcf(mt, self.output_vcf_path)
        print("Conversion complete.")

class genoqc:
    def __init__(self, mt_path):
        self.mt_path = mt_path

    def load_mt(self):
        """Load MatrixTable from file."""
        print("Loading MatrixTable...")
        mt = hl.read_matrix_table(self.mt_path)
        return mt

    def apply_qc(self, mt, call_rate=0.95, AF=0.01, hwe=1e-6):
        """
        Apply Sample and Variant QC.
        Parameters:
            mt: MatrixTable - The Hail MatrixTable to apply QC on.
            call_rate: float - The minimum call rate for filtering variants (default is 0.95).
            AF: float - The minimum allele frequency for filtering variants (default is 0.01).
            hwe: float - The minimum Hardy-Weinberg Equilibrium p-value for filtering variants (default is 1e-6).
        
            Example:
            mt_qc = processor.apply_qc(mt) # All default
            mt_qc = processor.apply_qc(mt, call_rate=0.98, AF=0.005, hwe=1e-6)
        """
        print("Applying sample QC...")
        mt_qc = hl.sample_qc(mt)
        
        print("Applying variant QC...")
        mt_qc = hl.variant_qc(mt_qc)
        
        # Filter on MAF and HWE
        print("Filtering on call rate > 0.95 & MAF > 0.01 & HWE > 1e-6...")
        mt_qc = mt_qc.filter_rows(
            (mt_qc.variant_qc.call_rate > call_rate) &
            (mt_qc.variant_qc.AF[1] > AF) &
            (mt_qc.variant_qc.p_value_hwe > hwe)
        )
        
        return mt_qc

    def annotate_samples(self, mt, file, sample_column):
        """
        Annotate MatrixTable with external pheno data from a text file.

        Parameters:
        mt: MatrixTable - Hail MatrixTable to be annotated.
        file: str - Path to the file containing annotation data.
        sample_column: str - Column name in the annotation file for the sample IDs.

        Returns:
        MatrixTable - Annotated MatrixTable.
        """
        print(f"Importing annotation table from {file} and using '{sample_column}' as the key.")
        
        # Import the table and set the key to the provided sample_column
        table = (hl.import_table(file, impute=True).key_by(sample_column))
        
        # Annotate MatrixTable with the external table
        mt_annotated = mt.annotate_cols(annotation=table[mt.col_key])
        
        return mt_annotated
    
    def print_summary(self, mt):
        """Print sample and variant counts."""
        print('Samples: %d  Variants: %d' % (mt.count_cols(), mt.count_rows()))
    
    def output_mt(self, mt, output_path, overwrite=False):
        """Output results"""
        mt.write(output_path, overwrite=overwrite)
        print(f'Finish writing to {output_path}')

class mt_2_sample:
    def __init__(self, mt):
        """
        Initialize the converter class with the annotated MatrixTable.
        
        Parameters:
        mt_annotated: MatrixTable - The Hail MatrixTable containing annotated data.
        """
        self.mt = mt
        
    def output_dict(self, chr, output_dir):
        """
        Output the reference dictionary with locus, alleles, and rsID for all variants
        
        Parameters:
        chr: int - chromosome number of the dictionary
        output_dir: str - directory where the reference dictionary will be saved
        """
        print("Saving reference dictionary...")
        
        # Extract rows (locus, alleles, rsid) and export directly to a csv without reading (collect)
        ref_dir = os.path.join(output_dir, 'ref_dict')
        os.makedirs(ref_dir, exist_ok=True)

        reference_file = os.path.join(ref_dir, f"chr{chr}_reference_dict.csv")
        
        # Unkey the data rows
        rows_table = self.mt.rows()
        unkeyed_mt = rows_table.key_by() # removes the key restrict
        formatted_table = unkeyed_mt.select(
            locus=unkeyed_mt.locus,
            alleles=hl.delimit(unkeyed_mt.alleles, delimiter=" "),  # Format alleles as comma-separated string
            rsid=unkeyed_mt.rsid
        )
        formatted_table.export(reference_file, delimiter=",")
        print(f"Saved chr{chr} reference dictionary to {reference_file}")

    def output_sample_gt(self, sample, chr, output_dir):
        """
        Output the genotype data for each individual sample into a separate compressed CSV file.
        (could not work as parallel, therefore simple for small sample size testing)
        Parameters:
        sample_mt: MatrixTable - The filtered MatrixTable for the sample.
        sample: str - The sample ID.
        chr: str - The chromosome number.
        output_dir: str - Where sample genotype files will be saved.
        """
        print(f"Saving individual sample chr{chr} genotype data...")
        # Filter the MatrixTable for the specific sample
        sample_mt = self.mt.filter_cols(self.mt.s == sample)
        
        # Create sample directory
        sample_dir = os.path.join(output_dir, f"{sample}")
        os.makedirs(sample_dir, exist_ok=True)
        output_file = os.path.join(sample_dir, f"{sample}_chr{chr}.csv")

        # Simplify the entries by removing key and globals, selecting only GT
        sample_entries_table = sample_mt.entries().key_by().select_globals().select('GT')

        # Directly export the simplified GT-only table without extra columns
        sample_entries_table.export(output_file)
        print(f"Saved {sample}'s genotype data to {output_file}")
    

    def export_sample_batches(self, chr, output_dir, batch_size=10):
        """
        Export genotype data for batches of samples.

        Parameters:
        chr: int - Chromosome number.
        output_dir: str - Directory to save genotype data.
        batch_size: int - Number of samples in each batch.
        """
        print(f"Exporting genotype data for chromosome {chr} in batches of {batch_size} samples...")

        # Get all sample IDs
        sample_ids = [col.s for col in self.mt.cols().collect()]

        # Split sample IDs into batches
        batches = [sample_ids[i:i + batch_size] for i in range(0, len(sample_ids), batch_size)]

        # Create a batch directory
        batch_dir = os.path.join(output_dir, "batches")
        os.makedirs(batch_dir, exist_ok=True)
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)}...")

            # Filter the MatrixTable for the current batch
            print(f"Debug: Filtering for batch {batch}...")
            batch_mt = self.mt.filter_cols(hl.literal(batch).contains(self.mt.s))

            # Simplify the table by annotating entries and selecting required fields
            print("Debug: Simplifying the batch table...")
            batch_table = batch_mt.annotate_entries(
                sample_id=batch_mt.col_key.s  # Broadcast 's' to entries
            ).entries()

            # Unkey the table to allow modifications
            batch_table = batch_table.key_by()

            # Select the required fields
            batch_table = batch_table.select(
                sample_id=batch_table.sample_id,
                GT=hl.if_else(
                    hl.is_defined(batch_table.GT),
                    hl.str(batch_table.GT),
                    './.'
                )
            )

            # Export the batch
            batch_table.export(os.path.join(batch_dir, f"chr{chr}_batch_{i + 1}.tsv.bgz"))
            print(f"Saved batch {i + 1} to {batch_dir}")
    
    @staticmethod
    def process_samples_from_batches(batch_dir, output_dir, chr, n_jobs=5):
        """
        Separate samples from batch files into individual folders and files.

        Parameters:
        batch_dir: str - Directory containing batch files.
        output_dir: str - Directory to save individual sample files.
        chr: int - Chromosomes that are used.
        n_jobs: int - Number of parallel jobs.
        """
        import pandas as pd
        import pysam
        import os
        from joblib import Parallel, delayed
        import glob

        # Get all batch files
        batch_files = glob.glob(os.path.join(batch_dir, "*.tsv.bgz"))
        
        # Define a function to process a single batch
        """"
        def process_batch(batch_file):
            print(f"Processing batch {batch_file}...")

            with pysam.BGZFile(batch_file) as f:
                header = next(f).decode("utf-8").strip().split("\t")
                data = [line.decode("utf-8").strip().split("\t") for line in f]

            # Convert to pandas DataFrame
            batch_data = pd.DataFrame(data, columns=header)

            for sample_id in batch_data['sample'].unique():
                print(f"Processing sample {sample_id}...")
                sample_dir = os.path.join(output_dir, sample_id)
                os.makedirs(sample_dir, exist_ok=True)
                output_file = os.path.join(sample_dir, f"{sample_id}_chr{chr}.csv")

                # Filter data for the sample and save
                sample_data = batch_data[batch_data['sample'] == sample_id]
                sample_data[['row_key', 'GT']].to_csv(output_file, index=False)
                print(f"Saved {sample_id} to {output_file}")

        # Parallelize the processing of batches
        for batch_file in batch_files[:1]:  # Process only the first file
            process_batch(batch_file)
        #Parallel(n_jobs=n_jobs)(
        #    delayed(process_batch)(batch_file) for batch_file in batch_files
        #)
        """
        def process_batch(batch_file):
        try:
            print(f"Processing batch {batch_file}...")

            # Open and read using pysam
            with pysam.BGZFile(batch_file, "r") as f:
                print("Successfully opened file with pysam.")
                
                # Read first 5 lines to verify encoding
                for i, line in enumerate(f):
                    try:
                        decoded_line = line.decode("utf-8").strip()
                        print(f"Line {i}: {decoded_line}")  # Print each line
                        if i == 4:
                            break  # Stop after 5 lines
                    except UnicodeDecodeError as e:
                        print(f"Encoding error on line {i}: {e}")
                        return  # Stop processing this file

                # Read full file into dataframe
                f.seek(0)  # Reset file pointer
                header = next(f).decode("utf-8").strip().split("\t")
                data = [line.decode("utf-8").strip().split("\t") for line in f]

            # Convert to DataFrame
            print(f"Header: {header}")
            batch_data = pd.DataFrame(data, columns=header)

            # Ensure expected columns exist
            if "sample" not in batch_data.columns:
                print(f"Column 'sample' not found in {batch_file}. Skipping...")
                return

            # Process each sample
            for sample_id in batch_data["sample"].unique():
                print(f"Processing sample {sample_id}...")
                sample_dir = os.path.join(output_dir, sample_id)
                os.makedirs(sample_dir, exist_ok=True)
                output_file = os.path.join(sample_dir, f"{sample_id}_chr{chr}.csv")

                # Filter and save
                sample_data = batch_data[batch_data["sample"] == sample_id]
                sample_data[["row_key", "GT"]].to_csv(output_file, index=False)
                print(f"Saved {sample_id} to {output_file}")

        except Exception as e:
            print(f"Error processing {batch_file}: {e}")

        for batch_file in batch_files[:1]:  # Process only the first file
            process_batch(batch_file)


class gt_apps:
    def compute_pca(mt_dir, output_dir, num_pcs=10):
        """
        Compute (PCA) on genotype data.
        
        Parameters:
        mt_dir: Path to the qc_filtered Hail (.mt).
        output_dir: Save PCA results.
        num_pcs: Number of principal components.
        """
        mt = hl.read_matrix_table(mt_dir)

        print(f"Computing PCA with {num_pcs} components...")
        eigenvalues, pcs, _ = hl.hwe_normalized_pca(mt.GT, k=num_pcs)

        # Save PCA scores in the column annotation
        mt = mt.annotate_cols(scores=pcs[mt.s].scores)

        # Save PCA results
        pca_output = f"{output_dir}/pca_scores.ht"
        print(f"Saving PCA scores to {pca_output}...")
        mt.cols().write(pca_output, overwrite=True)

        return pca_output
    
    def gwas(mt_dir, trait_dir, pca_path, output_dir, covariates=['age', 'sex'], trait_col='phenotype', num_pcs=10):
        """
        Perform GWAS with additional traits and PCA-based population structure correction.
        
        Parameters:
        mt_dir: Path to the qc_filtered Hail (.mt).
        trait_dir: Path to the trait (phenotype) file.
        pca_path: Path to the stored PCA scores file (.ht).
        output_dir: Directory to save GWAS results.
        covariates: List of covariates to include in the model.
        trait_col (str): Column name for the phenotype in the trait file.
        num_pcs (int): Number of principal components to use as covariates (default: 10).
        """
        
        print("Loading qc_filtered matrixtable for GWAS...")
        mt = hl.read_matrix_table(mt_dir)

        print("Loading phenotype data...")
        trait_table = hl.import_table(trait_dir, impute=True, key='sample_id')

        # Annotate the MatrixTable with phenotype
        print("Annotating genotype data with phenotype...")
        mt = mt.annotate_cols(pheno=trait_table[mt.s])

        # Ensure all covariates exist in phenotype data
        for cov in covariates:
            if cov not in trait_table.row_value:
                raise ValueError(f"Covariate '{cov}' not found in the trait file!")

        # Annotate covariates
        for cov in covariates:
            mt = mt.annotate_cols(**{cov: trait_table[mt.s][cov]})

        # Load precomputed PCA scores
        print("Loading precomputed PCA scores...")
        pca_table = hl.read_table(pca_path)
        mt = mt.annotate_cols(scores=pca_table[mt.s].scores)

        # Construct the list of covariates including PCs
        pc_covariates = [mt.scores[i] for i in range(num_pcs)]
        full_covariates = [1.0] + [mt[cov] for cov in covariates] + pc_covariates

        # Run GWAS using linear regression
        print("Running GWAS...")
        gwas_results = hl.linear_regression_rows(
            y=mt.pheno[trait_col],
            x=mt.GT.n_alt_alleles(),
            covariates=full_covariates
        )

        # Export results
        output_file = f"{output_dir}/gwas_results.tsv.bgz"
        print(f"Saving GWAS results to {output_file}...")
        gwas_results.export(output_file)

        # Generate Manhattan & QQ plots
        print("Generating Manhattan and QQ plots...")
        p_manhattan = hl.plot.manhattan(gwas_results.p_value)
        p_qq = hl.plot.qq(gwas_results.p_value)

        return p_manhattan, p_qq

#/space/chen-syn01/1/data/database/HCP/100206/T1w/fsaverage_LR32k