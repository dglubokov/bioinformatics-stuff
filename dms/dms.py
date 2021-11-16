import os
import re
import time
import json
import logging

import requests
import numpy as np
import pandas as pd
import hgvs
from hgvs import parser as hg_parser
import hgvs.dataproviders.uta
import pyhgvs
import pyhgvs.utils as pyhgvs_utils
from pyfaidx import Fasta

from mutant.classifier import add_mutant_annotation


FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('DMS')
logger.setLevel('INFO')

def startend(function):
    def wrapper(*args, **kwargs):
        logger.info(f'{function.__name__}: START')
        function(*args, **kwargs)
        logger.info(f'{function.__name__}: END')
    return wrapper


def mavedb_scoresets_variants_extractor(path, use_save):
    FILE_NAME = 'dms_variants.tsv'
    PATH_TO_FILE = os.path.join(path, FILE_NAME)
    if os.path.isfile(PATH_TO_FILE) and use_save:
        print('mavedb_scoresets_variants_extractor: Found')
    else:
        mavedb_scoresets = requests.get('https://www.mavedb.org/api/scoresets/').json()
        dms_variants = list()
        counter = 0
        for scoreset in mavedb_scoresets:
            for reference in scoreset['target']['reference_maps']:
                if reference['genome']['organism_name'] == 'Homo sapiens':
                    scoreset_variants = requests.get(
                        f"https://www.mavedb.org/scoreset/{scoreset['target']['scoreset']}",
                        headers={'X-Requested-With': 'XMLHttpRequest'}
                    ).json()

                    parameters_map = dict()
                    for column in scoreset_variants['columns']:
                        parameters_map[column['targets'][0]] = column['className']

                    pmids = ', '.join([pm['identifier'] for pm in scoreset['pubmed_ids']])

                    uniprot_id = np.nan
                    uniprot_offset = np.nan
                    uniprot = scoreset['target']['uniprot']
                    if uniprot is not None:
                        uniprot_id = uniprot['identifier']
                        uniprot_offset = uniprot['offset']
                        # Monkey fix of MaveDB косяк
                        if scoreset['target']['name'] == 'CXCR4':
                            uniprot_offset -= 1

                    ensembl_id = np.nan
                    ensembl_offset = np.nan
                    ensembl = scoreset['target']['ensembl']
                    if ensembl is not None:
                        ensembl_id = ensembl['identifier']
                        ensembl_offset = ensembl['offset']

                    refseq_id = np.nan
                    refseq_offset = np.nan
                    refseq = scoreset['target']['refseq']
                    if refseq is not None:
                        refseq_id = refseq['identifier']
                        refseq_offset = refseq['offset']

                    for variant_scores in scoreset_variants['data']:
                        dms_variant = {
                            'protein_name': scoreset['target']['name'],
                            'genome_version': reference['genome']['short_name'],
                            'pmids': pmids,
                            'uniprot_id': uniprot_id,
                            'uniprot_offset': uniprot_offset,
                            'ensembl_id': ensembl_id,
                            'ensembl_offset': ensembl_offset,
                            'refseq_id': refseq_id,
                            'refseq_offset': refseq_offset
                        }
                        for k, v in variant_scores.items():
                            dms_variant[parameters_map[int(k)]] = v
                        dms_variants.append(dms_variant)
            counter += 1
            print(f'scoresets: {counter}', end='\r')
        dms_variants_df = pd.DataFrame(dms_variants)
        dms_variants_df['uniprot_offset'] = dms_variants_df['uniprot_offset'].fillna(0)
        dms_variants_df = dms_variants_df.astype({'uniprot_offset': 'int32'})
        dms_variants_df.to_csv(PATH_TO_FILE, sep='\t', index=False)


class DMSAnalyzer:
    GENCODE = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
        }

    @startend
    def __init__(
        self, path_in, path_out,
        use_save: bool = False,
        genome_path: str = 'GCF_000001405.39_GRCh38.p13_genomic.fna',
        refseq_grch_json_path: str = 'pyhgvs_transcripts_refseq_grch38.json'
    ):
        self.path = path_out
        self.use_save = use_save

        all_dms = pd.DataFrame()
        for folder in os.listdir(path_in):
            protein_name = folder
            folder = os.path.join(path_in, folder)
            for file_name in os.listdir(folder):
                if 'urn_mavedb' in file_name:
                    path_to_file = os.path.join(folder, file_name)
                    mavedb_protein_variants = pd.read_csv(path_to_file)
                    mavedb_protein_variants = mavedb_protein_variants.assign(
                        **self.protein_variants_annotator(
                            protein_name=protein_name,
                        )
                    )
                    all_dms = all_dms.append(mavedb_protein_variants)
        all_dms.reset_index(drop=True).to_csv(os.path.join(path_out, 'dms_variants_raw.tsv'), sep='\t', index=False)

        gencode_reversed = dict()
        for k, v in self.GENCODE.items():
            if v not in gencode_reversed:
                gencode_reversed[v] = [k]
            else:
                gencode_reversed[v].append(k)
        self.GENCODE_REVERSED = gencode_reversed

        self.genome_path = genome_path
        self.refseq_grch_json_path = refseq_grch_json_path

    def protein_variants_annotator(self, protein_name: str):
        dms_variants_df = pd.read_table(os.path.join(self.path, 'dms_variants.tsv'), na_values='None')
        dms_proteins = dms_variants_df[['protein_name', 'genome_version', 'uniprot_id', 'uniprot_offset', 'ensembl_id']]
        protein_info = dms_proteins.drop_duplicates(subset='uniprot_id').reset_index(drop=True)
        return protein_info[protein_info['protein_name'] == protein_name].to_dict('records')[0]
    
    def mega_pipeline(self):
        self.ensg_transcript_by_uniprot()
        self.refseq_NM_NP_search_filter()
        self.splitter_integration()
        self.hgvsp_formater()
        self.refseq_id_mining()
        self.add_protein_codons_info()
        self.get_full_variants_info()
        self.mutant_maf_perapation()
        self.dms_and_mutant_scores()

    @startend
    def ensg_transcript_by_uniprot(self):
        """
        Майнинг ENSG транскриптов к соответствующим белкам через Uniprot.
        """
        PATH_TO_FILE = os.path.join(self.path, 'dms_ensembl_annotated.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_variants_raw = pd.read_table(os.path.join(self.path, 'dms_variants_raw.tsv'))

        transcripts_info = list()
        for uniprot_id in set(dms_variants_raw['uniprot_id'].values):
            # Get info about proteins
            response = requests.get(
                f'https://www.ebi.ac.uk/proteins/api/coordinates/{uniprot_id}',
                headers={'Accept': 'application/json'}
            )
            if response.ok:
                response = response.json()
                gene_name = response['gene'][0]['value']
                chrom = response['gnCoordinate'][0]['genomicLocation']['chromosome']
                protein_name = response['protein']['recommendedName']['fullName']
                ensg = response['gnCoordinate'][0]['ensemblGeneId']
                seq = response['sequence']

                transcripts_info.append({
                    'uniprot_id': uniprot_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'protein_name': protein_name,
                    'ensembl_id': ensg,
                    'protein_sequence': seq
                })
        transcripts_info_df = pd.DataFrame(transcripts_info)
        dms_vars_ens = dms_variants_raw.drop(columns=['ensembl_id', 'protein_name'])
        dms_vars_ens = dms_vars_ens.merge(transcripts_info_df, on=['uniprot_id'])
        dms_vars_ens.to_csv(PATH_TO_FILE, sep='\t', index=False)

    @startend
    def refseq_NM_NP_search_filter(self):
        """
        RefSeq NP & NM search and filter.
        """
        PATH_TO_FILE = os.path.join(self.path, 'dms_refseq_annotated.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_ensembl_annotated = pd.read_table(os.path.join(self.path, 'dms_ensembl_annotated.tsv'))

        refseq_info = list()
        for gene_name in set(dms_ensembl_annotated['gene_name'].to_list()):
            # Get info about NP and NM
            response = requests.get(
                f'https://www.ncbi.nlm.nih.gov/nuccore/?term={gene_name}')
            if response.ok:
                NM_id = re.findall(r'NM_\d*\.\d', response.text)[0]
                try: 
                    NM_id = re.findall(r'NM_\d*\.\d', response.text)[0]
                    NP_id = re.findall(r'NP_\d*\.\d', response.text)[0]

                    refseq_info.append({
                        'NM_id': NM_id,
                        'NP_id': NP_id,
                        'gene_name': gene_name
                    })
                except:
                    print('Failed: ', gene_name, )
            else:
                print('Reponse failed :( ', gene_name, response.status_code, response.reason)
            time.sleep(2)

        refseq_info_for_every_gene_name = pd.DataFrame(refseq_info)
        
        dms_vars_refseq = dms_ensembl_annotated.merge(refseq_info_for_every_gene_name, how='left', on='gene_name')
        dms_vars_refseq = dms_vars_refseq.dropna(subset=['NP_id', 'hgvs_pro'])
        dms_vars_refseq = dms_vars_refseq[dms_vars_refseq['hgvs_pro'].apply(lambda x: '=' not in x)]
        dms_vars_refseq = dms_vars_refseq[dms_vars_refseq['hgvs_pro'].apply(lambda x: 'p.' in x)]
        dms_vars_refseq = dms_vars_refseq.reset_index(drop=True)
        dms_vars_refseq.to_csv(PATH_TO_FILE, sep='\t', index=False)

    @startend
    def splitter_integration(self):
        """Расширить таблицу если в hgvs_pro записано больше одного варианта."""
        PATH_TO_FILE = os.path.join(self.path, 'dms_splited.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)

        dms_refseq_annotated = pd.read_table(os.path.join(self.path, 'dms_refseq_annotated.tsv'))

        def hgvsp_splitter(row):
            hgvsp = row['hgvs_pro'].split('.')
            if len(hgvsp) > 1:
                ass = hgvsp[1].strip('[]').split(';')
                ass_hgvsp = list()
                for aa in ass:
                    ass_hgvsp.append('p.' + aa)
                row['hgvs_pro_modified'] = ass_hgvsp
                return row

        dms_vars_splited = dms_refseq_annotated.apply(
            hgvsp_splitter, axis=1, result_type='expand'
        ).explode('hgvs_pro_modified', ignore_index=True)
        dms_vars_splited.to_csv(PATH_TO_FILE, sep='\t', index=False)

    @startend
    def hgvsp_formater(self, validate: bool = False):
        PATH_TO_FILE = os.path.join(self.path, 'dms_prepared.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_splited = pd.read_table(os.path.join(self.path, 'dms_splited.tsv'))

        hgvs_pro_positions = dms_splited['hgvs_pro_modified'].str.findall('\d+').apply(
            lambda x: int(x[0]) if len(x) > 0 else np.nan)

        dms_variants_all = dms_splited.copy(deep=True)
        dms_variants_all['hgvs_true_postions'] = hgvs_pro_positions + dms_variants_all['uniprot_offset']

        def change_type(row):
            if pd.notna(row['hgvs_true_postions']):
                return str(int(row['hgvs_true_postions']))
            return row['hgvs_true_postions']
        dms_variants_all['hgvs_true_postions'] = dms_variants_all.apply(change_type, axis=1)
        
        def replacer(row):
            if pd.notna(row['hgvs_true_postions']) and pd.notna(row['hgvs_pro_modified']):
                return {'hgvs_pro_replaced': re.sub(r'\d+', row['hgvs_true_postions'], row['hgvs_pro_modified'])}
            return np.nan

        dms_variants_all['hgvs_pro_replaced'] = dms_variants_all.apply(replacer, result_type='expand', axis=1)
        dms_variants_all['NP_offset_hgvs'] = dms_variants_all['NP_id'] + ':' + dms_variants_all['hgvs_pro_replaced']

        if validate:
            hdp = hgvs.dataproviders.uta.connect()
            hp = hg_parser.Parser()
            vr = hgvs.validator.Validator(hdp=hdp)
            def hgvs_validator(x):
                if pd.notna(x):
                    try:
                        vr.validate(hp.parse_hgvs_variant(x))
                        time.sleep(0.1)
                        return True
                    except hgvs.exceptions.HGVSError as e:
            #             print(e)
                        return False
                return False

            dms_variants_all = dms_variants_all[
                dms_variants_all['NP_offset_hgvs'].apply(hgvs_validator)
            ].reset_index(drop=True)
        dms_variants_all.to_csv(PATH_TO_FILE, sep='\t', index=False)

    @startend
    def refseq_id_mining(self):
        PATH_TO_FILE = os.path.join(self.path, 'dms_full_refseq.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_prepared = pd.read_table(os.path.join(self.path, 'dms_prepared.tsv'))
        dms_full_refseq = self._nm_mining(dms_prepared)
        dms_full_refseq = self._np_mining(dms_full_refseq)

        dms_full_refseq.to_csv(PATH_TO_FILE, sep='\t', index=False)

    def _nm_mining(self, dms_prepared):
        """
        NM ID mining and mrna reference sequence search in RefSeq.
        """
        all_mrna_seqs = list()
        for nm_id in set(dms_prepared['NM_id'].values):
            r = requests.get(
                f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={nm_id}&rettype=fasta'
            )
            result = re.findall(r'[ACGT]+', r.text)
            seqs = list()
            for i in range(len(result)-1):
                if len(result[i]) == 70:
                    seqs.append(result[i])
            seqs.append(result[-1])
            mrna_seq = ''.join(seqs)

            # Check for errors
            if len(mrna_seq) < 30:
                print(r.text)
                break

            all_mrna_seqs.append({
                'NM_id': nm_id,
                'mRNA_seq': mrna_seq
            })
            time.sleep(1)
        all_mrna_seqs_df = pd.DataFrame(all_mrna_seqs)
        vars_with_seqs = dms_prepared.merge(all_mrna_seqs_df, on=['NM_id'])
        return vars_with_seqs

    def _np_mining(self, dms_prepared):
        """
        NP ID mining and protein reference sequence search in RefSeq.
        """
        all_aa_seqs = list()
        for np_id in set(dms_prepared['NP_id'].values):
            r = requests.get(
                f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={np_id}&rettype=fasta'
            )
            result = re.findall(r'[ACDEFGHIKLMNPQRSTVWY]+', r.text)
            seqs = list()
            for i in range(len(result)-1):
                if len(result[i]) == 70:
                    seqs.append(result[i])
            seqs.append(result[-1])
            aa_seq = ''.join(seqs)

            # Check for errors
            if len(aa_seq) < 30:
                print(aa_seq)
                print(r.text)
                break

            all_aa_seqs.append({
                'NP_id': np_id,
                'AA_seq': aa_seq
            })
            time.sleep(1)
        all_aa_seqs_df = pd.DataFrame(all_aa_seqs)
        vars_with_seqs = dms_prepared.merge(all_aa_seqs_df, on=['NP_id'])
        return vars_with_seqs

    @startend
    def add_protein_codons_info(self):
        PATH_TO_FILE = os.path.join(self.path, 'dms_with_codons.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_full_refseq = pd.read_table(os.path.join(self.path, 'dms_full_refseq.tsv'))
 
        seqs_info = list()
        for nm_id, rows in dms_full_refseq.groupby('NM_id'):
            # Find nucleotide reference
            aa_reference = rows['AA_seq'].iloc[0]
            mrna_seq = rows['mRNA_seq'].iloc[0]
            for match_obj in re.finditer('ATG', mrna_seq):
                end_nucleotide_start = match_obj.span()[1]
                slice_seq = mrna_seq[end_nucleotide_start:]
                codons = re.findall('...', slice_seq)
                aa_found = 'M'
                nucleotides = 'ATG'
                for i in range(len(codons)):
                    if i+2 > len(aa_reference) or self.GENCODE[codons[i]] != aa_reference[i+1]:
                        break
                    else:
                        aa_found += self.GENCODE[codons[i]]
                        nucleotides += codons[i]
                if aa_found == aa_reference:
                    break
            assert aa_found == aa_reference
            seqs_info.append({
                'NM_id': nm_id,
                'AA_seq_found': aa_found,
                'AA_codons': nucleotides,
            })

        variants_all_info = dms_full_refseq.merge(pd.DataFrame(seqs_info), on='NM_id')
        variants_all_info.to_csv(PATH_TO_FILE, sep='\t', index=False)

    def init_reference_info(self):
        """
        The NM_sequences don't necessarily have a "true" position on GRCh37.
        They are derived independently of the reference genome,
            so may or may not map to the genome (and when they do map, they might not map 100%).
        This means that any means of mapping these NM sequences
            to the genome is just one possible mapping out of many potential mappings.
        Also, keep in mind that there are cases
            where a single NM gene could map to multiple locations on the genome,
            since the genome has several copies of some genes.
        https://github.com/counsyl/hgvs/issues/60
        """
        self.genome = Fasta(self.genome_path)

        with open(self.refseq_grch_json_path) as f:
            self.pyhgvs_transcripts_38 = json.load(f)

    @startend
    def get_full_variants_info(self):
        PATH_TO_FILE = os.path.join(self.path, 'dms_with_vcf_info.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_with_codons = pd.read_table(os.path.join(self.path, 'dms_with_codons.tsv'))
        self.init_reference_info()

        protein_letters_1to3 = {
            "A": "Ala",
            "C": "Cys",
            "D": "Asp",
            "E": "Glu",
            "F": "Phe",
            "G": "Gly",
            "H": "His",
            "I": "Ile",
            "K": "Lys",
            "L": "Leu",
            "M": "Met",
            "N": "Asn",
            "P": "Pro",
            "Q": "Gln",
            "R": "Arg",
            "S": "Ser",
            "T": "Thr",
            "V": "Val",
            "W": "Trp",
            "Y": "Tyr",
            "*": "Ter",
        }

        protein_letters_3to1 = {v: k for k, v in protein_letters_1to3.items()}

        def variant_info_extractor(row):
            # Init.
            p_ref = p_alt = p_position = np.nan
            hgvs_nm_c_variant = hgvs_name_parsed = np.nan
            chrom = offset = ref = alt = np.nan

            # Find all needed information.
            hgvs_pro = row['hgvs_pro_replaced']
            aa_found = row['AA_seq_found']
            if 'fs' not in hgvs_pro and 'del' not in hgvs_pro:
                try:
                    p_ref_3, p_alt_3 = re.findall(r'[a-zA-Z]{3}', hgvs_pro)
                except ValueError as e:
                    print(hgvs_pro)
                    raise e
                p_ref = protein_letters_3to1[p_ref_3]
                p_alt = protein_letters_3to1[p_alt_3]
                p_position = int(re.findall(r'\d+', hgvs_pro)[0])
                # print(row['protein_name'], row['AA_seq'][p_position-1], p_ref, p_position, hgvs_pro)
                try:
                    assert p_ref == row['AA_seq'][p_position-1]
                except AssertionError as e:
                    print(p_ref, p_position, row['AA_seq'][p_position-1], row['AA_seq'], row['uniprot_id'], row['NP_id'], row['NM_id'])
                    raise e
                nucl_positions = {
                    1: p_position * 3 - 3,
                    2: p_position * 3 - 2,
                    3: p_position * 3 - 1,
                }
                nucleotides = row['AA_codons']
                ref_codon = ''.join([nucleotides[pos] for pos in nucl_positions.values()])

                if ref_codon in self.GENCODE_REVERSED[p_ref] or aa_found[p_position-1] == p_ref:
                    # Find most relevant alt codon.
                    changes = dict()
                    for alt_codon in self.GENCODE_REVERSED[p_alt]:
                        change_counter = 0
                        for ref_nucl, alt_nucl in zip(ref_codon, alt_codon):
                            if ref_nucl != alt_nucl:
                                change_counter += 1
                        changes[alt_codon] = change_counter

                    selected_alt_codon = None
                    for k, v in changes.items():
                        if v == 1:
                            selected_alt_codon = k
                            break
                    if selected_alt_codon is not None:
                        # Generate NM variant.
                        nucl_positions = {
                            1: p_position * 3 - 2,
                            2: p_position * 3 - 1,
                            3: p_position * 3,
                        }
                        hgvs_nm_c_variant = None
                        nucl_counter = 0
                        for ref_nucl, alt_nucl in zip(ref_codon, selected_alt_codon):
                            nucl_counter += 1
                            if ref_nucl != alt_nucl:
                                affected_poistion = nucl_positions[nucl_counter]
                                hgvs_nm_c_variant = row['NM_id'] + ':c.' + str(affected_poistion) + ref_nucl + '>' + alt_nucl
                                break
                        assert hgvs_nm_c_variant


                        # Get genome position.
                        transcript = self.pyhgvs_transcripts_38['transcripts'].get(hgvs_nm_c_variant.split(':')[0])
                        transcript_object = pyhgvs_utils.make_transcript(transcript)
                        chrom, offset, ref, alt = pyhgvs.parse_hgvs_name(hgvs_nm_c_variant, self.genome, transcript=transcript_object)

                        hgvs_name_parsed = pyhgvs.format_hgvs_name(chrom, offset, ref, alt, self.genome, transcript_object)

                        assert hgvs_nm_c_variant.split(':')[1] == hgvs_name_parsed.split(':')[1]

            return {
                'NP_offset_hgvs_test': row['NP_offset_hgvs'],
                'p_REF': p_ref,
                'p_ALT': p_alt,
                'p_POSITION': p_position,
                'NM_hgvs_found': hgvs_nm_c_variant,
                'NM_hgvs_parsed': hgvs_name_parsed,
                'chrom_id': chrom,
                'genome_position': offset,
                'REF': ref,
                'ALT': alt,
            }

        found_ref_alt = dms_with_codons.apply(variant_info_extractor, axis=1)
        found_ref_alt = pd.DataFrame(found_ref_alt.to_list())

        def error_finder(row):
            if pd.notna(row['NM_hgvs_found']):
                if row['NM_hgvs_found'].split(':')[1] != row['NM_hgvs_parsed'].split(':')[1]:
                    return True
            return False

        assert len(found_ref_alt[found_ref_alt.apply(error_finder, axis=1)]) == 0
        assert all(dms_with_codons['NP_offset_hgvs'] == found_ref_alt['NP_offset_hgvs_test'])

        found_ref_alt = found_ref_alt.drop(columns=['NP_offset_hgvs_test'])
        the_end_df = pd.concat([dms_with_codons, found_ref_alt], axis=1)
        the_end_df.to_csv(PATH_TO_FILE, sep='\t', index=False)

    @startend
    def mutant_maf_perapation(self):
        PATH_TO_FILE = os.path.join(self.path, 'dms_for_mutant.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_with_vcf_info = pd.read_table(os.path.join(self.path, 'dms_with_vcf_info.tsv'))
        dms_with_vcf_info.astype({'chrom': 'str'})

        the_end_mutant = dms_with_vcf_info.dropna(subset=['REF', 'ALT', 'genome_position']).reset_index(drop=True)
        the_end_mutant = the_end_mutant.rename(columns={
            'genome_position': 'Start_Position',
            'chrom': 'Chromosome',
            'hgvs_pro_replaced': 'HGVSp',
            'REF': 'Reference_Allele',
            'ALT': 'Tumor_Seq_Allele2'
        })
        the_end_mutant = the_end_mutant[[
            'uniprot_id',
            'hgvs_pro',
            'Chromosome',
            'Start_Position',
            'Reference_Allele',
            'Tumor_Seq_Allele2',
            'HGVSp',
            'score',
        ]]
        the_end_mutant['End_Position'] = the_end_mutant['Start_Position']
        the_end_mutant['Variant_Type'] = 'SNP'
        
        the_end_mutant.to_csv(PATH_TO_FILE, sep='\t', index=False)

    @startend
    def dms_and_mutant_scores(self):
        PATH_TO_FILE = os.path.join(self.path, 'dms_mutant_annotated.tsv')
        
        if os.path.isfile(PATH_TO_FILE) and self.use_save:
            return pd.read_table(PATH_TO_FILE)
        
        dms_for_mutant = pd.read_table(os.path.join(self.path, 'dms_for_mutant.tsv'))

        temp_df = add_mutant_annotation(
            dms_for_mutant,
            file_type='maf',
            genome_version='hg38',
            development_mode=False,
        )
        temp_df_filtered = temp_df.drop_duplicates(subset=['HGVSp', 'Start_Position'], ignore_index=True)
        temp_df_filtered['score_normalized'] = (
            temp_df_filtered['score']-temp_df_filtered['score'].min()
        )/(temp_df_filtered['score'].max()-temp_df_filtered['score'].min())
        temp_df_filtered.to_csv(PATH_TO_FILE, sep='\t', index=False)


mavedb_scoresets_variants_extractor('./results/', use_save=True)
dms_analyzer = DMSAnalyzer(path_in='./proteins/', path_out='./results/', use_save=False)
# dms_analyzer.mega_pipeline()
# dms_analyzer.get_full_variants_info()
dms_analyzer.mutant_maf_perapation()
dms_analyzer.dms_and_mutant_scores()
