##get a script to convert vcf format to maf format
import pandas as pd
use_cols = ['CHROM', 'POS', 'REF', 'ALT']
maf = pd.read_csv('/home/janaya2/Desktop/tmb_paper/files/germline/new_data/tcga.somatic.maf',
                  sep='\t',
                  low_memory=False,
                  usecols=use_cols)
maf.drop_duplicates(inplace=True)

starts = []
ends = []
refs = []
alts = []

for index, row in enumerate(maf.itertuples()):
    position = 0
    position_ref = row.REF[position]
    position_alt = row.ALT[position]
    X = False
    while position_ref == position_alt:
        position += 1
        if position == len(row.REF):
            refs.append('-')
            alts.append(row.ALT[position:])
            starts.append(row.POS + position - 1)
            ends.append(row.POS + position)
            X = True
            break
        elif position == len(row.ALT):
            refs.append(row.REF[position:])
            alts.append('-')
            starts.append(row.POS + position)
            ends.append(row.POS + position + len(row.REF[position:]) - 1)
            X = True
            break
        else:
            position_ref = row.REF[position]
            position_alt = row.ALT[position]
    if not X:
        refs.append(row.REF[position:])
        alts.append(row.ALT[position:])
        starts.append(row.POS + position)
        ends.append(row.POS + position)


maf['new_ref'] = refs
maf['new_alt'] = alts
maf['start'] = starts
maf['end'] = ends


use_cols = ['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Tumor_Seq_Allele2']
original_maf = pd.read_csv('/home/janaya2/Desktop/pan-genie-cds-mc3.maf',
                  sep='\t',
                  low_memory=False,
                  usecols=use_cols
                  )
original_maf.drop_duplicates(inplace=True)
original_maf.sort_values(by=['Chromosome', 'Start_Position'], inplace=True)

merged = pd.merge(maf, original_maf, how='left', left_on=['CHROM', 'start', 'end', 'new_ref', 'new_alt'],
                  right_on=['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])


len(maf.drop_duplicates(subset=['CHROM', 'new_ref', 'new_alt', 'start', 'end']))

sum(pd.isna(merged)['Tumor_Seq_Allele2'])