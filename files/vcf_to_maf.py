##if you made a maf from a VCF it likely isn't in the correct format.
##A VCF never has empty alleles while a MAF does.
##This script will convert a left normalized VCF to MAF format.

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