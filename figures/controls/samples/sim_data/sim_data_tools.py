import numpy as np
from Bio.Seq import Seq

#helper function for generating a string of nucleotides of any length
def gen_ran_nuc(len_nuc):
    return ''.join(np.random.choice(['A', 'T', 'C', 'G'], len_nuc))

#generate a variant (5p, 3p, ref, alt, chromosome, position, strand) with a certain chance of being an indel
def generate_variant(length=6, indel_percent=0):
    assert length >= 2
    assert length % 2 == 0
    five_p = gen_ran_nuc(length)
    three_p = gen_ran_nuc(length)
    choices = ['A', 'T', 'C', 'G']
    if np.random.sample() < indel_percent:
        size = int(np.random.choice(range(1, length + 1), 1))
        ##even chance of being insertion vs. deletion
        if np.random.sample() < .5:
            ref = '-' * length
            alt = ''.join(np.random.choice(choices, size)) + '-' * (length - size)
        else:
            ref = ''.join(np.random.choice(choices, size)) + '-' * (length - size)
            alt = '-' * length
    else:
        ref = ''.join(np.random.choice(choices, 1))
        remaining_choices = choices.copy()
        remaining_choices.pop(choices.index(ref))
        alt = ''.join(np.random.choice(remaining_choices, 1))
        ref += '-' * (length - 1)
        alt += '-' * (length - 1)
    chromosome = np.random.choice(range(1, 25))
    position = np.random.sample()
    strand = np.random.choice([1, 2])
    return np.array(list(five_p)), np.array(list(three_p)), np.array(list(ref)), np.array(list(alt)), chromosome, position, strand

##to make sure the reverse of a simulated variant, only check the part of the variant which is consistent
def check_variant(variant, positive_variants, to_check=['five_p']):
    five_p = ''.join(variant[0])
    three_p = ''.join(variant[1])
    ref = ''.join(variant[2])
    alt = ''.join(variant[3])

    x = 0
    X = False
    for pos in positive_variants:
        if 'five_p' in to_check:
            if five_p == ''.join(pos[0]):
                x += 1
        if 'three_p' in to_check:
            if three_p == ''.join(pos[1]):
                x += 1
        if 'ref' in to_check:
            if ref == ''.join(pos[2]):
                x += 1
        if 'alt' in to_check:
            if alt == ''.join(pos[3]):
                x += 1
        if x == len(to_check):
            X = True
            break
    if X:
        return x
    else:
        five_p_rev = str(Seq.reverse_complement(Seq(five_p)))
        three_p_rev = str(Seq.reverse_complement(Seq(three_p)))
        ref_rev = str(Seq.reverse_complement(Seq(ref.replace('-', ''))))
        alt_rev = str(Seq.reverse_complement(Seq(alt.replace('-', ''))))
        x = 0
        for pos in positive_variants:
            if 'three_p' in to_check:
                if five_p_rev == str(Seq.reverse_complement(Seq(''.join(pos[1])))):
                    x += 1
            if 'five_p' in to_check:
                if three_p_rev == str(Seq.reverse_complement(Seq(''.join(pos[0])))):
                    x += 1
            if 'ref' in to_check:
                if ref_rev == str(Seq.reverse_complement(Seq(''.join(pos[2]).replace('-', '')))):
                    x += 1
            if 'alt' in to_check:
                if alt_rev == str(Seq.reverse_complement(Seq(''.join(pos[3]).replace('-', '')))):
                    x += 1
            if x == len(to_check):
                X = True
                break
    return X
