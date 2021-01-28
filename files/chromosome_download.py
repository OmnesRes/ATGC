##Assumes a linux operating system
##If that's not the case the chromosomes can be downloaded manually.
##Run this script from command line.  Will download chromosomes, remove newlines, delete original files, make directory chromosomes, and move files there.

import subprocess

link = "ftp://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome."

for i in list(range(1,23)) + ['X', 'Y']:
    subprocess.run(['wget', link+str(i)+'.fa.gz'])

cmd = ['ls']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = p.communicate()

for i in files[0].split():
    if '.gz' in str(i):
        subprocess.run(['gunzip', str(i, 'utf-8')])


cmd = ['ls']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = p.communicate()


for i in files[0].split():
    if '.fa' in str(i):
        with open(str(i,'utf-8')) as f:
            sequence = ''
            f.readline()
            for line in f:
                sequence += line.strip()
            with open('chr' + str(i, 'utf-8').split('.')[-2] + '.txt', 'w') as w:
                w.write(sequence)
        subprocess.run(['rm', str(i, 'utf-8')])


subprocess.run(['mkdir', 'chromosomes'])
subprocess.run('mv chr*.txt chromosomes', shell=True)







