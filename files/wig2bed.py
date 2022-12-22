##download wigs from https://www.synapse.org/#!Synapse:syn21785741
from tqdm import tqdm
import subprocess
wig_path = '/home/janaya2/Desktop/ATGC/files/wigs/'
bed_path = '/home/janaya2/Desktop/ATGC/files/beds/'
cmd = ['ls', wig_path]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = [str(i, 'utf-8') for i in p.communicate()[0].split() if '.gz' in str(i)]
files = sorted(files)

for file in tqdm(files):
    name = file.split('.gz')[0] + '.bed'
    cmd = "zcat " + wig_path + file + " | wig2bed -d | awk '$5 == 1' | bedtools merge > " + bed_path + name
    subprocess.run(cmd, shell=True)