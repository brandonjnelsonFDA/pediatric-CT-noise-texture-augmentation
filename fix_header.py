from pathlib import Path
     
def fix_header(header_file):
     with open(header_file,'r') as f:
         header = f.read()
         fixed = header.replace('    ','')
     with open(header_file,'w') as f:
         f.write(fixed)

s = Path('/gpfs_projects/brandon.nelson/RSTs/Peds_datasets/CCT189')
     
for p in s.rglob('*.mhd'):
     fix_header(p)