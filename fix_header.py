from pathlib import Path
     
def fix_header(header_file):
     with open(header_file,'r') as f:
         header = f.read()
         fixed = header.replace('    ','')
     with open(header_file,'w') as f:
         f.write(fixed)

s = Path(r'D:\Dev\Datasets\CCT189_CT_sims\CCT189_peds')

paths = list(s.glob('diameter*'))
     
for p in paths:
     fix_header(p)