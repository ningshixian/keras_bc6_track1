

tmp_1 = """E. Relative mRNA expression in proximal hind limb muscle of AR21Q 
(n = 5) and AR113Q (n = 3) males backcrossed to C57BL/6J. ** p<0.01, ***p<0.001 
by Student's t test. F. Relative mRNA expression in spinal cord of AR21Q (n = 5) 
and AR113Q (n = 3) males (mean +/− SEM). n. s. = not significant by Student's t test."""

tmp_2 = """(b) Titration of phospholipids (for example, TOCL) to LC3 was 
performed to evaluate the ratio that prevents 50% of the LC3 from 
entering the gels (IC50) as an index of relative affinity. 
Bottom left inset: representative gel for TOCL/LC3 = 12; 
top right inset: comparison of IC50 values for TOCL versus 
dioleoyl-phosphatidic acid (DOPA) and tetralinoleoyl-CL (TLCL) 
versus monolyso-trilinoleoyl-CL (lyso-CL). The IC50 for DOPG was >15. 
*P0.05 versus TOCL; †P0.05 versus TLCL."""

tmp = "A, B Quantitative analysis of (A) TUNEL-positive cells and (B) caspase-3-positive cells in IHC in the spleens of CLP + PBS, CLP + Pal-Scram #1, and CLP + Smaducin-6mice. Three independent experiments (n = 3 mice per group per experiments) were performed. At least five hot spots in a section of TUNEL and IHC per experiment were selected, and average count was determined. The data were expressed as a mean percentage of total cell numbers and statistically analyzed by a t-test and show the mean ± SD of three independent experiments. **P < 0.005, ***P < 0.001 compared to sham or vehicle control (CLP + Pal-Scram #1)."

for specific_symbol in "!\"#$%'()*+,-./:;<=>?@[\\]_`{|}~":     # °C ^
    tmp2 = tmp.replace(specific_symbol, ' '+specific_symbol+' ')
print('tmp2: ' + tmp2)

tmp3 = ' '.join(tmp.split())

for specific_symbol in "!\"#$%'()*+,-./:;<=>?@[\\]_`{|}~":     # °C ^
    tmp3 = tmp3.replace(specific_symbol, ' '+specific_symbol+' ')
print(tmp3)

tmp2 = tmp2.replace('   ', ' ').replace('  ', ' ')
tmp3 = tmp3.replace('   ', ' ').replace('  ', ' ')
print(tmp2)
print(tmp3)

if '' in tmp2.split():
    print('df')
if '  ' in tmp3:
    print('dfdf')