# DM analysis

This is based on the Ben's dark matter analysis package (https://github.com/VERITAS-Observatory/DM-Aspen).

- Requirement:
    python 3 and pyROOT

- Prerequisite:
    - Define an environment variable "DM", which points to the location of your analysis directory.
    e.g., export DM=$HOME/DarkMatter
    - Prepare IRF files in $DM/RefData, e.g., effArea-v483-auxv01-CARE_June1702-Cut-NTel2-PointSource-Soft-TMVA-BDT-GEO-V6_2019_2020-ATM61-T1234.root
    - Prepare anasum root files in $DM/Data/'dwarf name', e.g., $DM/Data/segue_1/48865.anasum.root

For usage, you can refer to tutorials. 
