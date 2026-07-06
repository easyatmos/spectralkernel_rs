# get_version.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spharm_fortran_reference

print(spharm_fortran_reference.__version__)