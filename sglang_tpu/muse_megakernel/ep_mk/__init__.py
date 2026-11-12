# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel decode megakernel: the row-tiled megakernel with its expert section
replaced (mk/hooks.py) by 32 whole experts per rank and a y return to the row owners."""
