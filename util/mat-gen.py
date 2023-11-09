# This file is taken from
# https://gist.github.com/pjattke/b56baff62be77f16ad8d33376789be67
# The bank functions are modified to the address mapping the gem5 uses.
# The mapping functions of gem5 are:
# * RoCoRaBaCh
# * RoRaBaCoCh
# * RoRaBaChCo
#
# To find more info on how these mappings exactly works, please refer to
# https://gem5-review.googlesource.com/c/public/gem5/+/51614/2/src/python/gem5/components/memory/ReadMe_MultiChannel_Memory.md
#
# We only write the summary of the mappings here:
# addr and its corresponding chan, rank, bank and row are given by:
#         (a) RoCoRaBaCh
#             chan: 1
#             rank: 1
#             bank: 0010
#             row : 0000000000000000
#             The lowerOrderColumnBits, i.e., 0000000 are interleaved.
#             The higherOrderColumnBits are 000010
#             Visual: 0b00000000000000000000011100101000000
#                       └───────┬──────┴───┬───┼─┬──┼──┬──┘
#                               │          │   │ │  │  │
#                               │          │   │ │  │  └─ lowerOrderColumnBits
#                               │          │   │ │  └──── chan
#                               │          │   │ └─────── bank
#                               │          │   └───────── rank
#                               │          └───────────── higherOrderColumnBits
#                               └──────────────────────── row
#         (b) RoRaBaCoCh
#             chan: 1
#             rank: 0
#             bank: 0000
#             row : 0000000000000000
#             The lowerOrderColumnBits, i.e., 0000000 are interleaved.
#             The higherOrderColumnBits are 1110010
#             Visual: 0b00000000000000000000011100101000000
#                       └───────┬───────┼──┬─┴───┬──┼──┬──┘
#                               │       │  │     │  │  │
#                               │       │  │     │  │  └─ lowerOrderColumnBits
#                               │       │  │     │  └──── chan
#                               │       │  │     └─────── higherOrderColumnBits
#                               │       │  └───────────── bank
#                               │       └──────────────── rank
#                               └──────────────────────── row
#         (c) RoRaBaChCo
#             chan: 1
#             rank: 0
#             bank: 0000
#             row : 0000000000000000
#             The lowerOrderColumnBits and higherOrderColumnBits are merged
#             together in this case, represented by column.
#             Visual: 0b00000000000000000000011100101000000
#                       └───────┬───────┼──┬─┼──────┬────┘
#                               │       │  │ │      │
#                               │       │  │ │      └──── column
#                               │       │  │ │
#                               │       │  │ └─────────── chan
#                               │       │  └───────────── bank
#                               │       └──────────────── rank
#                               └──────────────────────── row
# 

import sys
import os
import math
import numpy as np
import pprint as pp

class BinInt(int):
    def __repr__(s):
        return s.__str__()

    def __str__(s):
        return f"{s:#032b}"


class DRAMFunctions():

    def __init__(self, bank_fns, row_fn, col_fn, num_channels, num_dimms, num_ranks, num_banks):
        def to_binary_array(v):
            vals = []
            for x in range(30):
                if (v >> x) & 1:
                    vals.append(1 << x)
            return list(reversed(vals))

        def gen_mask(v):
            len_mask = bin(v).count("1")
            mask = (1 << len_mask)-1
            return (len_mask, mask)

        bank_mask = (1 << len(bank_fns))-1
        row_arr = to_binary_array(row_fn)
        len_row_mask, row_mask = gen_mask(row_fn)
        col_arr = to_binary_array(col_fn)
        len_col_mask, col_mask = gen_mask(col_fn)

        self.row_arr = row_arr
        self.col_arr = col_arr
        self.bank_arr = bank_fns
        self.row_shift = 0
        self.col_shift = len_row_mask
        self.bank_shift = len_row_mask + len_col_mask
        self.row_mask = BinInt(row_mask)
        self.col_mask = BinInt(col_mask)
        self.bank_mask = BinInt(bank_mask)
        self.num_channels = num_channels
        self.num_dimms = num_dimms
        self.num_ranks = num_ranks
        self.num_banks = num_banks

    def to_dram_mtx(self):
        mtx = self.bank_arr + self.col_arr + self.row_arr
        return list(map(lambda v: BinInt(v), mtx))

    def to_addr_mtx(self):
        dram_mtx = self.to_dram_mtx()
        mtx = np.array([list(map(int, list(f"{x:030b}"))) for x in dram_mtx])
        print(mtx.shape, "\n", mtx)
        assert mtx.shape == (30, 30)
        inv_mtx = list(map(abs, np.linalg.inv(mtx).astype('int64')))
        inv_arr = []
        for i in range(len(inv_mtx)):
            inv_arr.append(BinInt("0b" + "".join(map(str, inv_mtx[i])), 2))
        return inv_arr

    def __repr__(self):
        dram_mtx = self.to_dram_mtx()
        addr_mtx = self.to_addr_mtx()
        sstr = "void DRAMAddr::initialize_configs() {\n"
        sstr += "  struct MemConfiguration dram_cfg = {\n"
        sstr += f"      .IDENTIFIER = (CHANS({self.num_channels}UL) | DIMMS({self.num_dimms}UL) | RANKS({self.num_ranks}UL) | BANKS({self.num_banks}UL)),\n"
        sstr += "      .BK_SHIFT = {0},\n".format(self.bank_shift)
        sstr += "      .BK_MASK = ({0}),\n".format(self.bank_mask)
        sstr += "      .ROW_SHIFT = {0},\n".format(self.row_shift)
        sstr += "      .ROW_MASK = ({0}),\n".format(self.row_mask)
        sstr += "      .COL_SHIFT = {0},\n".format(self.col_shift)
        sstr += "      .COL_MASK = ({0}),\n".format(self.col_mask)
        
        str_mtx = pp.pformat(dram_mtx, indent=10)
        trans_tab = str_mtx.maketrans('[]', '{}')
        str_mtx = str_mtx.translate(trans_tab)
        str_mtx = str_mtx.replace("{", "{          \n ")
        str_mtx = str_mtx.replace("}", "\n       },")
        sstr += f"      .DRAM_MTX = {str_mtx}\n"
        
        str_mtx = pp.pformat(addr_mtx, indent=10)
        trans_tab = str_mtx.maketrans('[]', '{}')
        str_mtx = str_mtx.translate(trans_tab)
        str_mtx = str_mtx.replace("{", "{          \n ")
        str_mtx = str_mtx.replace("}", "\n       }")
        sstr += f"      .ADDR_MTX = {str_mtx}"
        
        sstr += "\n  };\n"
        sstr += "  DRAMAddr::Configs = {\n       {" + f"(CHANS({self.num_channels}UL) | DIMMS({self.num_dimms}UL) | RANKS({self.num_ranks}UL) | BANKS({self.num_banks}UL)), dram_cfg" + "}\n };\n"

        sstr += "}"
        return sstr

# TODO Fill this section with information determined by DRAMA =========
# optional information
# num_channels = 1 # the number of channels used (choose 1 for a 1 DIMM system)
# num_dimms = 1 # blacksmith has not been tested with multi-DIMM systems
# num_ranks = 2 # can be determined by "sudo dmidecode -t memory"
# num_banks = 16 # typically 16 banks for x8 devices, 8 banks for x16 devices (see datasheet)

# gem5 changes are added here.
# DRAMA information is -> bank0, bank1 .. bankN-1, channel
# For all the three mappings, we add a different class
class Gem5Mappings:
    def __init__(self,
            name,
            num_channels,
            num_dimms,
            num_ranks,
            num_banks,
            dram_fn = None,
            row_fn = None,
            col_fn = None
        ):
        self.name = name
        self.num_channels = num_channels
        self.num_dimms = num_dimms
        self.num_ranks = num_ranks
        self.num_banks = num_banks

        self.dram_fn = dram_fn
        self.row_fn = row_fn
        self.col_fn = col_fn
    
    def check_lengths(self):
        _expected_functions = \
                math.log(num_ranks, 2) + math.log(num_banks, 2)

        if int(_expected_functions) != len(self.dram_fn) - 1:
            print(_expected_functions)
            print("fatal; number of functions mismatch!")
            exit(-1)

        return True

    def get_name(self):
        return self.name


# TODO: Add argparse to make it more user friendly
# Adding RoCoRaBaCh
RoCoRaBaCh = Gem5Mappings("RoCoRaBaCh", 1, 1, 2, 16)
RoCoRaBaCh.dram_fn = [0x80, 0x100, 0x200, 0x400, 0x800, 0x40]
RoCoRaBaCh.row_fn = 0x3ffc0000
RoCoRaBaCh.col_fn = 0x3f03f
RoCoRaBaCh.check_lengths()

# Adding RoRaBaCoCh
RoRaBaCoCh = Gem5Mappings("RoRaBaCoCh", 

print("Now replace the existing function initialize_configs() in DRAMAddr.cpp by the following code:\n--------------")

print(DRAMFunctions(RoCoRaBaCh.dram_fn,
                RoCoRaBaCh.row_fn,
                RoCoRaBaCh.col_fn,
                RoCoRaBaCh.num_channels,
                RoCoRaBaCh.num_dimms,
                RoCoRaBaCh.num_ranks,
                RoCoRaBaCh.num_banks))
