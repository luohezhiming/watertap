#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
Thermophysical property package to be used in conjunction with ASM2d reactions.

Reference:
[1] Henze, M., Gujer, W., Mino, T., Matsuo, T., Wentzel, M.C., Marais, G.v.R.,
Van Loosdrecht, M.C.M., "Activated Sludge Model No.2D, ASM2D", 1999,
Wat. Sci. Tech. Vol. 39, No. 1, pp. 165-182

"""

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    LiquidPhase,
    Component,
    Solute,
    Solvent,
)
from pyomo.common.config import ConfigValue, In
import idaes.logger as idaeslog
from watertap.property_models.activated_sludge.asm2d_properties import (
    ASM2dParameterData,
    _ASM2dStateBlock,
    ASM2dStateBlockData,
)

from idaes.core.util.exceptions import ConfigurationError, InitializationError

# Some more information about this module
__author__ = "Chenyu Wang"


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ModifiedASM2dParameterBlock")
class ModifiedASM2dParameterData(ASM2dParameterData):
    CONFIG = ASM2dParameterData.CONFIG()

    CONFIG.declare(
        "additional_solute_list",
        ConfigValue(
            domain=list,
            description="List of additional solute species names apart from conventional ASM2D",
        ),
    )

    def build(self):
        super().build()
        self._state_block_class = ModifiedASM2dStateBlock
        # Group components into different sets
        for j in self.config.additional_solute_list:
            if j == "H2O":
                raise ConfigurationError(
                    "'H2O'is reserved as the default solvent and cannot be a solute."
                )
            # Add valid members of solute_list into IDAES's Solute() class.
            # This triggers the addition of j into component_list and solute_set.
            self.add_component(j, Solute())

        # self.S_K = Solute(doc="Potassium")
        # self.S_Mg = Solute(doc="Magnesium")


class _ModifiedASM2dStateBlock(_ASM2dStateBlock):
    pass


@declare_process_block_class(
    "ModifiedASM2dStateBlock", block_class=_ModifiedASM2dStateBlock
)
class ModifiedASM2dStateBlockData(ASM2dStateBlockData):
    pass
