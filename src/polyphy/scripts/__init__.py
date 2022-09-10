from ..lib.simulator import Poly

polyphy_module = ["polyphy2d", "polyphy3d"]
module_list = [
    __import__("polyphy.scripts." + module, fromlist=[None]) for module in polyphy_module
]
polyphy_list = [module.polyphys() for module in module_list]
name_modulemap = {"run2d": "polyphy2d", "run3d": "polyphy3d"}


def get_polyphys(poly_version):
    """"Select the class to use"""

    obj = Poly()
    polyphy_version = poly_version.lower()
    for poly_modules in polyphy_list:
        if name_modulemap[polyphy_version] == poly_modules.name:
            obj = poly_modules
            break
    return obj
