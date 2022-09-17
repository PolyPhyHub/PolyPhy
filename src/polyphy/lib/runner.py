from ..scripts import get_polyphys


def __runner(args):
    """Run similation based on polyphy selected"""
    polyphy_obj = get_polyphys(args)
    polyphy_obj.run()


def run2d(args):
    """Run polyphy 2d"""
    return __runner(args)


def run3d(args):
    """Run polyphy 3d"""
    return __runner(args)
