class Atom:
    def __init__(self, aname, anumb, x, y, z, resnumb, residue):
        self.aname = aname
        self.anumb = anumb
        self.resname = residue
        self.resnumb = resnumb
        self.coords = (x, y, z)

    def calc_sqdist(self, atom) -> float:
        x, y, z = self.coords
        x2, y2, z2 = atom.coords

        return (x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2