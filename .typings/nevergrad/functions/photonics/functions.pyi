def addupml2d(ER2, UR2, NPML):
    """
    ADDUPML2D     Add UPML to a 2D Yee Grid

    [ERxx,ERyy,ERzz,URxx,URyy,URzz] = addupml2d(ER2,UR2,NPML)

    INPUT ARGUMENTS
    ================
    ER2       Relative Permittivity on 2x Grid
    UR2       Relative Permeability on 2x Grid
    NPML      [NXLO NXHI NYLO NYHI] Size of UPML on 1x Grid

    OUTPUT ARGUMENTS
    ================
    ERxx      xx Tensor Element for Relative Permittivity
    ERyy      yy Tensor Element for Relative Permittivity
    ERzz      zz Tensor Element for Relative Permittivity
    URxx      xx Tensor Element for Relative Permeability
    URyy      yy Tensor Element for Relative Permeability
    URzz      zz Tensor Element for Relative Permeability
    """
def yeeder2d(NS, RES, BC, kinc: int = 0):
    """
    YEEDER2D      Derivative Matrices on a 2D Yee Grid

    [DEX,DEY,DHX,DHY] = yeeder2d(NS,RES,BC,kinc)

    INPUT ARGUMENTS
    =================
    NS    [Nx Ny] Grid Size
    RES   [dx dy] Grid Resolution
    BC    [xbc ybc] Boundary Conditions
            0: Dirichlet boundary conditions
            1: Periodic boundary conditions
    kinc  [kx ky] Incident Wave Vector
          This argument is only needed for PBCs.

    Note: For normalized grids use k0 * RES and kinc/k0

    OUTPUT ARGUMENTS
    =================
    DEX   Derivative Matrix wrt x for Electric Fields
    DEY   Derivative Matrix wrt to y for Electric Fields
    DHX   Derivative Matrix wrt to x for Magnetic Fields
    DHY   Derivative Matrix wrt to y for Magnetic Fields
    """
def block(xa2, ya2, ER2, pos_x, pos_y, Len_x, Len_y, n): ...
