# cdhydro

May 4, 2021:

A 2+1D relativistic barotropic hydrodynamics code on curved spacetime, implementing a constraint-damping scheme in the Hamiltonian formulation to conserve circulation. Design goals do not include shock-capturing capability.

Written in Rust. Under development.

Target studies:

1) Demonstrate the constraint-damping scheme in simple cases.

    i) irrotational case: irrotational test-fluid "star" on a Kerr-like background metric.

    - non-compact star, so don't have to deal with vacuum regions
    - spongy buffer zone plus periodic boundary conditions

    ii) rotational case: destabilizing shear flow on flat spacetime.

2) Study intrinsically relativistic turbulence, AKA relativistically irrotational turbulence.

    - use random metric to force the fluid
    - study inertial range(s)

3) Study whether an artificial atmosphere comprimises the constraint-damping scheme.

    - use hybrid Hamiltonian/Valencia scheme as in JRWS et al 2020 Class. Quantum Grav. 37 155005

4) Implement surface-tracking scheme in 2D, try to combine with constraint-damping hybrid scheme.

    - surface-tracking scheme a la JRWS arXiv:2010.05126
