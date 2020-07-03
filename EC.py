def energy_change(sigma,spin_index,J,h):
    """
    Computes the energy change if spin at
    position spin_index is flipped.

    Parameters:
    sigma: configuration
    spin_index: spin to be flipped
    J, h : additional parameters

    Returns:
    ec: energy change
    """
    N_spins = len(sigma)
    ec = 2*J/N_spins*sum(sigma[spin_index]
                         *np.delete(sigma, spin_index))
    +2*h*sigma[spin_index]
    return(ec)
