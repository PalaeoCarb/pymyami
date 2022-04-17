import numpy as np
from .helpers import expand_dims, match_dims, standard_seawater, calc_Istr, calc_KF, calc_KS
from .params import TABLES, calc_lambda_zeta, Iind, calc_seawater_ions

# TODO: new file for user-facing functions.

def calc_gKs(TC, Sal, Na=None, K=None, Ca=None, Mg=None, Sr=None, Cl=None, BOH4=None, HCO3=None, CO3=None, SO4=None,
                  beta_0=None, beta_1=None, beta_2=None, C_phi=None, Theta_negative=None, Theta_positive=None, Phi_NNP=None, Phi_PPN=None):
    """
    Calculate Ks at given conditions using MyAMI model.

    Parameters
    ----------
    TC : array-like
        Temperature in Celcius
    Sal : array-like
        Salinity
    Na, K, Ca, Mg, Sr, Cl, BOH4, HCO3, CO3, SO4 : array-like, optional
        Average concentration of ions in seawater in mol kg-1, by default None
    beta_0, beta_1, beta_2, C_phi : numpy.NDarray, optional
        Matrices of ion interaction coefficients from tables A1-A9 of
        Millero and Pierrot (1998; doi:10.1023/A:1009656023546) provided
        by the params.PitzerParams function.
    Theta_negative, Theta_positive, Phi_NNP, Phi_PPN : numpy.NDarray, optional
        Matrices of ion interaction coefficients from tables A10 and A11 of
        Millero and Pierrot (1998; doi:10.1023/A:1009656023546) provided
        by the params.PitzerParams function.

    Returns
    -------
    dict of array-like
        Containing {gKspC, gKspA, gK1, gK2, gKW, gKB, gK0, gKS}
    """

    TK = TC + 273.15
    Istr = calc_Istr(Sal)
    m_cation, m_anion = calc_seawater_ions(Sal, Na=Na, K=K, Mg=Mg, Ca=Ca, Sr=Sr, Cl=Cl, BOH4=BOH4, HCO3=HCO3, CO3=CO3, SO4=SO4)

    gammas, alphas = calc_gamma_alpha(
        TK=TK, Sal=Sal, Istr=Istr, m_cation=m_cation, m_anion=m_anion, 
        beta_0=beta_0, beta_1=beta_1, beta_2=beta_2, C_phi=C_phi,
        Theta_negative=Theta_negative, Theta_positive=Theta_positive,
        Phi_NNP=Phi_NNP, Phi_PPN=Phi_PPN)

    gammaT_OH = gammas['anion'][0] * alphas['OH']
    gammaT_BOH4 = gammas['anion'][2]
    gammaT_HCO3 = gammas['anion'][3]
    gammaT_CO3 = gammas['anion'][5] * alphas['CO3']
    
    gammaT_Ht = gammas['cation'][0] * alphas['Ht']
    gammaT_Ca = gammas['cation'][4]

    gammaCO2_gammaB = calc_gammaCO2_gammaB(TK, m_anion, m_cation)

    out = {
    'gKspC': 1 / gammaT_Ca / gammaT_CO3,
    'gKspA': 1 / gammaT_Ca / gammaT_CO3,
    'gK1': 1 / gammaT_Ht / gammaT_HCO3 * gammaCO2_gammaB['gammaCO2'],
    'gK2': 1 / gammaT_Ht / gammaT_CO3 * gammaT_HCO3,
    'gKW': 1 / gammaT_Ht / gammaT_OH,
    'gKB': 1 / gammaT_BOH4 / gammaT_Ht * gammaCO2_gammaB['gammaB'],
    'gK0': 1 / gammaCO2_gammaB['gammaCO2'] * gammaCO2_gammaB['gammaCO2gas'],
    'gKS': 1 / gammas['anion'][6] / gammaT_Ht * gammas['anion'][4],
    }

    return out


def calc_gamma_alpha(TK, Sal, Istr, m_cation, m_anion,
                  beta_0=None, beta_1=None, beta_2=None, C_phi=None, Theta_negative=None, Theta_positive=None, Phi_NNP=None, Phi_PPN=None):
    """Calculate Gammas and Alphas for K calculations.

    Parameters
    ----------
    TK : array-like
        Temperature in Kelvin
    S : array-like
        Salinity in PSU
    Istr : array-like
        Ionic strength of solution
    m_cation : array-like
        Matrix of major cations in seawater in mol/kg in order:
        [H, Na, K, Mg, Ca, Sr]
    m_anion : array-like
        Matrix of major anions in seawater in mol/kg in order:
        [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4]

    Returns
    -------
    tuple of dicts
        (gammas: {cations, anions},
         alphas: {Hsws, Ht, OH, CO3})
    """
    # TODO: Derive this from paper tables?
    
    # Testbed case T=25C, I=0.7, seawatercomposition
    sqrtI = np.sqrt(Istr)
    
    # make tables of ion charges used in later calculations

    # cation order: [H, Na, K, Mg, Ca, Sr]
    cation_charges = np.array([1, 1, 1, 2, 2, 2])
    Z_cation = np.full(
        (cation_charges.size, *TK.shape),
        expand_dims(cation_charges, TK)
        )

    # anion order: [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4]
    anion_charges = np.array([-1, -1, -1, -1, -1, -2, -2])
    Z_anion = np.full(
        (anion_charges.size, *TK.shape),
        expand_dims(anion_charges, TK)
        )   

    ##########################################################################
    # Code below largely lifted from MyAMI V1 (Hain et al., 2015), but
    # vectorised for speed.
    ##########################################################################

    A_phi = (
        3.36901532e-01
        - 6.32100430e-04 * TK
        + 9.14252359 / TK
        - 1.35143986e-02 * np.log(TK)
        + 2.26089488e-03 / (TK - 263)
        + 1.92118597e-6 * TK * TK
        + 4.52586464e01 / (680 - TK)
    )  # note correction of last parameter, E + 1 instead of E-1
    # A_phi = 8.66836498e1 + 8.48795942e-2 * T - 8.88785150e-5 * T * T +
    # 4.88096393e-8 * T * T * T -1.32731477e3 / T - 1.76460172e1 * np.log(T)
    # # Spencer et al 1990

    f_gamma = -A_phi * (sqrtI / (1 + 1.2 * sqrtI) + (2 / 1.2) * np.log(1 + 1.2 * sqrtI))

    # E_cat = sum(m_cation * Z_cation)
    E_an = -sum(m_anion * Z_anion)
    E_cat = -E_an

    BMX_phi = beta_0 + beta_1 * np.exp(-2 * sqrtI)
    BMX = beta_0 + (beta_1 / (2 * Istr)) * (1 - (1 + 2 * sqrtI) * np.exp(-2 * sqrtI))
    BMX_apostroph = (beta_1 / (2 * Istr * Istr)) * (-1 + (1 + (2 * sqrtI) + (2 * sqrtI)) * np.exp(-2 * sqrtI))
    CMX = C_phi / (2 * np.sqrt(-np.expand_dims(Z_anion, 0) * np.expand_dims(Z_cation, 1)))
    
    ################################################################################
    # BMX* and CMX are calculated differently for 2:2 ion pairs, corrections
    # below  # § alpha2= 6 for borates ...
    # TODO: Look at Simonson et al 1988 to understand this
    ################################################################################
    
    # MgBOH42
    cat, an = 3, 2
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-6 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (18 * Istr)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (18 * Istr)) * (
        -1 - (1 + 6 * sqrtI + 18 * Istr) * np.exp(-6 * sqrtI)
    )
    
    # MgSO4
    cat, an = 3, 6 
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-12 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (72 * Istr)) * (1 - (1 + 12 * sqrtI) * np.exp(-12 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (72 * Istr * Istr)) * (
        -1 - (1 + 12 * sqrtI + 72 * Istr) * np.exp(-12 * sqrtI)
    )
    # BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr)) * (-1 + (1 + 1.4
    # * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)) + (beta_2[cat, an] / (72 *
    # Istr)) * (-1-(1 + 12 * sqrtI + 72 * Istr) * np.exp(-12 * sqrtI)) # not 1 /
    # (0.98 * Istr * Istr) ... compare M&P98 equation A17 with Pabalan and Pitzer
    # 1987 equation 15c / 16b
    
    # CaBOH42
    cat, an = 4, 2 
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-6 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (18 * Istr)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (18 * Istr)) * (
        -1 - (1 + 6 * sqrtI + 18 * Istr) * np.exp(-6 * sqrtI)
    )
    
    # CaSO4
    cat, an = 4, 6
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-12 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (72 * Istr)) * (1 - (1 + 12 * sqrtI) * np.exp(-12 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (72 * Istr)) * (
        -1 - (1 + 12 * sqrtI + 72 * Istr) * np.exp(-12 * sqrtI)
    )

    # SrBOH42
    cat, an = 5, 2 
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-6 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (18 * Istr)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (18 * Istr)) * (
        -1 - (1 + 6 * sqrtI + 18 * Istr) * np.exp(-6 * sqrtI)
    )

    # H-SO4
    cat, an = 0, 6
    # BMX* is calculated with T-dependent alpha for H-SO4; see Clegg et al.,
    # 1994 --- Millero and Pierrot are completly off for this ion pair
    xClegg = (2 - 1842.843 * (1 / TK - 1 / 298.15)) * sqrtI
    # xClegg = (2) * sqrtI
    gClegg = 2 * (1 - (1 + xClegg) * np.exp(-xClegg)) / (xClegg * xClegg)
    # alpha = (2 - 1842.843 * (1 / T - 1 / 298.15)) see Table 6 in Clegg et al
    # 1994
    BMX[cat, an] = beta_0[cat, an] + beta_1[cat, an] * gClegg
    BMX_apostroph[cat, an] = beta_1[cat, an] / Istr * (np.exp(-xClegg) - gClegg)

    C1_HSO4 = 0  # not sure what this is, but was in original MyAMI...
    CMX[cat, an] = (
        C_phi[cat, an] + 4 * C1_HSO4 * 
        (6 - (6 + 2.5 * sqrtI * (6 + 3 * 2.5 * sqrtI + 2.5 * sqrtI * 2.5 * sqrtI)) *
        np.exp(-2.5 * sqrtI)) / 
        (.5 * sqrtI * 2.5 * sqrtI * 2.5 * sqrtI * 2.5 * sqrtI)
        )  # w = 2.5 ... see Clegg et al., 1994

    # unusual alpha=1.7 for Na2SO4
    # BMX[1, 6] = beta_0[1, 6] + (beta_1[1, 6] / (2.89 * Istr)) * 2 * (1 - (1 + 1.7 * sqrtI) * np.exp(-1.7 * sqrtI))
    # BMX[1, 6] = beta_0[1, 6] + (beta_1[1, 6] / (1.7 * Istr)) * (1 - (1 + 1.7 * sqrtI) * np.exp(-1.7 * sqrtI))

    # BMX[4, 6] =BMX[4, 6] * 0  # knock out Ca-SO4
    
    ################################################################################
    # Calculate gamma_anion and gamma_cation from BMX and CMX    
    ################################################################################
    
    # anion * cation * BMX or CMX matrices
    mR = (m_anion * np.expand_dims(m_cation, 1) * BMX_apostroph).sum((0,1))
    mS = (m_anion * np.expand_dims(m_cation, 1) * CMX).sum((0,1))

    # ln_gammaCl = Z_anion[1] * Z_anion[1] * f_gamma + R - S

    # Original ln_gamma_anion calculation loop:
    ln_gamma_anion = Z_anion * Z_anion * (f_gamma + mR) + Z_anion * mS
    for an in range(0, 7):
        for cat in range(0, 6):
            ln_gamma_anion[an] += 2 * m_cation[cat] * (
                BMX[cat, an] + E_cat * CMX[cat, an]
            )
        for an2 in range(0, 7):
            ln_gamma_anion[an] += m_anion[an2] * (
                2 * Theta_negative[an, an2]
            )
        for an2 in range(0, 7):
            for cat in range(0, 6):
                ln_gamma_anion[an] += (
                    m_anion[an2] * m_cation[cat] * Phi_NNP[an, an2, cat]
                )
        for cat in range(0, 6):
            for cat2 in range(cat + 1, 6):
                ln_gamma_anion[an] += (
                    m_cation[cat] * m_cation[cat2] * Phi_PPN[cat, cat2, an]
                )
    
    # vectorised ln_gamma_anion calculation:
    # TODO: Runs into memory problems with large inputs. Could be simplified further? 
    # cat, cat2 = np.triu_indices(6, 1)
    # ln_gamma_anion = (
    #     Z_anion * Z_anion * (f_gamma + R) + Z_anion * S + 
    #     (2 * np.expand_dims(m_cation, 1) * (BMX + E_cat * CMX)).sum(0) + 
    #     (np.expand_dims(m_anion, 1) * 2 * Theta_negative).sum(0) + 
    #     (np.expand_dims(m_anion, (0,2)) * np.expand_dims(m_cation, (0,1)) * Phi_NNP).sum(axis=(1,2)) +
    #     (np.expand_dims(m_cation[cat], 1) * np.expand_dims(m_cation[cat2], 1) * Phi_PPN[cat, cat2]).sum(axis=0)
    # )  
    gamma_anion = np.exp(ln_gamma_anion)


    # ln_gammaCl = Z_anion[1] * Z_anion[1] * f_gamma + R - S

    # Original ln_gamma_cation calculation loop:
    ln_gamma_cation = Z_cation * Z_cation * (f_gamma + mR) + Z_cation * mS
    for cat in range(0, 6):
        for an in range(0, 7):
            ln_gamma_cation[cat] += 2 * m_anion[an] * (
                BMX[cat, an] + E_cat * CMX[cat, an]
            )
        for cat2 in range(0, 6):
            ln_gamma_cation[cat] += m_cation[cat2] * (2 * Theta_positive[cat, cat2])
        for cat2 in range(0, 6):
            for an in range(0, 7):
                ln_gamma_cation[cat] += (
                    m_cation[cat2] * m_anion[an] * Phi_PPN[cat, cat2, an]
                )
        for an in range(0, 7):
            for an2 in range(an + 1, 7):
                ln_gamma_cation[cat] += (
                    + m_anion[an] * m_anion[an2] * Phi_NNP[an, an2, cat]
                )

    # vectorised ln_gamma_cation calculation:
    # TODO: Runs into memory problems with large inputs. Could be simplified further? 
    # an, an2 = np.triu_indices(7, 1)
    # ln_gamma_cation = (
    #     Z_cation * Z_cation * (f_gamma + R) + Z_cation * S +
    #     (2 * np.expand_dims(m_anion, 0) * (BMX + E_cat * CMX)).sum(axis=1) +
    #     (np.expand_dims(m_cation, 1) * (2 * Theta_positive)).sum(axis=0) +
    #     (np.expand_dims(m_cation, (0,2)) * np.expand_dims(m_anion, (0,1)) * Phi_PPN).sum(axis=(1,2))+
    #     (np.expand_dims(m_anion[an], 1) * np.expand_dims(m_anion[an2], 1) * Phi_NNP[an, an2]).sum(axis=0)
    # )
    gamma_cation = np.exp(ln_gamma_cation)

    # choice of pH-scale = total pH-scale [H]T = [H]F + [HSO4]
    # so far gamma_H is the [H]F activity coefficient (= free-H pH-scale)
    # thus, conversion is required
    K_HSO4_conditional = calc_KS(TK=TK, Sal=Sal, Istr=Istr)
    K_HF_conditional = calc_KF(TK=TK, Sal=Sal)
    TF = 0.0000683
    TS = m_anion[6]
    
    alpha_Hsws = 1 / (1 + TS / K_HSO4_conditional + TF / K_HF_conditional)
    alpha_Ht = 1 / (1 + TS / K_HSO4_conditional)

    # NOTE: Unclear where this next section about gamma_MGCO3 has come from - talk to Mathis!
    # A number of ion pairs are calculated explicitly: MgOH, CaCO3, MgCO3, SrCO3
    # since OH and CO3 are rare compared to the anions the anion alpha (free /
    # total) are assumed to be unity
    gamma_MgCO3 = gamma_CaCO3 = gamma_SrCO3 = 1

    b0b1CPhi_MgOH = np.array([-0.1, 1.658, 0, 0.028])  # where is this from?
    BMX_MgOH = b0b1CPhi_MgOH[0] + (b0b1CPhi_MgOH[1] / (2 * Istr)) * (1 - (1 + 2 * sqrtI) * np.exp(-2 * sqrtI))
    ln_gamma_MgOH = 1 * (f_gamma + mR) + (1) * mS
    ln_gamma_MgOH = ln_gamma_MgOH + 2 * m_anion[1] * (BMX_MgOH + E_cat * b0b1CPhi_MgOH[2])  # interaction between MgOH-Cl affects MgOH gamma
    ln_gamma_MgOH = ln_gamma_MgOH + m_cation[3] * m_anion[1] * b0b1CPhi_MgOH[3]  # interaction between MgOH-Mg-OH affects MgOH gamma
    gamma_MgOH = np.exp(ln_gamma_MgOH)

    K_MgOH = np.power(10, -(3.87 - 501.6 / TK)) / (gamma_cation[3] * gamma_anion[0] / gamma_MgOH)
    K_MgCO3 = np.power(10, -(1.028 + 0.0066154 * TK)) / (gamma_cation[3] * gamma_anion[5] / gamma_MgCO3)
    K_CaCO3 = np.power(10, -(1.178 + 0.0066154 * TK)) / (gamma_cation[4] * gamma_anion[5] / gamma_CaCO3)
    # K_CaCO3 = np.power(10, (-1228.732 - 0.299444 * T + 35512.75 / T +485.818 * np.log10(T))) / (gamma_cation[4] * gamma_anion[5] / gamma_CaCO3) # Plummer and Busenberg82
    # K_MgCO3 = np.power(10, (-1228.732 +(0.15) - 0.299444 * T + 35512.75 / T
    # +485.818 * np.log10(T))) / (gamma_cation[4] * gamma_anion[5] /
    # gamma_CaCO3)# Plummer and Busenberg82
    K_SrCO3 = np.power(10, -(1.028 + 0.0066154 * TK)) / (gamma_cation[5] * gamma_anion[5] / gamma_SrCO3)

    alpha_OH = 1 / (1 + (m_cation[3] / K_MgOH))
    alpha_CO3 = 1 / (1 + (m_cation[3] / K_MgCO3) + (m_cation[4] / K_CaCO3) + (m_cation[5] / K_SrCO3))

    return ({'cation': gamma_cation, 'anion': gamma_anion}, 
            {'Hsws': alpha_Hsws, 'Ht': alpha_Ht, 'OH': alpha_OH, 'CO3': alpha_CO3})

def calc_gammaCO2_gammaB(TK, m_an, m_cat):
    """
    Calculate gammaCO2 and gammaB

    Parameters
    ----------
    TC : array-like
        Temperature in Kelvin, used to determine array shapes
    m_an : dict
        Containing cation concentrations in mol/kgsw 
    m_cat : dict
        Containing anion concentrations in mol/kgsw

    Returns
    -------
    dict
        Containing {gammaCO2, gammaCO2gas, gammaB}
    """

    cations = ['H', 'Na', 'K', 'Mg', 'Ca']
    anions = ['Cl', 'SO4']
    
    m_cation = np.array([m_cat[Iind[c]] for c in cations])
    m_anion = np.array([m_an[Iind[a]] for a in anions])
    m_ion = np.concatenate([m_cation, m_anion])
    
    m_zeta = (np.expand_dims(m_anion,1) * np.expand_dims(m_cation,0))  # matrix for zeta calculation
        
    lambda_zeta = calc_lambda_zeta(TK)
    
    lambdaCO2 = lambda_zeta['lambdaCO2']
    zetaCO2 = lambda_zeta['zetaCO2']  # not used in Hain's MyAMI?
    lambdaB = lambda_zeta['lambdaB']
    zetaB = lambda_zeta['zetaB']
    
    ##########################
    # CALCULATION OF gammaCO2

    ln_gammaCO2 = (m_ion * 2 * lambdaCO2).sum(0)  # lambdaCO2
    # ln_gammaCO2 += (m_zeta * zetaCO2).sum((0,1))  # zetaCO2 (not used in original MyAMI, introduces small differences...)
    gammaCO2 = np.exp(ln_gammaCO2)  # as according to He and Morse 1993


    gammaCO2gas = np.exp(
        1 / (8.314462175 * TK *
            (0.10476 - 61.0102 / TK - 660000 / TK / TK / TK - 2.47e27 / np.power(TK, 12))
        )
    )  # unclear where this comes from.

    ##########################
    # CALCULATION OF gammaB
        
    ln_gammaB = (m_ion * 2 * match_dims(lambdaB, m_ion)).sum(0)  # lambdaB
    ln_gammaB += (m_zeta * match_dims(zetaB, m_zeta)).sum()  # zetaB
    
    gammaB = np.exp(ln_gammaB)  # as according to Felmy and Wear 1986
    # print gammaB

    return {
        'gammaCO2': gammaCO2, 
        'gammaCO2gas': gammaCO2gas, 
        'gammaB': gammaB
        }