<div align="right">
<a href="https://github.com/PalaeoCarb/MyAMI/actions/workflows/test-myami.yml"><img src="https://github.com/PalaeoCarb/MyAMI/workflows/Check%20MyAMI%20Performance/badge.svg" height=18></a>
<a href="https://pypi.org/project/pymyami"><img src="https://badge.fury.io/py/pymyami.svg" height=18></a>
</div>

# MyAMI
The MyAMI Specific Ion Interaction Model for correcting stoichiometric equilibrium constants (*Ks*) for variations in seawater composition, made available available as the `pymyami` python package.

This package is a re-factor of the MyAMI model published by [Hain et al. (2015)](https://doi.org/10.1002/2014GB004986), which is available [here](https://github.com/MathisHain/MyAMI). The key differences between the original model and this package are:
- **Speed**: All calculations have been vectorised using NumPy, making MyAMI 2-3 orders of magnitude faster.
- **Direct Calculation**: `pymyami` directly calculates correction factors using the MyAMI model. This differs from [Hain et al. (2015)](https://doi.org/10.1002/2014GB004986), where the focus was on modifying parameters that could be input into the standard equations for calculating stoichiometric equilibrium products.
- **Correction Factor Focus**: `pymyami` produces *corrections factors* (F<sub>X,MyAMI</sub>) that can be applied to adjust stoichiometric equilibrium constants for variations in seawater composition, following K<sub>X,corr</sub> = K<sub>X,empirical</sub> * F<sub>X,MyAMI</sub>. For the direct calculation of Ks, including the corrections calculated by `pymyami`, please see the [Kgen](https://github.com/PalaeoCarb/Kgen) project.
- **Available Ions**: `pymyami` allows the modification of any ion in the model, rather than just Mg and Ca: Na<sup>+</sup>, K<sup>+</sup>, Mg<sup>2+</sup>, Ca<sup>2+</sup>, Sr<sup>2+</sup>, Cl<sup>-</sup>, B(OH)<sub>4</sub><sup>-</sup>, HCO<sub>3</sub><sup>-</sup>, CO<sub>3</sub><sup>2-</sup> and SO<sub>4</sub><sup>2-</sup>.
- **Parameter Transparrency**: Wherever possible, parameter tables are now constructed on-the-fly from raw tables in the Appendix of [Millero & Pierrot, 1998](https://doi.org/10.1023/A:1009656023546), making the origin of parameters explicit.
- **Pure Python**: There is no longer interface code for interacting with other languages (i.e. MATLAB). This caused a substantial performance bottleneck, and is discouraged. The [Kgen](https://github.com/PalaeoCarb/Kgen) project provides a convenient interface to use `pymyami` in R and MATLAB.
- **Approximation Method**: Where very fast calculations are required (e.g. Monte Carlo methods), `pymyami` uses a high-dimensional polynomial to approximate F<sub>X,MyAMI</sub> as a function of temperature, salinity, Mg and Ca. This is a very fast approximation, but is only accurate to within ~0.25%.

## Kgen
`pymyami` only calculations *correction factors* that can be applied to stoichiometric equilibrium constants (Ks). If you are looking for a convenient way to adjust Ks for variations in seawater composition, please see the [Kgen](https://github.com/PalaeoCarb/Kgen) project.

## Consistency with Hain et al. (2015)
The K correction factors calculated by `pymyami` are identical to those calculated by the code of [Hain et al. (2015)](https://doi.org/10.1002/2014GB004986), with the exception of K<sub>2</sub>, K<sub>spA</sub> and K<sub>spC</sub>, which differ by 0.36%. This difference arises from minor typo corrections in CO<sub>3</sub> coefficients from the original code.

## Installation

The model is available as a PyPI package, which can be installed by:

```python
pip install pymyami
```

## Example Usage
```python
from pymyami import calc_Fcorr, approximate Fcorr

# run the model to calculate correction factors
calc_Fcorr(TempC=35, Sal=36.2, Mg=0.03, Ca=0.012)

# use the polynomial approximation to calculate correction factors
approximate_Fcorr(TempC=35, Sal=36.2, Mg=0.03, Ca=0.012)
```
