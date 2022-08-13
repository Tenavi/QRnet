import numpy as np

from .containers import Controls

mass = 11. # [kg]
rho = 1.2682 # air density, [kg / m^3]
g0 = 9.81 # gravity, [m/s^2]
b = 2.8956 # wing-span [m]
c = 0.18994 # wing chord [m]
S = 0.55 # wing area [m^2]
e = 0.9 # Oswald's Efficiency Factor []

AR = b ** 2 / S

blending_const = 50. # barrier function coefficient for stall angle of attack
alpha0 = np.deg2rad(20.) # angle at which stall occurs [deg]

Jxx = 0.8244  # [kg m^2]
Jyy = 1.135  # [kg m^2]
Jzz = 1.759  # [kg m^2]
Jxz = 0.1204  # [kg m^2]

J_body = np.array([[Jxx, 0., -Jxz], [0., Jyy, 0.], [-Jxz, 0., Jzz]])
J_det = (Jxx * Jzz - Jxz ** 2)
J_inv_body = np.array([[Jzz, 0., Jxz], [0., J_det / Jyy, 0.], [Jxz, 0., Jxx]])
J_inv_body /= J_det

# Aerodynamic Partial Derivatives

# Lift
CL0 = 0.23  # zero angle of attack lift coefficient
CLalpha = 5.61  # given in supplement
CLq = 7.95  # needs to be normalized by c/2*Va
CLdeltaE = 0.13  # lift due to elevator deflection

# Drag
CD0 = 0.0437 # parasitic drag
CDalpha = 0.03  # drag slope
CDq = 0.  # drag wrt pitch rate
CDdeltaE = 0.0135  # drag due to elevator deflection

# Pitching Moment
CM0 = 0.0135  # intercept of pitching moment
CMalpha = -2.74  # pitching moment slope
CMq = -38.21  # pitching moment wrt q
CMdeltaE = -0.99  # pitching moment from elevator
CMinf = 0.8 # largest post-stall pitching moment

# Sideforce
CY0 = 0.
CYbeta = -0.83
CYp = 0.
CYr = 0.
CYdeltaA = 0.075
CYdeltaR = 0.19

# Rolling Moment
Cl0 = 0.
Clbeta = -0.13
Clp = -0.51
Clr = 0.25
CldeltaA = 0.17
CldeltaR = 0.0024

# Yawing Moment
Cn0 = 0.
Cnbeta = 0.073
Cnp = -0.069
Cnr = -0.095
CndeltaA = -0.011
CndeltaR = -0.069

# Basic propeller model
Sprop = 0.2027 # propeller area [m^2]
kmotor = 32.  # motor constant, DIFFERENT FROM BEARD
kTp = 0.  # motor torque constant
kOmega = 0.  # motor speed constant
Cprop = 0.45  # thrust efficiency coefficient, DIFFERENT FROM BEARD

# Alternate propeller Model
D_prop = 0.508  # prop diameter [m]
KV = 145.  # from datasheet [RPM/V]
KQ = 60. / (2.*np.pi*KV)  # [V-s/rad]
R_motor = 0.042  # [ohms]
i0 = 1.5  # no-load (zero-torque) current [A]
ncells = 12.
V_max = 3.7 * ncells  # max voltage for specified number of battery cells

# Propeller coefficients
C_Q2 = -0.01664
C_Q1 = 0.004970
C_Q0 = 0.005230
C_T2 = -0.1079
C_T1 = -0.06044
C_T0 = 0.09357

# Throttle setting for zero torque
zero_throttle = i0 * R_motor / V_max

# Control constraints
max_angle = np.radians(25.)
min_controls = Controls(
    throttle=0., aileron=-max_angle, elevator=-max_angle, rudder=-max_angle
)
max_controls = Controls(
    throttle=1., aileron=max_angle, elevator=max_angle, rudder=max_angle
)

# roll rate and yaw rate derived parameters
'''Cp0 = JinvBody[0][0] * Cl0 + JinvBody[0][2] * Cn0
Cpbeta = JinvBody[0][0] * Clbeta + JinvBody[0][2] * Cnbeta
Cpp = JinvBody[0][0] * Clp + JinvBody[0][2] * Cnp
Cpr = JinvBody[0][0] * Clr + JinvBody[0][2] * Cnr
CpdeltaA = JinvBody[0][0] * CldeltaA + JinvBody[0][2] * CndeltaA
CpdeltaR = JinvBody[0][0] * CldeltaR + JinvBody[0][2] * CndeltaR
Cr0 = JinvBody[0][2] * Cl0 + JinvBody[2][2] * Cn0
Crbeta = JinvBody[0][2] * Clbeta + JinvBody[2][2] * Cnbeta
Crp = JinvBody[0][2] * Clp + JinvBody[2][2] * Cnp
Crr = JinvBody[0][2] * Clr + JinvBody[2][2] * Cnr
CrdeltaA = JinvBody[0][2] * CldeltaA + JinvBody[2][2] * CndeltaA
CrdeltaR = JinvBody[0][2] * CldeltaR + JinvBody[2][2] * CndeltaR'''
