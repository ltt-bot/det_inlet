import yt
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/jb5027/cfd-solver/')

from src.mesh import Mesh
from src.input import Input
from src.mech_reader import MechReader
from src.eos import temp_yk_to_e, temp_yk_to_h, rho_temp_yk_to_p
from src.fluxes import get_fluxes
from src.analysis import get_terms_for_analysis

# Load in two time steps
# ds_286092 = yt.load('/scratch/gpfs/jb5027/cavity-flame-data/C2/timesteps/plt286092')
# ds_286093 = yt.load('/scratch/gpfs/jb5027/cavity-flame-data/C2/timesteps/plt286093')
ds_286092 = yt.load('/scratch/gpfs/jb5027/cavity-flame-data/C1/nonunity/plt13376')
ds_286093 = yt.load('/scratch/gpfs/jb5027/cavity-flame-data/C1/nonunity/plt13377')

# Get the information needed for the grid
max_level = ds_286093.index.max_level
# lo=np.array([0.0,0.0,0.00])
# hi=np.array([7.2,1.8,0.45])
# lo=np.array([1.25,0.0,0.15])
# hi=np.array([1.65,0.4,0.35])
lo=np.array([0.75,0.0,0.15])
hi=np.array([1.35,0.4,0.35])
dxmin = ds_286093.index.get_smallest_dx()
dxmax = dxmin*2.0*2.0
dx_np = np.array(dxmin)
npts=np.floor((hi-lo)/dxmin)
# npts=np.floor((hi-lo)/dxmax)
npts_np = np.array(npts)
# fields_load=["vfrac","density","pressure", "Temp", "x_velocity", "y_velocity", "z_velocity"]
# fields_load=["density","pressure", "Temp", "x_velocity", "y_velocity", "z_velocity",
#     "Y(N2)", "Y(H2)", "Y(H)", "Y(O)", "Y(O2)", "Y(H2O)", "Y(HO2)", "Y(H2O2)"]
fields_load=["density","pressure", "Temp", "x_velocity", "y_velocity", "z_velocity",
    "rho_N2", "rho_H2", "rho_H", "rho_O", "rho_O2", "rho_H2O", "rho_HO2", "rho_H2O2", "rho_OH"]

# Get the values on the grid at level 2 (finest level)
ad_286092 = ds_286092.covering_grid(level=max_level, left_edge=lo, dims=npts, fields=fields_load)
ad_286093 = ds_286093.covering_grid(level=max_level, left_edge=lo, dims=npts, fields=fields_load)
# ad_286092 = ds_286092.covering_grid(level=0, left_edge=lo, dims=npts, fields=fields_load)
# ad_286093 = ds_286093.covering_grid(level=0, left_edge=lo, dims=npts, fields=fields_load)
dens_286092    = np.array(ad_286092["density"])
dens_286093    = np.array(ad_286093["density"])

x_arr = np.linspace(lo[0], hi[0], int(npts_np[0]+1))

# Build time derivatives
# time_plt286092 = 0.0020000052369055794
# time_plt286093 = 0.0020000080533389458
time_plt286092 = 0.00010012704338181725
time_plt286093 = 0.00010013483314447339
dt = time_plt286093 - time_plt286092
drho = (dens_286093 - dens_286092)
drho_dt = (dens_286093 - dens_286092)/dt
drhou_dt = ((np.array(ad_286093["x_velocity"])*dens_286093 - np.array(ad_286092["x_velocity"])*dens_286092) / dt)
drhov_dt = ((np.array(ad_286093["y_velocity"])*dens_286093 - np.array(ad_286092["y_velocity"])*dens_286092) / dt)
drhow_dt = ((np.array(ad_286093["z_velocity"])*dens_286093 - np.array(ad_286092["z_velocity"])*dens_286092) / dt)

# Build the continuity terms
u    = np.array(ad_286093["x_velocity"])
v    = np.array(ad_286093["y_velocity"])
w    = np.array(ad_286093["z_velocity"])
u_286092 = np.array(ad_286092["x_velocity"])
v_286092 = np.array(ad_286092["y_velocity"])
w_286092 = np.array(ad_286092["z_velocity"])
massflux_x = u*dens_286093
massflux_y = v*dens_286093
massflux_z = w*dens_286093
rhougrad = np.gradient(massflux_x,dxmin)
rhovgrad = np.gradient(massflux_y,dxmin)
rhowgrad = np.gradient(massflux_z,dxmin)

# Build the gradients using the new tools
mechreader = MechReader(
    mech_path = '/home/jb5027/cfd-solver/mechanisms/LiDryer/mechanism.inp',
)
input = Input(
    input_fname = '/home/jb5027/cavityflameanalysis/input_files_for_analysis/analysis.in'
)
input.parse_input_file()
mechreader.load_additional_transport_coefficient_data()
input.value_dict["nx1"] = int(npts_np[0] - 4)
input.value_dict["nx2"] = int(npts_np[1] - 4)
input.value_dict["nx3"] = int(npts_np[2] - 4)

# actual_lo = np.array([1.25, 0.0, 0.15])
# actual_hi = np.array([1.64726562, 0.39726563, 0.346875])
actual_lo=np.array([0.75,0.0,0.15])
actual_hi=np.array([1.34765625,0.39726563,0.346875])
# actual_lo = np.array([0.0, 0.0, 0.0])
# actual_hi = np.array([7.2, 1.8, 0.45])

actual_lo = actual_lo / 100.0
actual_hi = actual_hi / 100.0

input.value_dict["x1min"] = actual_lo[0]
input.value_dict["x2min"] = actual_lo[1]
input.value_dict["x3min"] = actual_lo[2]
input.value_dict["x1max"] = actual_hi[0]
input.value_dict["x2max"] = actual_hi[1]
input.value_dict["x3max"] = actual_hi[2]

mesh = Mesh(input, mechreader, np.float64)

mesh.dx1 = ((actual_hi[0] - actual_lo[0]) / (npts_np[0]))
mesh.dx2 = ((actual_hi[1] - actual_lo[1]) / (npts_np[1]))
mesh.dx3 = ((actual_hi[2] - actual_lo[2]) / (npts_np[2]))

CRHO_ID = mesh.CONS_ID["RHO"]
XMOM_ID = mesh.CONS_ID["XMOM"]
YMOM_ID = mesh.CONS_ID["YMOM"]
ZMOM_ID = mesh.CONS_ID["ZMOM"]
ENER_ID = mesh.CONS_ID["ENER"]
TEMP_ID = mesh.CONS_ID["TEMP"]
NUMCONS = mesh.NUM_CONS

PRHO_ID = mesh.PRIM_ID["RHO"]
XVEL_ID = mesh.PRIM_ID["XVEL"]
YVEL_ID = mesh.PRIM_ID["YVEL"]
ZVEL_ID = mesh.PRIM_ID["ZVEL"]
PRES_ID = mesh.PRIM_ID["PRES"]
NUMPRIM = mesh.NUM_PRIM

ng = mesh.ng
dx1 = mesh.dx1
dx2 = mesh.dx2
dx3 = mesh.dx3

N2_ID = mechreader.SPEC_ID["N2"]
O2_ID = mechreader.SPEC_ID["O2"]
H2_ID = mechreader.SPEC_ID["H2"]
O_ID = mechreader.SPEC_ID["O"]
H_ID = mechreader.SPEC_ID["H"]
OH_ID = mechreader.SPEC_ID["OH"]
HO2_ID = mechreader.SPEC_ID["HO2"]
H2O_ID = mechreader.SPEC_ID["H2O"]
H2O2_ID = mechreader.SPEC_ID["H2O2"]
NSPEC = mechreader.NUM_SPECS

temperature = np.array(ad_286093["temperature"])
yk = np.zeros((NSPEC, u.shape[0], u.shape[1], u.shape[2]))
yk[N2_ID] = ad_286093["rho_N2"] / ad_286093["density"]
yk[O2_ID] = ad_286093["rho_O2"] / ad_286093["density"]
yk[H2_ID] = ad_286093["rho_H2"] / ad_286093["density"]
yk[O_ID] = ad_286093["rho_O"] / ad_286093["density"]
yk[H_ID] = ad_286093["rho_H"] / ad_286093["density"]
yk[OH_ID] = ad_286093["rho_OH"] / ad_286093["density"]
yk[H2O_ID] = ad_286093["rho_H2O"] / ad_286093["density"]
yk[HO2_ID] = ad_286093["rho_HO2"] / ad_286093["density"]
yk[H2O2_ID] = ad_286093["rho_H2O2"] / ad_286093["density"]
eint = temp_yk_to_e(temperature, yk, mechreader)
enth = temp_yk_to_h(temperature, yk, mechreader)

temp_286092 = np.array(ad_286092["temperature"])
yk_286092 = np.zeros((NSPEC, u.shape[0], u.shape[1], u.shape[2]))
yk_286092[N2_ID] = ad_286092["rho_N2"] / ad_286092["density"]
yk_286092[O2_ID] = ad_286092["rho_O2"] / ad_286092["density"]
yk_286092[H2_ID] = ad_286092["rho_H2"] / ad_286092["density"]
yk_286092[O_ID] = ad_286092["rho_O"] / ad_286092["density"]
yk_286092[H_ID] = ad_286092["rho_H"] / ad_286092["density"]
yk_286092[OH_ID] = ad_286092["rho_OH"] / ad_286092["density"]
yk_286092[H2O_ID] = ad_286092["rho_H2O"] / ad_286092["density"]
yk_286092[HO2_ID] = ad_286092["rho_HO2"] / ad_286092["density"]
yk_286092[H2O2_ID] = ad_286092["rho_H2O2"] / ad_286092["density"]
eint_286092 = temp_yk_to_e(temp_286092, yk_286092, mechreader)
enth_286092 = temp_yk_to_h(temp_286092, yk_286092, mechreader)

# Need to first convert these values to SI units
rho_in = dens_286093 * (1.0e3)
u_in = u / 100.0
v_in = v / 100.0
w_in = w / 100.0
rho_in_286092 = dens_286092 * (1.0e3)
u_in_286092 = u_286092 / 100.0
v_in_286092 = v_286092 / 100.0
w_in_286092 = w_286092 / 100.0

mesh.Un[CRHO_ID, :, :, :] = rho_in
mesh.Un[XMOM_ID, :, :, :] = rho_in * u_in
mesh.Un[YMOM_ID, :, :, :] = rho_in * v_in
mesh.Un[ZMOM_ID, :, :, :] = rho_in * w_in
mesh.Un[ENER_ID, :, :, :] = rho_in * (eint + 0.5 * (u_in**2 + v_in**2 + w_in**2))
mesh.Un[TEMP_ID, :, :, :] = temperature
mesh.Un[NUMCONS:, :, :, :] = rho_in * yk

rho_E_286092 = rho_in_286092 * (eint_286092 + 0.5 * (u_in_286092**2 + v_in_286092**2 + w_in_286092**2))
rho_E_286093 = rho_in * (eint + 0.5 * (u_in**2 + v_in**2 + w_in**2))

rho_h_286092 = rho_in_286092 * enth_286092
rho_h_286093 = rho_in * enth

pres_286092 = rho_temp_yk_to_p(rho_in_286092, temp_286092, yk_286092, mechreader)
pres_286093 = rho_temp_yk_to_p(rho_in, temperature, yk, mechreader)
dp_dt = (pres_286093 - pres_286092) / dt
dp_dt = dp_dt[2:-2, 2:-2, 2:-2]

prims_x, prims_y, prims_z, fluxes_x, fluxes_y, fluxes_z, diff_fluxes_x, diff_fluxes_y, diff_fluxes_z = get_fluxes(mesh, input, mechreader)
prims_x, prims_y, prims_z, enth_flux_x, enth_flux_y, enth_flux_z, qx, qy, qz, viscous_heating = get_terms_for_analysis(mesh, input, mechreader)

# Continuity terms
updated_drhou_dx = (fluxes_x[PRHO_ID, 1:, :, :] - fluxes_x[PRHO_ID, :-1, :, :]) / mesh.dx1
updated_drhov_dy = (fluxes_y[PRHO_ID, :, 1:, :] - fluxes_y[PRHO_ID, :, :-1, :]) / mesh.dx2
updated_drhow_dz = (fluxes_z[PRHO_ID, :, :, 1:] - fluxes_z[PRHO_ID, :, :, :-1]) / mesh.dx3
updated_cont_conv = updated_drhou_dx + updated_drhov_dy + updated_drhow_dz

# X-momentum terms
xmom_terms = (fluxes_x[XMOM_ID, 1:, :, :] - fluxes_x[XMOM_ID, :-1, :, :]) / mesh.dx1 \
    + (fluxes_y[XMOM_ID, :, 1:, :] - fluxes_y[XMOM_ID, :, :-1, :]) / mesh.dx2 \
    + (fluxes_z[XMOM_ID, :, :, 1:] - fluxes_z[XMOM_ID, :, :, :-1]) / mesh.dx3 \
    + (diff_fluxes_x[XMOM_ID, 1:, :, :] - diff_fluxes_x[XMOM_ID, :-1, :, :]) / mesh.dx1 \
    + (diff_fluxes_y[XMOM_ID, :, 1:, :] - diff_fluxes_y[XMOM_ID, :, :-1, :]) / mesh.dx2 \
    + (diff_fluxes_z[XMOM_ID, :, :, 1:] - diff_fluxes_z[XMOM_ID, :, :, :-1]) / mesh.dx3
drhou_dt = drhou_dt[2:-2, 2:-2, 2:-2] * 10.0

xmom_terms = xmom_terms[:, 5, 5]
drhou_dt = drhou_dt[:, 5, 5]

# Y-momentum terms
ymom_terms = (fluxes_x[YMOM_ID, 1:, :, :] - fluxes_x[YMOM_ID, :-1, :, :]) / mesh.dx1 \
    + (fluxes_y[YMOM_ID, :, 1:, :] - fluxes_y[YMOM_ID, :, :-1, :]) / mesh.dx2 \
    + (fluxes_z[YMOM_ID, :, :, 1:] - fluxes_z[YMOM_ID, :, :, :-1]) / mesh.dx3 \
    + (diff_fluxes_x[YMOM_ID, 1:, :, :] - diff_fluxes_x[YMOM_ID, :-1, :, :]) / mesh.dx1 \
    + (diff_fluxes_y[YMOM_ID, :, 1:, :] - diff_fluxes_y[YMOM_ID, :, :-1, :]) / mesh.dx2 \
    + (diff_fluxes_z[YMOM_ID, :, :, 1:] - diff_fluxes_z[YMOM_ID, :, :, :-1]) / mesh.dx3
drhov_dt = drhov_dt[2:-2, 2:-2, 2:-2] * 10.0

ymom_terms = ymom_terms[:, 5, 5]
drhov_dt = drhov_dt[:, 5, 5]

# Z-momentum terms
zmom_terms = (fluxes_x[ZMOM_ID, 1:, :, :] - fluxes_x[ZMOM_ID, :-1, :, :]) / mesh.dx1 \
    + (fluxes_y[ZMOM_ID, :, 1:, :] - fluxes_y[ZMOM_ID, :, :-1, :]) / mesh.dx2 \
    + (fluxes_z[ZMOM_ID, :, :, 1:] - fluxes_z[ZMOM_ID, :, :, :-1]) / mesh.dx3 \
    + (diff_fluxes_x[ZMOM_ID, 1:, :, :] - diff_fluxes_x[ZMOM_ID, :-1, :, :]) / mesh.dx1 \
    + (diff_fluxes_y[ZMOM_ID, :, 1:, :] - diff_fluxes_y[ZMOM_ID, :, :-1, :]) / mesh.dx2 \
    + (diff_fluxes_z[ZMOM_ID, :, :, 1:] - diff_fluxes_z[ZMOM_ID, :, :, :-1]) / mesh.dx3
drhow_dt = drhow_dt[2:-2, 2:-2, 2:-2] * 10.0

zmom_terms = zmom_terms[:, 5, 5]
drhow_dt = drhow_dt[:, 5, 5]

# Energy terms
energy_terms = (fluxes_x[ENER_ID, 1:, :, :] - fluxes_x[ENER_ID, :-1, :, :]) / mesh.dx1 \
    + (fluxes_y[ENER_ID, :, 1:, :] - fluxes_y[ENER_ID, :, :-1, :]) / mesh.dx2 \
    + (fluxes_z[ENER_ID, :, :, 1:] - fluxes_z[ENER_ID, :, :, :-1]) / mesh.dx3 \
    + (diff_fluxes_x[ENER_ID, 1:, :, :] - diff_fluxes_x[ENER_ID, :-1, :, :]) / mesh.dx1 \
    + (diff_fluxes_y[ENER_ID, :, 1:, :] - diff_fluxes_y[ENER_ID, :, :-1, :]) / mesh.dx2 \
    + (diff_fluxes_z[ENER_ID, :, :, 1:] - diff_fluxes_z[ENER_ID, :, :, :-1]) / mesh.dx3
drhoE_dt = (rho_E_286093 - rho_E_286092) / dt
drhoE_dt = drhoE_dt[2:-2, 2:-2, 2:-2]

energy_terms = energy_terms[:, 5, 5]
drhoE_dt = drhoE_dt[:, 5, 5]

# Enthalpy terms
u_center = u_in[2:-2, 2:-2, 2:-2]
v_center = v_in[2:-2, 2:-2, 2:-2]
w_center = w_in[2:-2, 2:-2, 2:-2]
enthalpy_terms = (enth_flux_x[1:, :, :] - enth_flux_x[:-1, :, :]) / mesh.dx1 \
    + (enth_flux_y[:, 1:, :] - enth_flux_y[:, :-1, :]) / mesh.dx2 \
    + (enth_flux_z[:, :, 1:] - enth_flux_z[:, :, :-1]) / mesh.dx3 \
    - dp_dt \
    - u_center * (prims_x[PRES_ID, 1:, :, :] - prims_x[PRES_ID, :-1, :, :]) / mesh.dx1 \
    - v_center * (prims_y[PRES_ID, :, 1:, :] - prims_y[PRES_ID, :, :-1, :]) / mesh.dx2 \
    - w_center * (prims_z[PRES_ID, :, :, 1:] - prims_z[PRES_ID, :, :, :-1]) / mesh.dx3 \
    + (qx[1:, :, :] - qx[:-1, :, :]) / mesh.dx1 \
    + (qy[:, 1:, :] - qy[:, :-1, :]) / mesh.dx2 \
    + (qz[:, :, 1:] - qz[:, :, :-1]) / mesh.dx3 \
    + viscous_heating
drhoh_dt = (rho_h_286093 - rho_h_286092) / dt
drhoh_dt = drhoh_dt[2:-2, 2:-2, 2:-2]

enthalpy_terms = enthalpy_terms[:, 5, 5]
drhoh_dt = drhoh_dt[:, 5, 5]

# Old method
cont_conv = rhougrad[0] + rhovgrad[1] + rhowgrad[2]
continuity = cont_conv + drho_dt

# Check over x
continuity_over_x = continuity[:, 5, 7]
cont_conv_over_x = cont_conv[:, 5, 7]
drho_dt_over_x = drho_dt[:, 5, 7]

updated_cont_conv_over_x = updated_cont_conv[:, 5, 5]
drho_dt_over_x_john = drho_dt[2:-2, 2:-2, 2:-2] * 1000.0
updated_continuity = drho_dt_over_x_john + updated_cont_conv
drho_dt_over_x_john = drho_dt_over_x_john[:, 5, 5]
updated_continuity_over_x = updated_continuity[:, 5, 5]

test_cont = updated_cont_conv + drho_dt[2:-2, 2:-2, 2:-2] * 1000.0

print(f"Max of continuity over x at (yindex, zindex)=(5,28) : {np.max(updated_continuity_over_x):.4e} [kg/m^3s]")

# Plot to compare
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(
    continuity_over_x,
    label='drho_dt + drhouj_dxj',
    markersize=3
)
ax.plot(
    drho_dt_over_x,
    label='drho_dt',
    markersize=3,
)
ax.plot(
    cont_conv_over_x,
    label='drhouj_dxj',
    markersize=3,
)
ax.set_xlabel('x values at set (y,z)')
ax.set_ylabel('Continuity on grid points')
ax.legend()
plt.savefig(f"full_case_cont_over_x_haritechnique.png")
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(
    updated_continuity_over_x,
    label='drho_dt + drhouj_dxj',
    markersize=3
)
ax.plot(
    drho_dt_over_x_john,
    label='drho_dt',
    markersize=3,
)
ax.plot(
    updated_cont_conv_over_x,
    label='drhouj_dxj',
    markersize=3,
)
ax.set_xlabel('x values at set (y,z)')
ax.set_ylabel('Continuity on grid points')
ax.legend()
plt.savefig(f"full_case_cont_over_x_johntechnique.png")
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
plt.axhline(y=0.0, color='gray', linestyle='--')
ax.plot(
    drhou_dt + xmom_terms,
    label='Residual',
    markersize=5
)
ax.plot(
    drhou_dt,
    label='drhou_dt',
    markersize=5
)
ax.plot(
    xmom_terms,
    label='Other terms',
    markersize=5
)
ax.set_ylabel('X-momentum on grid points')
ax.legend()
plt.savefig(f"full_case_xmom.png")
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
plt.axhline(y=0.0, color='gray', linestyle='--')
ax.plot(
    drhov_dt + ymom_terms,
    label='Residual',
    markersize=5
)
ax.plot(
    drhov_dt,
    label='drhov_dt',
    markersize=5
)
ax.plot(
    ymom_terms,
    label='Other terms',
    markersize=5
)
ax.set_ylabel('Y-momentum on grid points')
ax.legend()
plt.savefig(f"full_case_ymom.png")
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
plt.axhline(y=0.0, color='gray', linestyle='--')
ax.plot(
    drhow_dt + zmom_terms,
    label='Residual',
    markersize=5
)
ax.plot(
    drhow_dt,
    label='drhow_dt',
    markersize=5
)
ax.plot(
    zmom_terms,
    label='Other terms',
    markersize=5
)
ax.set_ylabel('Z-momentum on grid points')
ax.legend()
plt.savefig(f"full_case_zmom.png")
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
plt.axhline(y=0.0, color='gray', linestyle='--')
ax.plot(
    drhoE_dt + energy_terms,
    label='Residual',
    markersize=5
)
ax.plot(
    drhoE_dt,
    label='drhoE_dt',
    markersize=5
)
ax.plot(
    energy_terms,
    label='Other terms',
    markersize=5
)
ax.set_ylabel('Total energy on grid points')
ax.legend()
plt.savefig(f"full_case_ener.png")
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
plt.axhline(y=0.0, color='gray', linestyle='--')
ax.plot(
    drhoh_dt + enthalpy_terms,
    label='Residual',
    markersize=5
)
ax.plot(
    drhoh_dt,
    label='drhoh_dt',
    markersize=5
)
ax.plot(
    enthalpy_terms,
    label='Other terms',
    markersize=5
)
ax.set_ylabel('Enthalpy on grid points')
ax.legend()
plt.savefig(f"full_case_enth.png")
plt.close()