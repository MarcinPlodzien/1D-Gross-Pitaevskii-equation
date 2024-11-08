#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:48:43 2024
@author: Marcin Płodzień
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Physical constants
a0 = 0.529e-10  # Bohr radius (m)
hbar = 1.05457e-34  # Reduced Planck's constant (J.s)
m_proton = 1.6726e-27  # Proton mass (kg)
G_SI = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Rubidium-87 parameters
a_s = 103 * a0  # s-wave scattering length for Rubidium-87 (m)
m_Rb87 = 87 * m_proton  # Rb-87 mass (kg)

# Trap parameters
omega_x = 14 * 2 * np.pi  # Angular frequency along x-axis (rad/s)
omega_perp = 425 * 2 * np.pi  # Angular frequency along perpendicular axes (rad/s)
omega_y = omega_perp
omega_z = omega_perp

# atomic units
E0 = hbar * omega_x
l0 = np.sqrt(hbar / (m_Rb87 * omega_x))
t0 = 1.0 / omega_x


# Physical parameters
N_particles = 1e2 # Number of Rb-87 atoms in BEC
L_SI = 120e-6  # Experimental system size (m)
L = L_SI / l0
n0 = N_particles / L
g_3D_SI = 4 * np.pi * hbar**2 / m_Rb87 * a_s
g_1D_SI = 2 * a_s * np.sqrt(omega_z * omega_y) / omega_x
g_1D = g_1D_SI / l0
mu = n0 * g_1D
R_TF = np.sqrt(2 * mu)  # Thomas-Fermi radius
ksi_BEC = 1.0 / np.sqrt(2 * mu)  # BEC healing length

t_R = ksi_BEC**2

# Grid parameters
dx = ksi_BEC / 20  # Condition for obtaining good BEC ground state
Nx = 2**int(np.ceil(np.log2(L / dx + 1)))  # Power of 2 for FFT calculations
dx = L / (Nx - 1)
x_vec = np.linspace(-L / 2, L / 2, Nx)
dt = 0.1 * dx**2  # Nonlinear evolution needs small dt << dx^2
 
# Grids in momentum space for Split-Operator method
dkx = 2 * np.pi / L
norm_factor_momentum = np.sqrt(dx / Nx / dkx)
momentum_kx = np.zeros(Nx)
for i in range(Nx // 2):
    momentum_kx[i] = i * 2 * np.pi / L
for i in range(Nx // 2, Nx):
    momentum_kx[i] = (i - Nx) * 2 * np.pi / L

momentum_propagator_free_dt = np.exp(-1j * dt * (momentum_kx**2) / 2)


# BEC initial state as Thomas-Fermi profile
psi_BEC_TF = np.zeros(Nx, dtype=complex)
for i in range(Nx):
    x = x_vec[i]
    if R_TF > abs(x):
        psi_BEC_TF[i] = np.sqrt((R_TF**2 - x**2) / (2 * g_1D * N_particles))

norm = np.sum(np.abs(psi_BEC_TF)**2 * dx)
psi_BEC_TF /= np.sqrt(norm)
rho_BEC_TF = np.abs(psi_BEC_TF)**2

# Imaginary time evolution of mean-field Hamiltonian
sigma = ksi_BEC
V_trap = 0.5 * x_vec**2  
fig, ax = plt.subplots(1,1, figsize = (10,10))

ax.plot(x_vec, V_trap)
# ax.set_xlim(-5,5)
plt.show()


# Kinetic energy Hamiltonian in continous space
H_0 = (1 / dx**2) * np.eye(Nx) + np.diag(-1 / (2 * dx**2) * np.ones(Nx - 1), -1) + np.diag(-1 / (2 * dx**2) * np.ones(Nx - 1), 1)



 

# Calculating BEC energy in TF ground state
H_psi_BEC_TF = H_0 @ psi_BEC_TF
energy_BEC_ground_state_TF = np.conj(psi_BEC_TF) @ H_psi_BEC_TF * dx
energy = energy_BEC_ground_state_TF

# Imaginary time evolution parameters
normalization_step = 100
epsilon = 1e-5
time_i = 1
energy_change_velocity = 1
psi = psi_BEC_TF.copy()


on_off_trap_GS = 1
on_off_interactions_GS = 1
# Imaginary time evolution loop to obtain ground state with interactions
while energy_change_velocity > epsilon:
    time_i += 1
    # Calculate the convolution term for dipolar interactions
    rho = np.abs(psi)**2
    H = H_0 + np.diag(on_off_trap_GS*V_trap + on_off_interactions_GS*rho)
    H_psi = H @ psi
    psi -= dt * H_psi
    norm = np.sum(np.abs(psi)**2) * dx
    psi /= np.sqrt(norm)
    
    if time_i % normalization_step == 0:
        energy_new = np.conj(psi) @ H_psi * dx
        energy_change_velocity = abs((energy_new - energy) / (normalization_step * dt))
        print(f' E = {energy_new} | dE/dt = {energy_change_velocity}')
        energy = energy_new

psi_BEC_ground_state = psi
rho_BEC_ground_state = np.abs(psi_BEC_ground_state)**2

# Time evolution with split-step method
psi_space = psi_BEC_ground_state.copy()  # Initial state for time-evolution
psi_0_momentum_k_x = np.fft.fft(psi_space) / np.sqrt(2 * np.pi)


#%%

def prepare_RK45_evolution_in_dt(H, psi, dt):
    """
    Evolve the wave function using the 4th-order Runge-Kutta (RK45) method.
    """
    # Compute intermediate steps K1, K2, K3, K4 for RK45
    psi_dt = psi.copy()
    K1 = -1j * dt * H @ psi_dt
    psi_tmp1 = psi_dt + 0.5 * K1
    K2 = -1j * dt * H @ psi_tmp1
    psi_tmp2 = psi_dt + 0.5 * K2
    K3 = -1j * dt * H @ psi_tmp2
    psi_tmp3 = psi_dt + K3
    K4 = -1j * dt * H @ psi_tmp3

    # Combine to get the final evolution step
    psi_t_dt = psi_dt + (K1 + 2.0 * K2 + 2.0 * K3 + K4) / 6.0
    return psi_t_dt 



# Time evolution parameters
t_max = 5
Nt = int(np.floor(t_max / dt + 1))
N_shots = 100
time_shot = int(Nt/N_shots)

 # Time evolution loop
counter = 0
on_off_trap = 0
on_off_interactions = 0
rho_split_step_vs_t_list = []
rho_RK45_vs_t_list = []
psi_space_split_step = psi_BEC_ground_state.copy()  # Initial state for time-evolution
psi_space_RK45 = psi_BEC_ground_state.copy()  # Initial state for time-evolution
for n in range(Nt + 1):
    t = n * dt
    
    ####################################################################
    # Two distinct method for evolving Schroedinger equation
    #
    ####################################################################
    
    ####################################################################
    # # 1. Split-operator Fourier method for time evolution
    # #
    # # Evolve with kinetic energy in momentum space
    psi_momentum = np.fft.fft(psi_space_split_step)
    psi_momentum *= momentum_propagator_free_dt
    psi_space_split_step = np.fft.ifft(psi_momentum)

    
    # Evolve with potential energy in real space
    V_split_step = on_off_trap*V_trap + on_off_interactions*np.abs(psi_space_split_step)**2
    psi_space_split_step *= np.exp(-1j * dt * V_split_step)
    ####################################################################
    

    ####################################################################
    ## 2. Runge-Kutta 45 algorithm for solving Schroedinger equation
    V_RK45 = on_off_trap*V_trap + on_off_interactions*np.abs(psi_space_RK45)**2
    H = H_0 + np.diag(V_RK45)
    
    
    psi_space_RK45 = prepare_RK45_evolution_in_dt(H, psi_space_RK45, dt)

    ####################################################################
    
    
    if(np.mod(n, time_shot) == 0):               
        counter += 1      
        rho = np.abs(psi_space)**2                    
        rho_split_step_vs_t_list.append((t, np.abs(psi_space_split_step)**2))
        rho_RK45_vs_t_list.append((t, np.abs(psi_space_RK45)**2))
 
 
 

FontSize = 20

images = []  # List to store frames as images
duration = 200
# Loop over each time step and create a frame

fig, ax = plt.subplots(1, 1)
for k in range(0, len(rho_split_step_vs_t_list)):
    ax.clear()  # Clear the previous plot
    
    
    t = rho_split_step_vs_t_list[k][0]
    rho_split_step_t = rho_split_step_vs_t_list[k][1]
    norm_t = np.sum(rho_split_step_t)*dx
    
    rho_RK45_t = rho_RK45_vs_t_list[k][1]
    norm_t = np.sum(rho_RK45_t)*dx
    
    print(t)
    
    title_string = "Initial state parameters : trap: {:d} | interactions: {:d}".format(on_off_trap_GS, on_off_interactions_GS)
   
    ax.set_title(title_string)
    ax.plot(x_vec, rho_BEC_ground_state, '.-', color = 'black', label='Initial state: Ground State')
    label_string = "trap: {:d} | interactions : {:d}".format(on_off_trap, on_off_interactions)
    ax.plot(x_vec, rho_split_step_t, color = 'blue', lw = 3, label=r" | split-step: $t = {:2.2f} | norm = {:2.2f}$".format(t,norm))
    ax.plot(x_vec, rho_RK45_t, color='red', ls = '--', lw = 2, label=r" | RK45: $t = {:2.2f} | norm = {:2.2f}$".format(t,norm))
  
    
    ax.text(0.1, 0.9, label_string, transform = ax.transAxes)
    y_max = np.max(rho_BEC_ground_state)
    ax.set_ylim([0, y_max])
    ax.legend()
    
   
    # Draw the plot but do not show it
    fig.canvas.draw()

    # Convert the plot to an image and append to the list
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(Image.fromarray(image))

# Save all images as an animated GIF

path_gif = './'
filename_animation = "rho_vs_t_on_off_trap.{:d}_on_off_interactions.{:d}.gif".format(on_off_trap, on_off_interactions)
filename = path_gif + filename_animation
images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)

# Close the plot to free resources
plt.close()
print(f"Animated GIF saved as {filename}")