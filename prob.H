#ifndef PROB_H
#define PROB_H

#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Random.H>

#include "mechanism.H"

#include "PeleC.H"
#include "IndexDefines.H"
#include "Constants.H"
#include "PelePhysics.H"
#include "Tagging.H"

#include "ProblemSpecificFunctions.H"
#include "prob_parm.H"
#include "Utilities.H"
#include "turbinflow.H"
#include <cmath>
#include <fstream>
#include <iostream>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    amrex::Real velocity[3] = {0.0};
    amrex::Real temperature = prob_parm.reac_temp;
    amrex::Real pressure = prob_parm.reac_pres;
    amrex::Real rho;
    amrex::Real eint;

    // Find location in domain
    amrex::Real xval = geomdata.ProbLo(0) + (amrex::Real(i) + 0.5) * geomdata.CellSize(0);
    amrex::Real yval = geomdata.ProbLo(1) + (amrex::Real(j) + 0.5) * geomdata.CellSize(1);
    // amrex::Real middle_x_val = (geomdata.ProbHi(0) - geomdata.ProbLo(0)) / 2.0;

    // Default to Air
    massfrac[N2_ID] = 0.767;  
    massfrac[H_ID] = 0.0;
    massfrac[O2_ID] = 0.233; 
    massfrac[O_ID] =  0.0;
    massfrac[OH_ID] = 0.0;
    massfrac[H2_ID] = 0.0; 
    massfrac[H2O_ID] = 0.0;
    massfrac[HO2_ID] = 0.0;
    massfrac[H2O2_ID] = 0.0;
    amrex::Real sum = 0.0;
    for (int n = 0; n < NUM_SPECIES; n++) {
      sum+= massfrac[n];
    }
    amrex::Real extra_for_sum_species_check = 1.0 - sum;
    massfrac[H2O_ID] = massfrac[H2O_ID] + extra_for_sum_species_check;
    
    velocity[0] = 0.0; // before 2327.5e2 (units in cm/s?) now zero
    velocity[1] = 0.0;
    velocity[2] = 0.0;

    temperature = 298.15;
    pressure = 1013250.0;
    eos.PYT2R(pressure, massfrac, temperature, rho);
    eos.RTY2E(rho, temperature, massfrac, eint);

    // Overwrite if in ignition region 
    if (xval <= 0.1 && yval <= 1.2) {
        // Burned / just water
        //massfrac[N2_ID] = 0.7451;
        massfrac[N2_ID] = 0.0;
        massfrac[H_ID] = 0.0;
        massfrac[O2_ID] = 0.0;
        massfrac[O_ID] = 0.0;
        massfrac[OH_ID] = 0.0;
        massfrac[H2_ID] = 0.0;
        //massfrac[H2O_ID] = 0.2549;
        massfrac[H2O_ID] = 1.0;
        massfrac[HO2_ID] =  0.0;
        massfrac[H2O2_ID] = 0.0;
        amrex::Real sum = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          sum+= massfrac[n];
        }
        amrex::Real extra_for_sum_species_check = 1.0 - sum;
        massfrac[O2_ID] = massfrac[O2_ID] + extra_for_sum_species_check;
        // velocity[0] = 2836.19e2; // before 2327.5e2 (units in cm/s?) might need to change this
        // temperature = 3676.76;
        // pressure = 1899976.032354744e1; // multiply by ten if in Pa (units?)
        velocity[0] = 2836.19e2; // before 2327.5e2 (units in cm/s?) might need to change this
        //velocity[0] = 0;
        temperature = 2500;
        pressure = 4.053e7; // multiply by ten if in Pa (units?)
        eos.PYT2R(pressure, massfrac, temperature, rho);
        eos.RTY2E(rho, temperature, massfrac, eint);
      }
      else if (xval <= 8 && yval <= 1.2) {
        // Unburned stoichiometric
        if (prob_parm.equiv_ratio == 1.0) {
          massfrac[N2_ID] = 0.7451;
          massfrac[H_ID] = 0.0;
          massfrac[O2_ID] = 0.2264;
          massfrac[O_ID] = 0.0;
          massfrac[OH_ID] = 0.0;
          massfrac[H2_ID] = 0.0285;
          massfrac[H2O_ID] = 0.0;
          massfrac[HO2_ID] =  0.0;
          massfrac[H2O2_ID] = 0.0;
          amrex::Real sum = 0.0;
          for (int n = 0; n < NUM_SPECIES; n++) {
            sum+= massfrac[n];
          }
          amrex::Real extra_for_sum_species_check = 1.0 - sum;
          massfrac[O2_ID] = massfrac[O2_ID] + extra_for_sum_species_check;
          velocity[0] = 0.0;
          temperature = 298.15;
          pressure = 1013250.0;
          eos.PYT2R(pressure, massfrac, temperature, rho);
          eos.RTY2E(rho, temperature, massfrac, eint);
        }
        else{
          amrex::Print() << "Only stoichiometric currently supported" << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    // Set the state
    state(i, j, k, URHO) = rho;
    state(i, j, k, UMX) = rho * velocity[0];
    state(i, j, k, UMY) = rho * velocity[1];
    state(i, j, k, UMZ) = rho * velocity[2];
    state(i, j, k, UEINT) = rho * eint;
    state(i, j, k, UEDEN) =
      rho * (eint + 0.5 * (velocity[0] * velocity[0] + velocity[1] * velocity[1] +
                          velocity[2] * velocity[2])
    );
    state(i, j, k, UTEMP) = temperature;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, UFS + n) = rho * massfrac[n];
    }
  }

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NVAR],
  amrex::Real s_ext[NVAR],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& /*turb_fluc*/)
{
  // std::ofstream outputFile("output.txt", std::ios::app);

  // if (outputFile.is_open()) {
   // check if the file was opened successfully
   if (prob_parm.inject_fuel) {
    //inlet boundary condition
     if (idir==1) {
      // at y low boundary
        if (sgn==1) {
        //check at an inlet
        amrex::Real num_inlet = prob_parm.num_inlet;
        // Find location in domain
        amrex::Real max_xval = geomdata.ProbHi(0);
        //amrex::Real wall_len = max_xval/(num_inlet*5 + 2);
        amrex::Real wall_len = max_xval/(num_inlet*5);
        amrex::Real x_val = x[0];
        amrex::Real eint;
        // this is kinda like skipping a time step I think actually - better to just let it default to wall
        // if (x_val<0) {
        //   x_val = max_xval + x_val;
        // }
         if ((fmod(abs(x_val), wall_len*5) > wall_len*2)) {
          auto eos = pele::physics::PhysicsType::eos();
          amrex::Real p_int;
          amrex::Real massfrac_int[NUM_SPECIES] = {0.0};
          amrex::Real massfrac_ext[NUM_SPECIES] = {0.0};
          amrex::Real inj_t0 = prob_parm.inj_t0;
          amrex::Real inj_p0 = prob_parm.inj_p0;
          amrex::Real R = prob_parm.inj_R;
          amrex::Real p_ext;
          amrex::Real p_cr;
          amrex::Real vx_in = 0.0;
          amrex::Real vy_in = 0.0;
          amrex::Real vz_in = 0.0;
          amrex::Real rho = s_ext[URHO];

          for (int n = 0; n < NUM_SPECIES; n++) {
            massfrac_int[n] = s_int[UFS + n] / s_int[URHO];
          }
        
          // calculate pressure on the inside from known values
          eos.RTY2P(s_int[URHO], s_int[UTEMP], massfrac_int, p_int);
          amrex::Real Cp = 0;
          amrex::Real Cv = 0;
          amrex::Real gamma = 0;
          eos.TY2Cp(s_int[UTEMP],massfrac_int,Cp);
          eos.TY2Cv(s_int[UTEMP],massfrac_int,Cv);
          gamma = Cp/Cv;

          p_cr = inj_p0*pow(2/(gamma+1),(gamma)/(gamma-1));
          // no injection
          if (p_int >= inj_p0) {
             p_ext = p_int;
             s_ext[UTEMP] = inj_t0 * pow((p_ext / inj_p0),(gamma-1)/gamma);
             vy_in = 0;
             s_ext[UMX] = rho * vx_in;
             s_ext[UMY] = rho * vy_in; // keep zero
             s_ext[UMZ] = rho * vz_in;
             s_ext[UEDEN] = s_ext[UEINT] +
              0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
          }
          // subsonic injection
          else if (p_int>p_cr){
              p_ext = p_int;
              s_ext[UTEMP] = inj_t0 * pow((p_ext / inj_p0),(gamma-1)/gamma);
              massfrac_ext[N2_ID] = 0.7451;
              massfrac_ext[O2_ID] = 0.2264;
              massfrac_ext[H2_ID] = 0.0285;
          amrex::Real sum = 0.0;
              //eos.PYT2RE(prob_parm.reac_pres, massfrac_ext, s_ext[UTEMP], rho, eint);
              eos.PYT2RE(p_ext, massfrac_ext, s_ext[UTEMP], rho, eint);
              for (int n = 0; n < NUM_SPECIES; n++) {
                s_ext[UFS + n] = rho * massfrac_ext[n];
              }
              vy_in = sqrt(((2*gamma)/(gamma-1))*R*inj_t0*(1 - pow((p_ext / inj_p0),(gamma-1)/gamma)));
              s_ext[UMX] = rho * vx_in;
              s_ext[UMY] = rho * vy_in; 
              s_ext[UMZ] = rho * vz_in;
              s_ext[UEINT] = rho * eint;
              s_ext[UEDEN] = s_ext[UEINT] + 0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
          }
          // sonic injection
           else {
              p_ext = p_cr;
              s_ext[UTEMP] = inj_t0 * (2/(gamma+1));
              massfrac_ext[N2_ID] = 0.7451;
              massfrac_ext[O2_ID] = 0.2264;
              massfrac_ext[H2_ID] = 0.0285;
              eos.PYT2RE(p_ext, massfrac_ext, s_ext[UTEMP], rho, eint);
              for (int n = 0; n < NUM_SPECIES; n++) {
                s_ext[UFS + n] = rho * massfrac_ext[n];
              }
              vy_in = sqrt(((2*gamma)/(gamma+1))*R*inj_t0);
              s_ext[UMX] = rho * vx_in;
              s_ext[UMY] = rho * vy_in; 
              s_ext[UMZ] = rho * vz_in;
              s_ext[UEINT] = rho * eint;
              s_ext[UEDEN] = s_ext[UEINT] + 0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
           }
        }
       }
     }
  }

  // outlet boundary condition
 if (idir == 1) {
  // at y high boundary
    if ( sgn== -1) {
      auto eos = pele::physics::PhysicsType::eos();
      amrex::Real p_int;
      amrex::Real massfrac_int[NUM_SPECIES] = {0.0};
      amrex::Real massfrac_ext[NUM_SPECIES] = {0.0};
      // we assume pressure on the outer boundary is equal to the reactant pressure (1 atm)
      amrex::Real p_ext = prob_parm.p_outlet;
      amrex::Real c0;
      // initialize mass fractions
      for (int n = 0; n < NUM_SPECIES; n++) {
        massfrac_int[n] = s_int[UFS + n] / s_int[URHO];
      }
      eos.RTY2Cs(s_int[URHO], s_int[UTEMP], massfrac_int, c0);
      for (int n = 0; n < NUM_SPECIES; n++) {
        massfrac_ext[n] = massfrac_int[n];
      }
      if (s_int[UMY]/s_int[URHO] < c0) {
        amrex::Real Cp = 0;
        amrex::Real Cv = 0;
        amrex::Real gamma = 0;
        eos.TY2Cp(s_int[UTEMP],massfrac_int,Cp);
        eos.TY2Cv(s_int[UTEMP],massfrac_int,Cv);
        gamma = Cp/Cv;
        // calculate pressure on the inside from known values
        eos.RTY2P(s_int[URHO], s_int[UTEMP], massfrac_int, p_int);
        
        // calculate stagnation speed of sound, density, and temperature
        //amrex::Real R = p_int/(s_int[URHO]*s_int[UTEMP]); // probably can get this in input but idk how (gas constant) assumes gas constant...
        amrex::Real R = Cp - Cv;
        amrex::Real vel = sqrt((s_int[UEDEN] - s_int[UEINT])/(0.5 * s_int[URHO])); // find the overall velocity
        //amrex::Real vel = s_int[UMY]/s_int[URHO] // this is just y velocity
        amrex::Real mach = vel / c0;
        amrex::Real rho0 = s_int[URHO]* (pow((1 + ((gamma-1)/2)*pow(mach,2)),1/(gamma-1)));
        amrex::Real rho, eint;
        // interpolate value for rho
        rho = s_int[URHO] + (p_ext - p_int)/c0;
        s_ext[URHO] = rho;
        // set mass fractions equal to inside now we have density
        for (int n = 0; n < NUM_SPECIES; n++) {
            s_ext[UFS + n] = rho * massfrac_ext[n];
        }
        // find temperature and energy from our known values
        eos.RYP2T(rho, massfrac_ext, p_ext, s_ext[UTEMP]);
        eos.RYP2E(rho, massfrac_ext, p_ext, eint);
        s_ext[UEINT] = rho * eint;

        // interpolate values for the velocities
        s_ext[UMX] = rho * (s_int[UMX]/ s_int[URHO] + sgn*(p_int - p_ext)/(rho0*c0));
        s_ext[UMY] = rho * (s_int[UMY]/ s_int[URHO] + (p_int - p_ext)/(rho0*c0));
        s_ext[UMZ] = rho * (s_int[UMZ]/ s_int[URHO] + (p_int - p_ext)/(rho0*c0));
        s_ext[UEDEN] =
        s_ext[UEINT] +
            0.5 * rho * ((s_ext[UMX]/rho) * (s_ext[UMX]/rho) + (s_ext[UMY]/rho) * (s_ext[UMY]/rho) + (s_ext[UMZ]/rho) * (s_ext[UMZ]/rho));
    }
    else {
      s_ext[URHO] = s_int[URHO];
      s_ext[UEINT] = s_int[UEINT];
      s_ext[UTEMP] = s_int[UTEMP];
      s_ext[UMX] = s_int[UMX];;
      s_ext[UMY] = s_int[UMY];
      s_ext[UMZ] = s_int[UMZ];
      s_ext[UEDEN] = s_int[UEDEN];
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[UFS + n] =  s_int[URHO] * massfrac_ext[n];
      }
    }
    }
   }
}

void pc_prob_close();

using ProblemSpecificFunctions = DefaultProblemSpecificFunctions;

#endif