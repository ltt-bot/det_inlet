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

    // Default to Oxygen
    massfrac[N2_ID] = 0.0;  // no nitrogen
    massfrac[H_ID] = 0.0;
    massfrac[O2_ID] = 1.0; 
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
    if (xval <= 0.09375 && yval <= 1.5) {
        // Burned 
        massfrac[N2_ID] = 0.0;  // no nitrogen
        massfrac[H_ID] = 0.00556;
        massfrac[O2_ID] = 0.10329; 
        massfrac[O_ID] =  0.04124;
        massfrac[OH_ID] = 0.16588;
        massfrac[H2_ID] = 0.02253; 
        massfrac[H2O_ID] = 0.66105;
        massfrac[HO2_ID] = 0.00042;
        massfrac[H2O2_ID] = 0.00005;
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
        temperature = 2500;
        pressure = 4.053e7; // multiply by ten if in Pa (units?)
        eos.PYT2R(pressure, massfrac, temperature, rho);
        eos.RTY2E(rho, temperature, massfrac, eint);
      }
      else if (xval <= 10 && yval <= 1.5) {
        // Unburned stoichiometric
        if (prob_parm.equiv_ratio == 1.0) {
          massfrac[N2_ID] = 0.0;
          massfrac[H_ID] = 0.0;
          massfrac[O2_ID] = 0.88809;
          massfrac[O_ID] = 0.0;
          massfrac[OH_ID] = 0.0;
          massfrac[H2_ID] = 0.1119;
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
  amrex::GeometryData const& /*geomdata*/,
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
        // amrex::Real max_xval = geomdata.ProbHi(0);
        amrex::Real max_xval = 16.;
        amrex::Real wall_len = max_xval/(num_inlet*5);
        amrex::Real x_val = x[0];
        amrex::Real eint;
         if ((fmod(x_val, wall_len*5) > wall_len*2) && x_val > 0) {
          // outputFile << fmod(wall_len*5, abs(x_val)); // write data to the file
          // outputFile << " , x_val: "; // write data to the file
          // outputFile << x_val; // write data to the file
          // outputFile << "\n"; // write data to the file
          auto eos = pele::physics::PhysicsType::eos();
          amrex::Real p_int;
          amrex::Real massfrac_int[NUM_SPECIES] = {0.0};
          amrex::Real massfrac_ext[NUM_SPECIES] = {0.0};
          amrex::Real gamma = 1.4;
          amrex::Real inj_t0 = prob_parm.inj_t0;
          amrex::Real inj_p0 = prob_parm.inj_p0;
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
          amrex::Real R = p_int/(s_int[URHO]*s_int[UTEMP]); // probably can get this in input but idk how (gas constant)

          p_cr = inj_p0*pow(2/(gamma+1),(gamma-1)/gamma);
          // no injection
          // if (p_int >= inj_p0) {
            //  p_ext = p_int;
            //  s_ext[UTEMP] = inj_t0 * pow((p_ext / inj_p0),(gamma-1)/gamma);
            //  vy_in = 0;
            //  s_ext[UMX] = rho * vx_in;
            //  s_ext[UMY] = rho * vy_in; // keep zero
            //  s_ext[UMZ] = rho * vz_in;
            //  s_ext[UEDEN] = s_ext[UEINT] +
            //   0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
          // }
          // // subsonic injection
          // else if (p_int>p_cr){
            //   p_ext = p_int;
            //   s_ext[UTEMP] = inj_t0 * pow((p_ext / inj_p0),(gamma-1)/gamma);
            //   vy_in = sqrt(((2*gamma)/(gamma-1))*R*inj_t0*(1 - pow((p_ext / inj_p0),(gamma-1)/gamma)));
            //   s_ext[UMX] = rho * vx_in;
            //   s_ext[UMY] = rho * vy_in; 
            //   s_ext[UMZ] = rho * vz_in;
            //   s_ext[UEINT] = rho * eint;
            //  //  s_ext[UEDEN] = s_ext[UEINT] + 0.5 * s_ext[UMX] * s_ext[UMX] / rho;
            //   //s_ext[UEDEN] = s_ext[UEINT] + 0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
            //   massfrac_ext[O2_ID] = 0.88809;
            //   massfrac_ext[H2_ID] = 0.1119;
            //   eos.PYT2RE(prob_parm.reac_pres, massfrac_ext, s_ext[UTEMP], rho, eint);
            //   s_ext[UEINT] = rho * eint;
            //   s_ext[UEDEN] = s_ext[UEINT] + 0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
            //   for (int n = 0; n < NUM_SPECIES; n++) {
            //    s_ext[UFS + n] = rho * massfrac_ext[n];
            //  } 
          // }
          // // sonic injection
          //  else {
              p_ext = p_cr;
              s_ext[UTEMP] = inj_t0 * (2/(gamma+1));
              vy_in = sqrt(((2*gamma)/(gamma-1))*R*inj_t0);
              s_ext[UMX] = rho * vx_in;
              s_ext[UMY] = rho * vy_in; 
              s_ext[UMZ] = rho * vz_in;
              massfrac_ext[O2_ID] = 0.88809;
              massfrac_ext[H2_ID] = 0.1119;
              eos.PYT2RE(prob_parm.reac_pres, massfrac_ext, s_ext[UTEMP], rho, eint);
              s_ext[UEINT] = rho * eint;
              //s_ext[UEDEN] = s_ext[UEINT] + 0.5 * rho * (vx_in * vx_in + vy_in * vy_in + vz_in * vz_in);
              s_ext[UEDEN] = s_ext[UEINT] + 0.5 * s_ext[UMY] * s_ext[UMY] / rho;
              for (int n = 0; n < NUM_SPECIES; n++) {
              s_ext[UFS + n] = rho * massfrac_ext[n];
            }
          // }
        }
       }
     }
  }
   
    // outputFile.close(); // close the file when done
    // std::cout << "Data was written to output.txt\n";
  // }
  // else {
  //   std::cerr << "Error opening file\n";
  // }
  
  // output boundary condition
//   // get rid of this since I wanted to make sure this was on
//  if (idir == 1) {
//   // at y high boundary
//     if ( sgn== -1) {
//       auto eos = pele::physics::PhysicsType::eos();
//       amrex::Real p_int;
//       amrex::Real massfrac_int[NUM_SPECIES] = {0.0};
//       amrex::Real massfrac_ext[NUM_SPECIES] = {0.0};
//       amrex::Real gamma = 1.4;
//       // we assume pressure on the outer boundary is equal to the reactant pressure (1 atm)
//       amrex::Real p_ext = prob_parm.reac_pres;
//       // initialize mass fractions
//       for (int n = 0; n < NUM_SPECIES; n++) {
//         massfrac_int[n] = s_int[UFS + n] / s_int[URHO];
//       }
      
//       // calculate pressure on the inside from known values
//       eos.RTY2P(s_int[URHO], s_int[UTEMP], massfrac_int, p_int);
//       // massfrac_ext[NUM_SPECIES] = massfrac_int[NUM_SPECIES];
//       for (int n = 0; n < NUM_SPECIES; n++) {
//         massfrac_ext[n] = massfrac_int[n];
//       }
      
//       // calculate stagnation speed of sound, density, and temperature
//       amrex::Real c0;
//       amrex::Real R = p_int/(s_int[URHO]*s_int[UTEMP]); // probably can get this in input but idk how (gas constant)
//       amrex::Real vel =  sqrt(s_int[UEDEN]/(0.5 * s_int[URHO])); // find the overall velocity
//       amrex::Real T0 = s_int[UTEMP]*(1 + (vel*vel) / (2*s_int[UTEMP]*((gamma*R)/(gamma-1)))); 
//       amrex::Real rho0 = s_int[URHO]* (pow((1 + ((gamma-1)/2))*pow(vel/sqrt(gamma*R*s_int[UTEMP]),2),1/(gamma-1)));
//       c0 = sqrt(gamma*R*T0);
//       amrex::Real rho, eint;
//       // interpolate value for rho
//       rho = s_int[URHO] + (p_ext - p_int)/c0;
//       s_ext[URHO] = rho;
//       // set mass fractions equal to inside now we have density
//       for (int n = 0; n < NUM_SPECIES; n++) {
//           s_ext[UFS + n] = rho * massfrac_ext[n];
//       }
//       // s_ext[UTEMP] = p_ext/(rho*R); <- ideal gas approx
//       // find temperature and energy from our known values
//       eos.RYP2T(rho, massfrac_ext, p_ext, s_ext[UTEMP]);
//       eos.RYP2E(rho, massfrac_ext, p_ext, eint);
//       s_ext[UEINT] = rho * eint;

//       // interpolate values for the velocities
//       s_ext[UMX] = rho * (s_int[UMX]/ s_int[URHO] + sgn*(p_int - p_ext)/(rho0*c0));
//       s_ext[UMY] = rho * (s_int[UMY]/ s_int[URHO] + (p_int - p_ext)/(rho0*c0));
//       s_ext[UMZ] = rho * (s_int[UMZ]/ s_int[URHO] + (p_int - p_ext)/(rho0*c0));
//       s_ext[UEDEN] =
//       s_ext[UEINT] +
//           0.5 * rho * ((s_ext[UMX]/rho) * (s_ext[UMX]/rho) + (s_ext[UMY]/rho) * (s_ext[UMY]/rho) + (s_ext[UMZ]/rho) * (s_ext[UMZ]/rho));
      
//       for (int n = 0; n < NUM_SPECIES; n++) {
//         s_ext[UFS + n] = rho * massfrac_ext[n];
//       }
//      }
//    }
  

}

void pc_prob_close();

using ProblemSpecificFunctions = DefaultProblemSpecificFunctions;

#endif