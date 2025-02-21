#ifndef PROB_PARM_H
#define PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

// using namespace amrex::literals;

struct ProbParmDevice
{
  amrex::Real equiv_ratio = 1.0;
  amrex::Real reac_temp = 298.15;
  amrex::Real reac_pres = 1013250.0; // In CGS unitss

  // inlet conditions
  amrex::Real inj_p0 = 5000000;
  amrex::Real inj_t0 = 300; 
  int inject_fuel = 1;
  amrex::Real num_inlet = 30;
  amrex::Real inj_R = 8.31446261815324e+07/(0.147*31.998 + 0.297*2.016 + 0.556*28.014);
  amrex::Real p_outlet = 1013250.0;
};

struct ProbParmHost
{
  ProbParmHost() = default;
};

#endif