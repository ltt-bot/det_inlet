#include "prob.H"
#include "mechanism.H"
#include "IndexDefines.H"

void
pc_prob_close()
{
}

extern "C" {
void
amrex_probinit(
  const int* /*init*/,
  const int* /*name*/,
  const int* /*namelen*/,
  const amrex::Real* problo,
  const amrex::Real* probhi)
{
  amrex::ParmParse pp("prob");

  // Need to fill all the values in ProbParmDevice
  pp.query("equiv_ratio", PeleC::h_prob_parm_device->equiv_ratio);
  pp.query("reac_temp", PeleC::h_prob_parm_device->reac_temp);
  pp.query("reac_pres", PeleC::h_prob_parm_device->reac_pres);
}
}

void
PeleC::problem_post_timestep()
{
}

void
PeleC::problem_post_init()
{
}

void
PeleC::problem_post_restart()
{
}