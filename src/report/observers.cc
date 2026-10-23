/*

    report/observers.cc

*/

#include "report/observers.h"

#include <cstdio>
#include <iostream>

#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace report
{

void ConsoleObserver::on_start(const admm::KktResiduals &, double)
{
    if (verbosity_ < 1)
        return;
    std::cout << "  it. | p infeas d infeas | primal obj.   dual obj. rel. gap |  time |   sigma " << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
}

void ConsoleObserver::on_check(int done, const admm::KktResiduals &r, double sigma, double time_s, bool stopping)
{
    if (verbosity_ < 1)
        return;
    const int iter = done + 1; // the cadence of main: at the start of iteration `iter`
    if (stopping || (iter <= 200 && (iter % 50) == 1) || (iter > 200 && (iter % 100) == 1))
    {
        printf(" %4d | %3.2e %3.2e | %- 5.4e %- 5.4e %3.2e | %5.1f | %2.1e ", done, r.eta_p, r.eta_d, r.pobj, r.dobj, r.eta_g, time_s, sigma);
        std::cout << std::endl;
    }
}

void HistoryObserver::on_iteration(const IterationEvent &e)
{
    HistoryRow row;
    row.iteration = e.k;
    row.time_s = e.time_s;
    row.pobj = e.r.pobj;
    row.dobj = e.r.dobj;
    row.eta_p = e.r.eta_p;
    row.eta_d = e.r.eta_d;
    row.eta_g = e.r.eta_g;
    row.sigma = e.sigma_next;
    row.tau = e.tau;
    row.sgs = e.sgs;
    rows_.push_back(row);
}

void LegacyInfoObserver::on_iteration(const IterationEvent &e)
{
    SDPSolver &s = engine_;
    s.info_pobj_arr.push_back(e.r.pobj);
    s.info_dobj_arr.push_back(e.r.dobj);
    s.info_errRp_arr.push_back(e.r.eta_p);
    s.info_errRd_arr.push_back(e.r.eta_d);
    s.info_relgap_arr.push_back(e.r.eta_g);
    s.info_sig_arr.push_back(e.sigma_next);
    s.info_tau_arr.push_back(e.tau);
    s.info_bscale_arr.push_back(s.bscale);
    s.info_Cscale_arr.push_back(s.Cscale);
    s.info_time_arr.push_back(e.time_s);
    s.info_iter_num++;
}

} // namespace report
} // namespace cuadmm
