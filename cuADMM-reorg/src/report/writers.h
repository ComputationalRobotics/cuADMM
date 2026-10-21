/*

    writers.h

    Output writers of a run (plan Section 7.5: histories with 17 significant digits, sigma log, JSON summary, the
    validation history). Phase 2: functions over a SolveResult, called by the CLI; Phase 4 wraps them in observers.

*/

#ifndef CUADMM_REPORT_WRITERS_H
#define CUADMM_REPORT_WRITERS_H

#include <string>

#include "cuadmm/options.h"
#include "cuadmm/problem.h"
#include "cuadmm/result.h"

namespace cuadmm
{
namespace report
{

// A double as a JSON number, "null" when not finite; 17 significant digits.
std::string json_number(double x);
std::string json_string(const std::string &s);

// iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase (17 significant digits)
void write_history_csv(const std::string &path, const SolveResult &result);

// iter,old_sigma,new_sigma,rule,eta_p,eta_d,eta_g,feasratio,prim_win,dual_win
void write_sigma_log_csv(const std::string &path, const SolveResult &result);

// Appends one JSON object (one line) describing the run: problem, options, build, results.
void append_summary_json(const std::string &path, const std::string &problem_path, const ProblemData &problem,
                         const SolverOptions &options, const SolveResult &result, double load_time_s, double cuda_init_time_s);

// The CSV lines of the external validations (empty without validation)
void write_validation_history_csv(const std::string &path, const SolveResult &result);

// Writes X.txt, y.txt, S.txt (17 digits) of a snapshot; returns false if a file could not be written completely.
bool write_snapshot_txt(const std::string &directory, const std::vector<double> &X, const std::vector<double> &y,
                        const std::vector<double> &S);

} // namespace report
} // namespace cuadmm

#endif // CUADMM_REPORT_WRITERS_H
