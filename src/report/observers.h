/*

    report/observers.h

    IterationObserver (plan Section 7.5): the bookkeeping of a run moves out of the solver into observers, which
    receive the post-step residuals of each iteration paired with the sigma of the next one (as the history of today)
    and never influence the iteration. Provided: the console table, the history (17 significant digits through the
    writers), the legacy info arrays of the ported engine (for the copied tests).

*/

#ifndef CUADMM_REPORT_OBSERVERS_H
#define CUADMM_REPORT_OBSERVERS_H

#include <memory>
#include <vector>

#include "admm/penalty_policy.h" // KktResiduals
#include "cuadmm/result.h"

class SDPSolver;

namespace cuadmm
{
namespace report
{

struct IterationEvent
{
    int k = 0;                 // the iteration just completed (1-based)
    admm::KktResiduals r;      // its residuals and objectives (original units)
    double sigma_used = 0.0;   // sigma of this iteration
    double sigma_next = 0.0;   // sigma of the next iteration (what the history records, as today)
    double tau = 0.0;          // step length of this iteration
    bool sgs = false;          // the iteration used the sGS step
    double time_s = 0.0;       // seconds since the start of the iterations, after this iteration
};

class IterationObserver
{
public:
    virtual ~IterationObserver() = default;
    virtual void on_start(const admm::KktResiduals &r0, double sigma0) { (void)r0; (void)sigma0; }
    // before the stopping check of iteration k (done = k - 1 completed iterations), with the current residuals
    virtual void on_check(int done, const admm::KktResiduals &r, double sigma, double time_s, bool stopping)
    {
        (void)done; (void)r; (void)sigma; (void)time_s; (void)stopping;
    }
    virtual void on_iteration(const IterationEvent &event) { (void)event; }
};

class ObserverSet
{
public:
    void add(std::shared_ptr<IterationObserver> observer) { observers_.push_back(std::move(observer)); }
    void on_start(const admm::KktResiduals &r0, double sigma0)
    {
        for (auto &o : observers_)
            o->on_start(r0, sigma0);
    }
    void on_check(int done, const admm::KktResiduals &r, double sigma, double time_s, bool stopping)
    {
        for (auto &o : observers_)
            o->on_check(done, r, sigma, time_s, stopping);
    }
    void on_iteration(const IterationEvent &event)
    {
        for (auto &o : observers_)
            o->on_iteration(event);
    }
    bool empty() const { return observers_.empty(); }

private:
    std::vector<std::shared_ptr<IterationObserver>> observers_;
};

/// @brief The console table of today (" it. | p infeas d infeas | ..."): a line at the iterations 0, 50, 100, 150,
/// 200, 300, 400, ... and at the stop.
class ConsoleObserver final : public IterationObserver
{
public:
    explicit ConsoleObserver(int verbosity) : verbosity_(verbosity) {}
    void on_start(const admm::KktResiduals &r0, double sigma0) override;
    void on_check(int done, const admm::KktResiduals &r, double sigma, double time_s, bool stopping) override;

private:
    int verbosity_;
};

/// @brief The per-iteration history as SolveResult rows.
class HistoryObserver final : public IterationObserver
{
public:
    void on_iteration(const IterationEvent &event) override;
    const std::vector<HistoryRow> &rows() const { return rows_; }
    std::vector<HistoryRow> take() { return std::move(rows_); }

private:
    std::vector<HistoryRow> rows_;
};

/// @brief Appends to the info_* arrays of the ported engine (the copied tests of main read them).
class LegacyInfoObserver final : public IterationObserver
{
public:
    explicit LegacyInfoObserver(SDPSolver &engine) : engine_(engine) {}
    void on_iteration(const IterationEvent &event) override;

private:
    SDPSolver &engine_;
};

} // namespace report
} // namespace cuadmm

#endif // CUADMM_REPORT_OBSERVERS_H
