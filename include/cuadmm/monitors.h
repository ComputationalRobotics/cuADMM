/*

    monitors.h

    Define monitors in cuADMM solving process.

*/

#ifndef CUADMM_MONITORS_H
#define CUADMM_MONITORS_H

#include <deque>
#include <optional>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cstddef>
#include <cmath>

template <class T>
class SlidingDiffQueue {
public:
    // public members (as you asked)
    std::size_t cap;
    std::deque<T> q1;
    std::deque<T> dq;

    template <class Deque, class Compare>
    static bool all_of(const Deque& d, Compare comp, const T& c) {
        return std::all_of(d.begin(), d.end(),
                           [&](const T& x){ return comp(x, c); });
    }

    explicit SlidingDiffQueue(std::size_t capacity) : cap(capacity) {
        if (this->cap == 0) throw std::invalid_argument("Capacity must be >= 1!\n");
    }

    // add new data
    void push(const T& value) {
        if (!this->q1.empty()) {
            this->dq.push_back(value - this->q1.back());
        }
        this->q1.push_back(value);

        // pop out the front element if exceeding capacity
        if (this->q1.size() > this->cap) {
            this->q1.pop_front();
            if (!this->dq.empty()) this->dq.pop_front();
        }
        return;
    }

    // check size/capacity/full/empty
    std::size_t get_size() const { return this->q1.size(); }
    std::size_t get_capacity() const { return this->cap; }        // fixed
    bool if_full() const { return this->q1.size() == this->cap; }
    bool if_empty() const { return this->q1.empty(); }

    // predicates (vacuous-true on empty)
    bool data_all_greater(const T& c) const { return this->all_of(this->q1, std::greater<T>{}, c); }
    bool data_all_smaller(const T& c) const { return this->all_of(this->q1, std::less<T>{},    c); }
    bool diff_all_greater(const T& c) const { return this->all_of(this->dq, std::greater<T>{}, c); }
    bool diff_all_smaller(const T& c) const { return this->all_of(this->dq, std::less<T>{},    c); }

    // reset both queues
    void reset() { this->q1.clear(); this->dq.clear(); }

    // print options (minimal)
    void print_q1(std::ostream& os = std::cout, const char* sep = " ") const {
        bool first = true;
        char buf[64];
        for (const auto& x : this->q1) {
            if (!first) os << sep;
            first = false;
            if (std::is_floating_point<T>::value) {
                std::snprintf(buf, sizeof(buf), "%3.2e", static_cast<double>(x));
                os << buf;
            } else {
                os << x;
            }
        }
        if (first) os << "(empty)";
    }

    void print_dq(std::ostream& os = std::cout, const char* sep = " ") const {
        bool first = true;
        char buf[64];
        for (const auto& x : this->dq) {
            if (!first) os << sep;
            first = false;
            if (std::is_floating_point<T>::value) {
                std::snprintf(buf, sizeof(buf), "%3.2e", static_cast<double>(x));
                os << buf;
            } else {
                os << x;
            }
        }
        if (first) os << "(empty)";
    }
};

// Monitor1: identify the chasing phenomenon
class Monitor1 {
public:
    std::size_t cap;
    SlidingDiffQueue<double> buffer_pobj;
    SlidingDiffQueue<double> buffer_dobj;
    SlidingDiffQueue<double> buffer_obj_diff;
    SlidingDiffQueue<double> buffer_relgap_feas_ratio;

    // some auxilary constants
    std::size_t update_interval;

    explicit Monitor1(std::size_t capacity = 20): cap(capacity), buffer_pobj(capacity), buffer_dobj(capacity),
        buffer_obj_diff(capacity), buffer_relgap_feas_ratio(capacity)
    {
        if (cap == 0) throw std::invalid_argument("Capacity must be >= 1!\n");
        this->update_interval = 25;
    }

    void reset() {
        this->buffer_pobj.reset();
        this->buffer_dobj.reset();
        this->buffer_relgap_feas_ratio.reset();
        this->buffer_obj_diff.reset();
        return;
    }

    void push(double pobj, double dobj, double pinf, double dinf, double relgap) {
        this->buffer_pobj.push(pobj);
        this->buffer_dobj.push(dobj);
        this->buffer_obj_diff.push(pobj - dobj);
        this->buffer_relgap_feas_ratio.push(relgap / (1e-16 + std::max(pinf, dinf)));
        return;
    }

    bool if_empty() const { return this->buffer_pobj.if_empty(); }
    bool if_full() const { return this->buffer_pobj.if_full(); }

    double chase_update_sig(double sig) {
        if (this->if_empty()) return sig;

        bool if_ratio_large = this->buffer_relgap_feas_ratio.data_all_greater(1.2);
        if (!if_ratio_large) return sig;

        bool if_pobj_ascend = this->buffer_pobj.diff_all_greater(0.0);
        bool if_pobj_descend = this->buffer_pobj.diff_all_smaller(0.0);
        bool if_dobj_ascend = this->buffer_dobj.diff_all_greater(0.0);
        bool if_dobj_descend = this->buffer_dobj.diff_all_smaller(0.0);
        bool if_obj_diff_positive = this->buffer_obj_diff.data_all_greater(0.0);
        bool if_obj_diff_negative = this->buffer_obj_diff.data_all_smaller(0.0);

        double sig_help_dual = std::max(1e2, sig);
        double sig_help_primal = std::min(1e-2, sig);

        if (if_pobj_ascend && if_dobj_ascend) {
            if (if_obj_diff_positive) {
                std::cout << "pobj and dobj both ascend, pobj > dobj, help dual!" << std::endl;
                this->reset();
                return sig_help_dual;
            } else if (if_obj_diff_negative) {
                std::cout << "pobj and dobj both ascend, pobj < dobj, help primal!" << std::endl;
                this->reset();
                return sig_help_primal;
            }
        }
        
        if (if_pobj_descend && if_dobj_descend) {
            if (if_obj_diff_positive) {
                std::cout << "pobj and dobj both descend, pobj > dobj, help primal!" << std::endl;
                this->reset();
                return sig_help_primal;
            } else if (if_obj_diff_negative) {
                std::cout << "pobj and dobj both descend, pobj < dobj, help dual!" << std::endl;
                this->reset();
                return sig_help_dual;
            }
        }

        return sig;
    }
};

#endif // CUADMM_MONITORS_H