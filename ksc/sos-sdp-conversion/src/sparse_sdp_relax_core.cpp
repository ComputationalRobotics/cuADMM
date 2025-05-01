#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "mex.hpp"
#include "mexAdapter.hpp"
#pragma GCC diagnostic pop
#include <iostream>
#include <iomanip>

#include <cstdint>
#include <limits>

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <algorithm>
#include <ctime>

#include "indices.hpp"

class MexFunction : public matlab::mex::Function {
#ifdef assert
#define ORIGINAL_ASSERT_67E38C24 assert
#undef assert
#endif
#ifdef NDEBUG
#define assert(EX) (void)0
#else
#define assert(EX) ((EX) ? (void)0 : mexassert(#EX, __FILE__, __LINE__))
#endif
	std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
	matlab::data::ArrayFactory factory{};

	void mexassert(const char* expr, const char* file, int line) {
		matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar(file + (": Line " + std::to_string(line)) + ":\n\tAssertion " + expr + " failed!\n") }));
	}
	void mexerror(bool val, auto msg) {
		if (!val)
			matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar(msg) }));
	}
	std::unique_ptr <std::tuple <size_t, size_t, size_t, size_t>[]> moment_constraint_generator(size_t num_vars, size_t deg) {
		thread_local std::vector <size_t> choice_i, choice_j;
		choice_i.clear(); choice_i.resize(deg); choice_j.resize(deg);
		const size_t size_matrix = binom(num_vars + deg, deg), expected_size = size_matrix * (size_matrix + 1) / 2 - binom(num_vars + deg * 2, deg * 2);
		auto moment_constraints = std::make_unique <std::tuple <size_t, size_t, size_t, size_t>[]> (expected_size);
		if (deg == 0) return moment_constraints;
		size_t idx = 0;
		for (size_t i = 0; i < size_matrix; ++i) {
			std::copy(&choice_i[0], &choice_i[deg], &choice_j[0]);
			for (size_t j = i; j < size_matrix; ++j) {
				if (choice_i[deg - 1] <= choice_j[0]) {
					assert((j == i && choice_i[0] == choice_i[deg - 1]) || size_matrix - j == binom(num_vars - choice_i[deg - 1] + deg, deg));
					break;
				}
				auto [pivot_i, pivot_j] = get_previous_idx(&choice_i[0], &choice_i[deg], &choice_j[0], &choice_j[deg], num_vars);
				mexerror(i > pivot_i, "i = " + std::to_string(i) + ", pivot_i = " + std::to_string(pivot_i));
				mexerror(j < pivot_j, "i = " + std::to_string(j) + ", pivot_i = " + std::to_string(pivot_j));
				moment_constraints[idx++] = std::make_tuple(i, j, pivot_i, pivot_j);
				assert(idx != expected_size);
				next_choice(&choice_j[0], &choice_j[deg], num_vars);
			}
			next_choice(&choice_i[0], &choice_i[deg], num_vars);
		}
		assert(idx == expected_size);
		return moment_constraints;
	}

	std::unique_ptr <std::pair <size_t, size_t>[]> pivot_generator(size_t num_vars, size_t deg) {
		thread_local std::vector <size_t> choice;
		choice.clear(); choice.resize(2 * deg);
		const size_t expected_size = binom(num_vars + 2 * deg, 2 * deg);
		auto pivots = std::make_unique <std::pair <size_t, size_t>[]> (expected_size);
		size_t i = 0;
		do pivots[i++] = get_pivot_idx(choice.begin(), choice.end(), num_vars); while (next_choice(choice.begin(), choice.end(), num_vars));
		return pivots;
	}
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
		using namespace std::literals;
		std::cout << "\n===================== Sparse SDP Relaxation =====================" << std::endl;
		matlab::data::StructArray I(std::move(inputs[0]));
		matlab::data::StructArray F(std::move(inputs[1]));
		matlab::data::Struct f(std::move(F[0]));
		matlab::data::StructArray G(std::move(inputs[2]));
		matlab::data::StructArray H(std::move(inputs[3]));
		matlab::data::TypedArray <uint64_t> KAPPA(std::move(inputs[4]));
		matlab::data::TypedArray <uint64_t> RIP(std::move(inputs[5]));
		matlab::data::StructArray Reg(std::move(inputs[6]));
		uint64_t kappa = KAPPA[0];
		matlab::data::TypedArray <double> dim_f = std::move(f["dim"]);
		auto num_cliques = I.getNumberOfElements();
		auto id_index_map = std::make_unique <std::unordered_map <double, size_t>[]> (num_cliques);
		auto num_vars = std::make_unique <size_t[]> (num_cliques), num_ineq = std::make_unique <size_t[]> (num_cliques), num_eq = std::make_unique <size_t[]> (num_cliques);
		auto psd_info = std::make_unique <std::unique_ptr <std::unique_ptr <std::tuple <size_t, size_t, size_t, size_t>[]>[]>[]> (num_cliques);
		auto ineq_info = std::make_unique <std::unique_ptr <std::vector <std::tuple <double, size_t, size_t> >[]>[]> (num_cliques),
			   eq_info = std::make_unique <std::unique_ptr <std::vector <std::tuple <double, size_t, size_t> >[]>[]> (num_cliques);
		auto deg_ineq = std::make_unique <std::unique_ptr <size_t[]>[]> (num_cliques),
			   deg_eq = std::make_unique <std::unique_ptr <size_t[]>[]> (num_cliques);
		auto pivot_ineq = std::make_unique <std::unique_ptr <std::unique_ptr <std::pair <size_t, size_t>[]>[]>[]> (num_cliques);
		assert(G.getNumberOfElements() == num_cliques);

		std::cout << std::left << std::setw(60) << "Preprocessing..."; std::cout.flush();
		for (size_t k = 0; k < num_cliques; ++k) {
			matlab::data::Struct vars{ std::move(I[k]) };
			matlab::data::TypedArray <double> dim{ std::move(vars["dim"]) }, sub{ std::move(vars["sub"]) }, var{ std::move(vars["var"]) }, pow{ std::move(vars["pow"]) }, coeff{ std::move(vars["coeff"]) };
			num_vars[k] = dim[0];
			assert(dim[1] == 1);
			id_index_map[k].reserve(num_vars[k]);
			for (size_t i = 0; i < num_vars[k]; ++i) {
				assert(pow[i] == 1);
				assert(coeff[i] == 1);
				id_index_map[k].emplace(var[i], sub[i][0]);
			}
		}
		std::cout << "Done.\n"; std::cout.flush();

		std::cout << std::left << std::setw(60) << "Generating Inequality Constraints..."; std::cout.flush();
// #pragma omp parallel for
		for (size_t k = 0; k < G.getNumberOfElements(); ++k) {
			matlab::data::Struct ineq{ std::move(G[k]) };
			matlab::data::TypedArray <double> dim = std::move(ineq["dim"]), sub = std::move(ineq["sub"]), var = std::move(ineq["var"]), pow = std::move(ineq["pow"]), coeff = std::move(ineq["coeff"]);
			assert(dim.getNumberOfElements() == 2);
			num_ineq[k] = dim[0];
			psd_info[k].reset(new std::unique_ptr <std::tuple <size_t, size_t, size_t, size_t>[]> [num_ineq[k] + 1]);
			assert(dim[1] == 1);
			assert(sub.getDimensions().size() == 2);
			assert(var.getDimensions().size() == 2);
			assert(pow.getDimensions().size() == 2);
			assert(coeff.getDimensions().size() == 2);
			const size_t E = var.getDimensions()[0], v = var.getDimensions()[1];
			assert(sub.getDimensions()[0] == E);
			assert(sub.getDimensions()[1] == 2);
			assert(pow.getDimensions()[0] == E);
			assert(pow.getDimensions()[1] == v);
			assert(coeff.getDimensions()[0] == E);
			assert(coeff.getDimensions()[1] == 1);
			deg_ineq[k] = std::make_unique <size_t[]> (num_ineq[k] + 1);
			if (E) {
				ineq_info[k].reset(new std::vector <std::tuple <double, size_t, size_t> > [num_ineq[k]]());
				pivot_ineq[k].reset(new std::unique_ptr <std::pair <size_t, size_t>[]> [num_ineq[k]]);
				for (size_t i = 0; i < E; ++i) {
					const size_t polyid = sub[i][0];
					size_t d = 0;
					for (size_t j = 0; j < v; ++j)
						d += pow[i][j];
					deg_ineq[k][polyid] = std::max(deg_ineq[k][polyid], d);
				}
				for (size_t _ = 1; _ <= num_ineq[k]; ++_) {
					deg_ineq[k][_] = (deg_ineq[k][_] + 1) & ~1;
					assert(deg_ineq[k][_] <= 2 * kappa);
				}
				thread_local auto choice_raw = std::make_unique <size_t[]> (4 * kappa);
				for (size_t i = 0; i < E; ++i) {
					std::fill(&choice_raw[0], &choice_raw[2 * kappa], 0);
					const size_t polyid = sub[i][0];
					const size_t d = deg_ineq[k][polyid];
					size_t idx = d;
					for (size_t j = 0; j < v; ++j)
						for (size_t _ = 0; _ < pow[i][j]; ++_) {
							mexerror(id_index_map[k].find(var[i][j]) != id_index_map[k].end(), "In clique " + std::to_string(k + 1) + ", inequality " + std::to_string(polyid) + " contains variables outside the bound of predefined variable set");
							assert(idx > 0);
							choice_raw[--idx] = id_index_map[k][var[i][j]];
						}
					std::sort(&choice_raw[idx], &choice_raw[d]);
					do {
						const auto [pivot_i, pivot_j] = get_pivot_idx(&choice_raw[0], &choice_raw[d], &choice_raw[d], &choice_raw[2 * kappa], num_vars[k]);
						ineq_info[k][polyid - 1].emplace_back(coeff[i], pivot_i, pivot_j);
					} while (next_choice(&choice_raw[d], &choice_raw[2 * kappa], num_vars[k]));
				}
			}
		}
		std::cout << "Done.\n";

		std::cout << std::left << std::setw(60) << "Generating Moment Constraints..."; std::cout.flush();
// #pragma omp parallel for
		for (size_t k = 0; k < num_cliques; ++k)
			for (size_t _ = 0; _ <= num_ineq[k]; ++_) {
				mexerror(deg_ineq[k][_] <= 2 * kappa, "Degree of inequality " + std::to_string(_ + 1) + " in clique " + std::to_string(k + 1) + " has degree higher than problem.relaxation_order");
				psd_info[k][_] = moment_constraint_generator(num_vars[k], kappa - deg_ineq[k][_] / 2);
				if (_)
					pivot_ineq[k][_ - 1] = pivot_generator(num_vars[k], kappa - deg_ineq[k][_] / 2);
			}
		std::cout << "Done.\n";

		std::cout << std::left << std::setw(60) << "Generating Consensus Constraints..."; std::cout.flush();
		auto consensus_info = std::make_unique <std::vector <std::tuple <size_t, size_t, size_t, size_t> >[]> (num_cliques);
// #pragma omp parallel for
		for (size_t k = 1; k < num_cliques; ++k) {
			// s: Running In Process Ik ∩ (I1 ∪ I2 ∪ ... ∪ I(k - 1)) ⊆ Is
			const size_t s = RIP[k] - 1;
			mexerror(s < k, "Clique " + std::to_string(k + 1) + " has a predecessor of clique " + std::to_string(s + 1) + " which violates running in process property. Please rearrange the cliques.");
			thread_local std::vector <std::pair <size_t, size_t> > intersect;
			thread_local std::unique_ptr <size_t[]> choice(new size_t[2 * kappa]), choice_k(new size_t[2 * kappa]), choice_s(new size_t[2 * kappa]);
			intersect.clear();
			for (auto [key, idx_k] : id_index_map[k])
				if (auto it = id_index_map[s].find(key); it != id_index_map[s].end())
					intersect.emplace_back(idx_k, it->second);
			std::fill(&choice[0], &choice[2 * kappa], 0);
			do {
				for (size_t i = 0; i < 2 * kappa; ++i)
					if (choice[i] == 0)
						choice_k[i] = choice_s[i] = 0;
					else
						std::tie(choice_k[i], choice_s[i]) = intersect[choice[i] - 1];
				std::sort(&choice_k[0], &choice_k[2 * kappa]);
				std::sort(&choice_s[0], &choice_s[2 * kappa]);
				const auto [k_i, k_j] = get_pivot_idx(&choice_k[0], &choice_k[2 * kappa], num_vars[k]);
				const auto [s_i, s_j] = get_pivot_idx(&choice_s[0], &choice_s[2 * kappa], num_vars[s]);
				consensus_info[k].emplace_back(k_i, k_j, s_i, s_j);
			} while (next_choice(&choice[0], &choice[2 * kappa], intersect.size()));
		}
		std::cout << "Done." << std::endl;

		std::cout << std::left << std::setw(60) << "Generating Equality Constraints..."; std::cout.flush();
		//assert(H.getNumberOfElements() == num_cliques);
// #pragma omp parallel for
		for (size_t k = 0; k < H.getNumberOfElements(); ++k) {
			matlab::data::Struct vars{ std::move(H[k]) };
			matlab::data::TypedArray <double> dim{ std::move(vars["dim"]) }, sub{ std::move(vars["sub"]) }, var{ std::move(vars["var"]) }, pow{ std::move(vars["pow"]) }, coeff{ std::move(vars["coeff"]) };
			assert(dim.getNumberOfElements() == 2);
			num_eq[k] = dim[0];
			eq_info[k].reset(new std::vector <std::tuple <double, size_t, size_t> > [num_eq[k]]());
			assert(dim[1] == 1);
			assert(sub.getDimensions().size() == 2);
			assert(var.getDimensions().size() == 2);
			assert(pow.getDimensions().size() == 2);
			assert(coeff.getDimensions().size() == 2);
			const size_t E = var.getDimensions()[0], v = var.getDimensions()[1];
			assert(sub.getDimensions()[0] == E);
			assert(sub.getDimensions()[1] == 2);
			assert(pow.getDimensions()[0] == E);
			assert(pow.getDimensions()[1] == v);
			assert(coeff.getDimensions()[0] == E);
			assert(coeff.getDimensions()[1] == 1);
			if (E != 0) {
				deg_eq[k] = std::make_unique <size_t[]> (num_eq[k]);
				for (size_t i = 0; i < E; ++i) {
					const size_t polyid = sub[i][0] - 1;
					size_t d = 0;
					for (size_t j = 0; j < v; ++j)
						d += pow[i][j];
					mexerror(d <= 2 * kappa, "Degree of equality " + std::to_string(polyid + 1) + " in clique " + std::to_string(k + 1) + " has degree higher than problem.relaxation_order");
					deg_eq[k][polyid] = std::max(deg_eq[k][polyid], d);
				}
				thread_local auto choice_raw = std::make_unique <size_t[]> (4 * kappa);
				for (size_t i = 0; i < E; ++i) {
					std::fill(&choice_raw[0], &choice_raw[2 * kappa], 0);
					const size_t polyid = sub[i][0] - 1;
					const size_t d = deg_eq[k][polyid];
					size_t idx = d;
					for (size_t j = 0; j < v; ++j)
						for (size_t _ = 0; _ < pow[i][j]; ++_) {
							mexerror(id_index_map[k].find(var[i][j]) != id_index_map[k].end(), "In clique " + std::to_string(k + 1) + ", equality " + std::to_string(polyid + 1) + " contains variables outside the bound of predefined variable set");
							assert(idx > 0);
							choice_raw[--idx] = id_index_map[k][var[i][j]];
						}
					std::sort(&choice_raw[idx], &choice_raw[d]);
					do {
						const auto [pivot_i, pivot_j] = get_pivot_idx(&choice_raw[0], &choice_raw[d], &choice_raw[d], &choice_raw[2 * kappa], num_vars[k]);
						eq_info[k][polyid].emplace_back(coeff[i], pivot_i, pivot_j);
					} while (next_choice(&choice_raw[d], &choice_raw[2 * kappa], num_vars[k]));
				}
			}
		}
		std::cout << "Done." << std::endl;
		std::cout << std::left << std::setw(60) << "Generating Objective..."; std::cout.flush();
		matlab::data::TypedArray <double> dim{ std::move(f["dim"]) }, sub{ std::move(f["sub"]) }, var{ std::move(f["var"]) }, pow{ std::move(f["pow"]) }, coeff{ std::move(f["coeff"]) };
		assert(dim.getNumberOfElements() == 2);
		assert(dim[0] <= num_cliques);
		assert(dim[1] == 1);
		assert(sub.getDimensions().size() == 2);
		assert(var.getDimensions().size() == 2);
		assert(pow.getDimensions().size() == 2);
		assert(coeff.getDimensions().size() == 2);
		const size_t E = var.getDimensions()[0], v = var.getDimensions()[1];
		assert(sub.getDimensions()[0] == E);
		assert(sub.getDimensions()[1] == 2);
		assert(pow.getDimensions()[0] == E);
		assert(pow.getDimensions()[1] == v);
		assert(coeff.getDimensions()[0] == E);
		assert(coeff.getDimensions()[1] == 1);
		auto objective_info = std::make_unique <std::vector <std::tuple <double, size_t, size_t> >[]> (num_cliques);
		thread_local auto choice_merge = std::make_unique <size_t[]> (2 * kappa);
		for (size_t i = 0; i < E; ++i) {
			const size_t k = sub[i][0] - 1;
			assert(k < num_cliques);
			size_t idx = 2 * kappa;
			for (size_t j = 0; j < v; ++j)
				for (size_t _ = 0; _ < pow[i][j]; ++_) {
					mexerror(idx > 0, "Degree of objective function is higher than relaxation order");
					mexerror(id_index_map[k].find(var[i][j]) != id_index_map[k].end(), "Objective function in clique " + std::to_string(k + 1) + " contains illegal variables");
					choice_merge[--idx] = id_index_map[k][var[i][j]];
				}
			std::sort(&choice_merge[idx], &choice_merge[2 * kappa]);
			std::fill(&choice_merge[0], &choice_merge[idx], 0);
			const auto [pivot_i, pivot_j] = get_pivot_idx(&choice_merge[0], &choice_merge[2 * kappa], num_vars[k]);
			objective_info[k].emplace_back(coeff[i], pivot_i, pivot_j);
		}
		std::cout << "Done." << std::endl;

		std::cout << std::left << std::setw(60) << "Generating Regularization Constraints..."; std::cout.flush();
		auto reg_info = std::make_unique <std::unique_ptr <std::vector <std::tuple <double, size_t, size_t> >[]>[]> (num_cliques);
		auto reg_val = std::make_unique <std::unique_ptr <size_t[]>[]> (num_cliques);
		auto num_reg = std::make_unique <size_t[]> (num_cliques);
		for (size_t k = 0; k < Reg.getNumberOfElements(); ++k) {
			matlab::data::StructArray var_arr{ std::move(Reg[k]["expression"]) };
			assert(var_arr.getNumberOfElements() == 1);
			matlab::data::Struct vars{ std::move(var_arr[0]) };
			matlab::data::TypedArray <double> val{ std::move(Reg[k]["value"]) };
			matlab::data::TypedArray <double> dim{ std::move(vars["dim"]) }, sub{ std::move(vars["sub"]) }, var{ std::move(vars["var"]) }, pow{ std::move(vars["pow"]) }, coeff{ std::move(vars["coeff"]) };
			assert(dim.getNumberOfElements() == 2);
			num_reg[k] = dim[0];
			reg_info[k].reset(new std::vector <std::tuple <double, size_t, size_t> > [num_reg[k]]());
			mexerror(num_reg[k] == val.getNumberOfElements(), "In clique " + std::to_string(k + 1) + ", number of expressions and values mismatch.");
			assert(dim[1] == 1);
			assert(sub.getDimensions().size() == 2);
			assert(var.getDimensions().size() == 2);
			assert(pow.getDimensions().size() == 2);
			assert(coeff.getDimensions().size() == 2);
			const size_t E = var.getDimensions()[0], v = var.getDimensions()[1];
			assert(sub.getDimensions()[0] == E);
			assert(sub.getDimensions()[1] == 2);
			assert(pow.getDimensions()[0] == E);
			assert(pow.getDimensions()[1] == v);
			assert(coeff.getDimensions()[0] == E);
			assert(coeff.getDimensions()[1] == 1);
			for (size_t i = 0; i < E; ++i) {
				const size_t polyid = sub[i][0] - 1;
				size_t idx = 2 * kappa;
				for (size_t j = 0; j < v; ++j)
					for (size_t _ = 0; _ < pow[i][j]; ++_) {
						mexerror(id_index_map[k].find(var[i][j]) != id_index_map[k].end(), "In clique " + std::to_string(k + 1) + ", regularization " + std::to_string(polyid + 1) + " contains variables outside the bound of predefined variable set");
						mexerror(idx > 0, "Degree of regularization " + std::to_string(polyid + 1) + " is higher than relaxation order");
						choice_merge[--idx] = id_index_map[k][var[i][j]];
					}
				std::fill(&choice_merge[0], &choice_merge[idx], 0);
				std::sort(&choice_merge[idx], &choice_merge[2 * kappa]);
				const auto [pivot_i, pivot_j] = get_pivot_idx(&choice_merge[0], &choice_merge[2 * kappa], num_vars[k]);
				reg_info[k][polyid].emplace_back(coeff[i], pivot_i, pivot_j);
			}
			reg_val[k].reset(new size_t[num_reg[k]]);
			std::copy(val.begin(), val.end(), &reg_val[k][0]);
		}
		std::cout << "Done." << std::endl;
		auto SDP = factory.createStructArray({ 1, 1 }, { "sedumi", "sdpt3", "mosek" });
		auto info = factory.createStructArray({ 1, 1 }, { "At_clique" });
		std::cout << std::left << std::setw(60) << "Post Processing..."; std::cout.flush();
		const size_t num_psd = std::accumulate(&num_ineq[0], &num_ineq[num_cliques], num_cliques);
		auto psd_size = std::make_unique <size_t[]> (num_psd), bias_sqr = std::make_unique <size_t[]> (num_psd), psd = std::make_unique <size_t[]> (num_cliques), bias = std::make_unique <size_t[]> (num_psd);
		size_t constraint_id = 0;
		std::vector <std::tuple <double, size_t, size_t> > A_sedumi;
		auto A_sdpt3 = std::make_unique <std::vector <std::tuple <double, size_t, size_t> >[]> (num_psd);
		for (size_t k = 0, j = 0; k < num_cliques; ++k) {
			psd[k] = j;
			for (size_t i = 0; i <= num_ineq[k]; ++i, ++j)
				psd_size[j] = binom(num_vars[k] + kappa - deg_ineq[k][i] / 2, kappa - deg_ineq[k][i] / 2);
		}
		for (size_t i = 1; i < num_psd; ++i) {
			bias_sqr[i] = bias_sqr[i - 1] + psd_size[i - 1] * psd_size[i - 1];
			bias[i] = bias[i - 1] + psd_size[i - 1];
		}
		const size_t num_sedumi_vars = bias_sqr[num_psd - 1] + psd_size[num_psd - 1] * psd_size[num_psd - 1];
		auto add_constraint = [&](double coeff, size_t row, size_t k, size_t i, size_t j) {
			if (i == j)
				A_sedumi.emplace_back(coeff, row, bias_sqr[k] + i * psd_size[k] + j);
			else {
				A_sedumi.emplace_back(coeff * 0.5, row, bias_sqr[k] + i * psd_size[k] + j);
				A_sedumi.emplace_back(coeff * 0.5, row, bias_sqr[k] + j * psd_size[k] + i);
			}
			if (i < j) std::swap(i, j);
			A_sdpt3[k].emplace_back(coeff * (i == j ? 1 : std::sqrt(2) / 2), row, i * (i + 1) / 2 + j);
			i += bias[k]; j += bias[k];
		};
		size_t num_consensus_constraint = 0, num_equality_constraint = 0, num_inequality_constraint = 0, num_moment_constraint = 0, num_regularization_constraint = 0;
		std::vector <std::pair <double, size_t> > bufferb;
		auto info_At_clique_id = factory.createStructArray({ num_cliques, 1 }, { "moment", "inequality", "equality", "consensus", "regularization" });
		for (size_t k = 0; k < num_cliques; ++k) {
			if (k) {
				const size_t s = RIP[k];
				auto buf = factory.createArray <size_t> ({ consensus_info[k].size(), 1 });
				std::iota(buf.begin(), buf.end(), constraint_id + 1);
				info_At_clique_id[k]["consensus"] = std::move(buf);
				for (auto [i1, j1, i2, j2] : consensus_info[k]) {
					add_constraint(-1, constraint_id, psd[k], i1, j1);
					add_constraint(1, constraint_id++, psd[s - 1], i2, j2);
				}
				num_consensus_constraint += consensus_info[k].size();
			}
			auto buf = factory.createArray <size_t> ({ num_reg[k], 1 });
			std::iota(buf.begin(), buf.end(), constraint_id + 1);
			info_At_clique_id[k]["regularization"] = std::move(buf);
			for (size_t _ = 0; _ < num_reg[k]; ++_) {
				for (auto [c, i, j] : reg_info[k][_])
					add_constraint(c, constraint_id, psd[k], i, j);
				bufferb.emplace_back(reg_val[k][_], constraint_id++);
			}
			num_regularization_constraint += num_reg[k];
			auto ineq_buf = factory.createStructArray({ num_ineq[k], 1 }, { "moment", "consensus" });
			size_t ineq_id = constraint_id;
			for (size_t i_psd = 0; i_psd <= num_ineq[k]; ++i_psd) {
				const size_t num_mom = psd_size[psd[k] + i_psd] * (psd_size[psd[k] + i_psd] + 1) / 2 - binom(num_vars[k] + 2 * kappa - deg_ineq[k][i_psd], 2 * kappa - deg_ineq[k][i_psd]);
				auto buf = factory.createArray <size_t> ({ num_mom, 1 });
				std::iota(buf.begin(), buf.end(), constraint_id + 1);
				for (size_t _ = 0; _ < num_mom; ++_) {
					auto [i1, j1, i2, j2] = psd_info[k][i_psd][_];
					add_constraint(-1, constraint_id, psd[k] + i_psd, i1, j1);
					add_constraint(1, constraint_id++,psd[k] + i_psd, i2, j2);
				}
				if (i_psd == 0) {
					num_moment_constraint += num_mom;
					info_At_clique_id[k]["moment"] = std::move(buf);
					auto eq_id = constraint_id;
					for (size_t ___ = 0; ___ < num_eq[k]; ++___) {
						const size_t num_mono = binom(num_vars[k] + 2 * kappa - deg_eq[k][___], 2 * kappa - deg_eq[k][___]);
						assert(eq_info[k][___].size() % num_mono == 0);
						for (size_t __ = 0; __ < eq_info[k][___].size(); __ += num_mono)
							for (size_t _ = 0; _ < num_mono; ++_) {
								auto [c, i, j] = eq_info[k][___][__ + _];
								add_constraint(c, constraint_id + _, psd[k], i, j);
							}
						constraint_id += num_mono;
					}
					num_equality_constraint += constraint_id - eq_id;
					buf = factory.createArray <size_t> ({ constraint_id - eq_id, 1 });
					std::iota(buf.begin(), buf.end(), eq_id + 1);
					info_At_clique_id[k]["equality"] = std::move(buf);
					ineq_id = constraint_id;
				}
				else {
					ineq_buf[i_psd - 1]["moment"] = std::move(buf);
					const size_t num_mono = binom(num_vars[k] + 2 * kappa - deg_ineq[k][i_psd], 2 * kappa - deg_ineq[k][i_psd]);
					buf = factory.createArray <size_t> ({ num_mono, 1 });
					std::iota(buf.begin(), buf.end(), constraint_id + 1);
					ineq_buf[i_psd - 1]["consensus"] = std::move(buf);
					for (size_t _ = 0; _ < num_mono; ++_) {
						auto [i, j] = pivot_ineq[k][i_psd - 1][_];
						add_constraint(-1, constraint_id + _, psd[k] + i_psd, i, j);
					}
					assert(ineq_info[k][i_psd - 1].size() % num_mono == 0);
					for (size_t __ = 0; __ < ineq_info[k][i_psd - 1].size(); __ += num_mono)
						for (size_t _ = 0; _ < num_mono; ++_) {
							auto [c, i, j] = ineq_info[k][i_psd - 1][__ + _];
							add_constraint(c, constraint_id + _, psd[k], i, j);
						}
					constraint_id += num_mono;
				}
			}
			num_inequality_constraint += constraint_id - ineq_id;
			info_At_clique_id[k]["inequality"] = std::move(ineq_buf);
		}
		info[0]["At_clique"] = std::move(info_At_clique_id);
		std::vector <std::pair <double, size_t> > C_sedumi;
		for (size_t k = 0; k < num_cliques; ++k)
			for (auto [c, i, j] : objective_info[k]) {
				C_sedumi.emplace_back(c * 0.5, psd_size[psd[k]] * i + j + bias_sqr[psd[k]]);
				C_sedumi.emplace_back(c * 0.5, psd_size[psd[k]] * j + i + bias_sqr[psd[k]]);
				i += bias[psd[k]], j += bias[psd[k]];
			}
		auto createBuffer = [&](size_t size) { return std::make_tuple(factory.createBuffer <double> (size), factory.createBuffer <size_t> (size), factory.createBuffer <size_t> (size)); };
		auto [ AData, ARow, ACol ] = createBuffer(A_sedumi.size());
		for (size_t i = 0; i < A_sedumi.size(); ++i) {
			std::tie(AData[i], ARow[i], ACol[i]) = A_sedumi[i];
		}
		std::cout << "Done.\n";

		std::cout << std::left << std::setw(60) << "Storing Data in SeDuMi format..."; std::cout.flush();
		auto t0 = clock();
		auto sedumi = factory.createStructArray({ 1, 1 }, { "At", "b", "c", "K" });
		sedumi[0]["At"] = factory.createSparseArray({ num_sedumi_vars, constraint_id }, A_sedumi.size(), std::move(AData), std::move(ACol), std::move(ARow));
		auto [ bData, bRow, bCol ] = createBuffer(bufferb.size());
		std::fill(&bCol[0], &bCol[bufferb.size()], 0);
		for (size_t i = 0; i < bufferb.size(); ++i)
			std::tie(bData[i], bRow[i]) = bufferb[i];
		auto b = factory.createSparseArray({ constraint_id, 1 }, bufferb.size(), std::move(bData), std::move(bRow), std::move(bCol));
		sedumi[0]["b"] = b;
		auto [ cData, cRow, cCol ] = createBuffer(C_sedumi.size());
		std::fill(&cCol[0], &cCol[C_sedumi.size()], 0);
		for (size_t i = 0; i < C_sedumi.size(); ++i) {
			std::tie(cData[i], cRow[i]) = C_sedumi[i];
		}
		sedumi[0]["c"] = factory.createSparseArray({ num_sedumi_vars, 1 }, C_sedumi.size(), std::move(cData), std::move(cRow), std::move(cCol));
		auto K_sedumi = factory.createStructArray({ 1, 1 }, { "s" });
		K_sedumi[0]["s"] = factory.createArray <size_t*, double> ({ num_psd, 1 }, &psd_size[0], &psd_size[num_psd], matlab::data::InputLayout::COLUMN_MAJOR);
		sedumi[0]["K"] = std::move(K_sedumi);
		SDP[0]["sedumi"] = std::move(sedumi);
		std::cout << "Done.\n";
		std::cout << std::left << std::setw(60) << "Storing Data in SDPT3 format..."; std::cout.flush();
		auto sdpt3 = factory.createStructArray({ 1, 1 }, { "b", "At", "C", "blk" });
		auto At_sdpt3 = factory.createCellArray({ num_psd, 1 }), blk_sdpt3 = factory.createCellArray({ num_psd, 2 }), C_sdpt3 = factory.createCellArray({ num_psd, 1 });
		for (size_t k = 0; k < num_psd; ++k) {
			auto [ AData, ARow, ACol ] = createBuffer(A_sdpt3[k].size());
			for (size_t i = 0; i < A_sdpt3[k].size(); ++i)
				std::tie(AData[i], ARow[i], ACol[i]) = A_sdpt3[k][i];
			At_sdpt3[k] = factory.createSparseArray({ psd_size[k] * (psd_size[k] + 1) / 2, constraint_id }, A_sdpt3[k].size(), std::move(AData), std::move(ACol), std::move(ARow));
			blk_sdpt3[k][0] = factory.createCharArray("s");
			blk_sdpt3[k][1] = factory.createScalar(psd_size[k] * 1.0);
		}
		sdpt3[0]["At"] = std::move(At_sdpt3);
		sdpt3[0]["blk"] = std::move(blk_sdpt3);
		for (size_t k = 0; k < num_cliques; ++k) {
			auto [ CData, CRow, CCol ] = createBuffer(2 * objective_info[k].size());
			for (size_t i = 0; i < objective_info[k].size(); ++i) {
				CData[2 * i] = CData[2 * i + 1] = std::get<0>(objective_info[k][i]) * 0.5;
				CRow[2 * i] = CCol[2 * i + 1] = std::get<1>(objective_info[k][i]);
				CRow[2 * i + 1] = CCol[2 * i] = std::get<2>(objective_info[k][i]);
			}
			C_sdpt3[psd[k]] = factory.createSparseArray({ psd_size[psd[k]], psd_size[psd[k]] }, 2 * objective_info[k].size(), std::move(CData), std::move(CRow), std::move(CCol));
			for (size_t i = 1; i <= num_ineq[k]; ++i)
				C_sdpt3[psd[k] + i] = factory.createSparseArray({ psd_size[psd[k] + i], psd_size[psd[k] + i] }, 0, factory.createBuffer <double> (0), factory.createBuffer <size_t> (0), factory.createBuffer <size_t> (0));
		}
		sdpt3[0]["C"] = std::move(C_sdpt3);
		sdpt3[0]["b"] = b;
		SDP[0]["sdpt3"] = std::move(sdpt3);
		std::cout << "Done.\n";
		outputs[0] = std::move(SDP);
		outputs[1] = std::move(info);
		std::cout << "=================================================================\nSummary:\nTotal Number of Constraints: " << constraint_id << "\n\tNumber of Moment Constraints: " << num_moment_constraint << "\n\tNumber of Inequality Constraints: " << num_inequality_constraint << "\n\tNumber of Equality Constraints: " << num_equality_constraint << "\n\tNumber of Consensus Constraints: " << num_consensus_constraint << "\n\tNumber of Regularization Constraints: " << num_regularization_constraint << "\nNumber of PSD Variable Matrices: " << num_psd << "\n\tNumber of Sedumi Variables: " << num_sedumi_vars << std::endl;
	}
#undef assert
#ifdef ORIGINAL_ASSERT_67E38C24
#define assert ORIGINAL_ASSERT_67E38C24
#undef ORIGINAL_ASSERT_67E38C24
#endif
};
