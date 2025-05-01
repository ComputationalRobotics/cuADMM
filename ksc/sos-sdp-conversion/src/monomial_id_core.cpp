#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "mex.hpp"
#include "mexAdapter.hpp"
#pragma GCC diagnostic pop
#include <cstdint>

#include <vector>
#include <memory>
#include <algorithm>

#include "indices.hpp"

class MexFunction : public matlab::mex::Function {
	std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
	matlab::data::ArrayFactory factory{};

#ifdef assert
#define ORIGINAL_ASSERT_67E38C24 assert
#undef assert
#endif
#ifdef NDEBUG
#define assert(EX) (void)0
#else
#define assert(EX) ((EX) ? (void)0 : mexassert(#EX, __FILE__, __LINE__))
#endif
	void mexassert(const char* expr, const char* file, int line) {
		matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar(file + (": Line " + std::to_string(line)) + ":\n\tAssertion " + expr + " failed!\n") }));
	}
	void mexerror(bool val, auto msg) {
		if (!val)
			mexerror(msg);
	}
	void mexerror(auto msg) {
		matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar(msg) }));
	}

public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
		matlab::data::StructArray V = std::move(inputs[0]), M = std::move(inputs[1]);
		matlab::data::Struct v = std::move(V[0]), m = std::move(M[0]);
		matlab::data::TypedArray <double> dim_v(std::move(v["dim"])), sub_v(std::move(v["sub"])), var_v(std::move(v["var"])), var_m(std::move(m["var"])), pow_m(std::move(m["pow"])), coef_m(std::move(m["coeff"]));
		const size_t num_vars = dim_v[0];
		mexerror(num_vars == var_v.getNumberOfElements(), "variable set is not free: " + std::to_string(var_v.getNumberOfElements()) + " variables detected in " + std::to_string(num_vars) + " slots");
		std::unordered_map <double, size_t> id_index_map; id_index_map.reserve(num_vars);
		for (size_t i = 0; i < num_vars; ++i) {
			mexerror(id_index_map.find(var_v[i]) == id_index_map.end(), "variable set is not free: ID " + std::to_string(var_v[i]) + " duplicated.");
			id_index_map[var_v[i]] = sub_v[i][0];
		}
		for (size_t i = 0; i < pow_m.getDimensions()[0]; ++i)
			if (coef_m[i] != 0) {
				std::vector <size_t> mono;
				for (size_t j = 0; j < pow_m.getDimensions()[1]; ++j)
					if (size_t a = pow_m[i][j]; a != 0) {
						if (auto it = id_index_map.find(var_m[i][j]); it == id_index_map.end())
							mexerror("monomial contains variables outside variable set.");
						else
							mono.resize(mono.size() + a, it->second);
					}
				std::sort(mono.begin(), mono.end());
				outputs[0] = factory.createScalar(get_idx(mono.begin(), mono.end(), num_vars) + 1);
				return;
			}
		mexerror("monomial is 0");
	}
};
