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
		matlab::data::TypedArray <uint64_t> NUM_VARS = std::move(inputs[0]), DEGREE = std::move(inputs[1]);
		const size_t num_vars = NUM_VARS[0], deg = DEGREE[0];
		auto iter_i = std::make_unique <size_t[]> (deg), iter_j = std::make_unique <size_t[]> (deg), iter_merge = std::make_unique <size_t[]> (2 * deg);
		const size_t num_mono = binom(num_vars + deg, deg), mat_size = num_mono * num_mono, vec_size = num_mono * (num_mono + 1) / 2;
		auto bufferData = factory.createBuffer <double> (mat_size);
		auto bufferRow = factory.createBuffer <size_t> (mat_size), bufferCol = factory.createBuffer <size_t> (mat_size);
		size_t idx = 0;
		do {
			std::fill(&iter_j[0], &iter_j[deg], 0);
			do {
				std::merge(&iter_i[0], &iter_i[deg], &iter_j[0], &iter_j[deg], &iter_merge[0]);
				bufferRow[idx] = idx;
				bufferCol[idx] = get_idx(&iter_merge[0], &iter_merge[2 * deg], num_vars);
				bufferData[idx] = 1;
			} while (++idx, next_choice(&iter_j[0], &iter_j[deg], num_vars));
		} while (next_choice(&iter_i[0], &iter_i[deg], num_vars));
		const size_t COL = binom(num_vars + 2 * deg, 2 * deg);
		outputs[0] = factory.createSparseArray({ mat_size, COL }, mat_size, std::move(bufferData), std::move(bufferRow), std::move(bufferCol));
		bufferData = factory.createBuffer <double> (vec_size);
		bufferRow = factory.createBuffer <size_t> (vec_size), bufferCol = factory.createBuffer <size_t> (vec_size);
		size_t j = 0; idx = 0;
		do {
			std::copy(&iter_i[0], &iter_i[deg], &iter_j[0]);
			std::merge(&iter_i[0], &iter_i[deg], &iter_j[0], &iter_j[deg], &iter_merge[0]);
			size_t i = j;
			bufferRow[idx] = i * (i + 1) / 2 + j;
			bufferCol[idx] = get_idx(&iter_merge[0], &iter_merge[2 * deg], num_vars);
			bufferData[idx] = 1;
			while (++i, ++idx, next_choice(&iter_j[0], &iter_j[deg], num_vars)) {
				std::merge(&iter_i[0], &iter_i[deg], &iter_j[0], &iter_j[deg], &iter_merge[0]);
				bufferRow[idx] = i * (i + 1) / 2 + j;
				bufferCol[idx] = get_idx(&iter_merge[0], &iter_merge[2 * deg], num_vars);
				bufferData[idx] = std::sqrt(2);
			}
		} while (++j, next_choice(&iter_i[0], &iter_i[deg], num_vars));
		outputs[1] = factory.createSparseArray({ vec_size, COL }, vec_size, std::move(bufferData), std::move(bufferRow), std::move(bufferCol));
		outputs[2] = factory.createScalar <double> (num_mono);
	}
};
