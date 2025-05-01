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
			matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar(msg) }));
	}

public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
		matlab::data::TypedArray <uint64_t> N(std::move(inputs[0])), D(std::move(inputs[1]));
		const size_t n = N[0], d = D[0];
		std::vector <std::tuple <size_t, size_t, size_t> > buffer;
		auto mono = std::make_unique <size_t[]> (d + 1);
		size_t row = 0;
		do {
			for (size_t i = d; mono[i]; ) {
				buffer.emplace_back(row, mono[i], 1);
				while (mono[--i] == std::get<1>(buffer.back()))
					++std::get<2>(buffer.back());
				--std::get<1>(buffer.back());
			}
			++row;
		} while (next_choice(&mono[1], &mono[d + 1], n));
		auto bufferRow = factory.createBuffer <size_t> (buffer.size()), bufferCol = factory.createBuffer <size_t> (buffer.size());
		auto bufferData = factory.createBuffer <double> (buffer.size());
		for (size_t i = 0; i < buffer.size(); ++i)
			bufferRow[i] = std::get<0>(buffer[i]),
			bufferCol[i] = std::get<1>(buffer[i]),
			bufferData[i] = std::get<2>(buffer[i]);
		outputs[0] = factory.createSparseArray({ row, n }, buffer.size(), std::move(bufferData), std::move(bufferRow), std::move(bufferCol));
	}
};
