#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

constexpr size_t binom(size_t n, size_t m) {
	size_t ret = 1;
	for (size_t j = 1; j <= m; ++j)
		ret = ret * (n - j + 1) / j;
	return ret;
}

constexpr size_t psd_size(size_t num_vars, size_t deg) {
	return binom(num_vars + deg, deg);
}

template <typename InputIt, typename T>
constexpr size_t get_idx(InputIt first, InputIt last, T num_vars) {
	size_t d = std::distance(first, last);
	T minval = 0, ret = binom(num_vars - minval + d, d);
	while (d--) {
		ret -= binom(num_vars - *first + d, d + 1);
		++first;
	}
	return --ret;
}

template <typename InputIt, typename T>
constexpr std::pair <size_t, size_t> get_previous_idx(InputIt first1, InputIt last1, InputIt first2, InputIt last2, T num_vars) {
	InputIt it2 = std::lower_bound(first2, last2, *(last1 - 1));
	assert(it2 != first2);
	InputIt it1 = std::upper_bound(first1, last1, *--it2);
	assert(it1 != last1);
	std::swap(*it1, *it2);
	size_t i = get_idx(first1, last1, num_vars), j = get_idx(first2, last2, num_vars);
	std::swap(*it1, *it2);
	return std::make_pair(i, j);
}

#ifdef RANDOM_PIVOT
thread_local class get_random_idx_class {
	std::mt19937 rng{std::random_device{}()};
	std::vector <size_t> pool;
public:
	template <typename InputIt, typename T>
	inline std::pair <size_t, size_t> operator ()(InputIt first1, InputIt last1, InputIt first2, InputIt last2, T num_vars) {
		size_t deg = std::distance(first1, last1) + std::distance(first2, last2);
		pool.resize(deg);
		const std::vector <size_t>::iterator mid = pool.begin() + pool.size() / 2;
		std::vector <size_t>::iterator f[] { pool.begin(), mid };
		size_t d[] { pool.size() / 2, (pool.size() + 1) / 2 };
		while (first1 != last1 && first2 != last2) {
			const bool idx = rng() / (rng.max() - rng.min() + 1.) * (d[0] + d[1]) >= d[0];
			--d[idx];
			if (*first1 <= *first2)
				*(f[idx]++) = *(first1++);
			else
				*(f[idx]++) = *(first2++);
		}
		while (first1 != last1) {
			const bool idx = rng() / (rng.max() - rng.min() + 1.) * (d[0] + d[1]) >= d[0];
			--d[idx];
			*(f[idx]++) = *(first1++);
		}
		while (first2 != last2) {
			const bool idx = rng() / (rng.max() - rng.min() + 1.) * (d[0] + d[1]) >= d[0];
			--d[idx];
			*(f[idx]++) = *(first2++);
		}
		assert(d[0] == 0 && d[1] == 0);
		return std::make_pair(get_idx(pool.begin(), mid, num_vars), get_idx(mid, pool.end(), num_vars));
	}

	template <typename InputIt, typename T>
	inline std::pair <size_t, size_t> operator()(InputIt first, InputIt last, T num_vars) {
		size_t deg = std::distance(first, last);
		pool.resize(deg);
		const std::vector <size_t>::iterator mid = pool.begin() + pool.size() / 2;
		std::vector <size_t>::iterator f[] { pool.begin(), mid };
		size_t d[] { pool.size() / 2, (pool.size() + 1) / 2 };
		while (first != last) {
			const bool idx = rng() / (rng.max() - rng.min() + 1.) * (d[0] + d[1]) >= d[0];
			--d[idx];
			*(f[idx]++) = *(first++);
		}
		assert(d[0] == 0 && d[1] == 0);
		return std::make_pair(get_idx(pool.begin(), mid, num_vars), get_idx(mid, pool.end(), num_vars));
	}
} get_pivot_idx;
#else

template <typename InputIt, typename T>
constexpr std::pair <size_t, size_t> get_pivot_idx(InputIt first, InputIt last, T num_vars) {
	InputIt mid = first + (last - first) / 2;
	return std::make_pair(get_idx(first, mid, num_vars), get_idx(mid, last, num_vars));
}

template <typename InputIt, typename T>
std::pair <size_t, size_t> get_pivot_idx(InputIt first1, InputIt last1, InputIt first2, InputIt last2, T num_vars) {
	thread_local std::vector <typename std::remove_reference<decltype(*first1)>::type> cache;
	cache.resize(std::distance(first1, last1) + std::distance(first2, last2));
	std::merge(first1, last1, first2, last2, cache.begin());
	const auto mid = cache.begin() + cache.size() / 2;
	return std::make_pair(get_idx(cache.begin(), mid, num_vars), get_idx(mid, cache.end(), num_vars));
}
#endif

template <typename InputIt, typename T>
constexpr bool next_choice(InputIt first, InputIt last, T num_vars) {
	for (InputIt iter = last; iter-- != first; )
		if (*iter != num_vars) {
			std::fill(iter, last, *iter + 1);
			return true;
		}
	std::fill(first, last, 0);
	return false;
}
