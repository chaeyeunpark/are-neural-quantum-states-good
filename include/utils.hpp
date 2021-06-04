#pragma once
#include <tbb/tbb.h>

inline int get_num_threads()
{
	char* tbb_num_threads = std::getenv("TBB_NUM_THREADS");
	int num_threads = 0;
	if((tbb_num_threads != nullptr) && (tbb_num_threads[0] != '\0'))
	{
		num_threads = atoi(tbb_num_threads);
	}
	if(num_threads <= 0)
		num_threads = tbb::this_task_arena::max_concurrency();
	return num_threads;
}

struct count_pos_neg
{
	uint32_t n_pos;
	uint32_t n_neg;

	count_pos_neg() : n_pos(0u), n_neg(0u) {}
	count_pos_neg(count_pos_neg& s, tbb::split) 
	{
		n_pos = n_neg = 0u;
	}

	void operator()(tbb::blocked_range<std::vector<double>::const_iterator>& r)
	{
		uint32_t n_pos_l = n_pos;
		uint32_t n_neg_l = n_neg;

		for(double a: r)
		{
			if (a > 1e-8)
				++ n_pos_l;
			if (a < -1e-8)
				++ n_neg_l;
		}
		n_pos = n_pos_l;
		n_neg = n_neg_l;
	}

	void join(count_pos_neg& rhs)
	{
		n_pos += rhs.n_pos;
		n_neg += rhs.n_neg;
	}
};
