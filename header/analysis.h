/*
Description
This function analyzes the structure of the sparse matrix received in
the parameter gpu_L and returns the returns the result in the parameter info,
this result must be passed to the solver's multirow or order.

Pre:
Memory must be allocated for the first parameter info of type dfr_analysis_info_t.
gpu_L of type sp_mat_t must be initiated and allocated on a device with the values that describe the matrix to be analyzed.

Post:
the result of the analysis is returned on the parameter info,
user is responsible of freeing up memory allocated
*/

void multirow_analysis_base_GPU(dfr_analysis_info_t** info, sp_mat_t* gpu_L);
