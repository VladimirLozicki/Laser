#pragma OPENCL EXTENSION cl_khr_fp64 : enable // поддержка double

// граничные условия
    
    
double x0(double x,  double R0)
{	
	return x * R0;
}   

double xmax(double x, double Rmax)
{ 
	return x * Rmax;
}   


__kernel void laserDynamics(__global double * U_plus,__global double * U_minus)
{	
	    int pid = get_global_id(0);


        const double Rmax={{Rright}};
        const double R0={{Rleft}};


        const int local_N={{local_n}};
        const int size={{maxGlobalWorkSize}};
        double cl_dt={{dt}};
    	double cl_dx={{dx}};
    	int save={{save_freq}};
        // для правильного разбора массивов
        int start = (pid == 0) ? 1 : 0;
 	    int finish = local_N + start;

 	

	// вычисления будут проводиться в приватной (быстрой) памяти 
 	__private double u_plus[local_N + 1]; // на всех +1 для приватных точек; для унификации
 	__private double u_minus[local_N + 1];
 	__private double u_tmp_plus[local_N + 1];
 	__private double u_tmp_minus[local_N + 1];

 	__private double u_half_plus[local_N + 1];
 	__private double u_half_minus[local_N + 1];

 	// общая память 
 	// начальный слой 
 	__local double u_left_plus[size]; // не size - 1, т. к. нужны ещё точки для граничных условий
 	__local double u_right_plus[size];
 	__local double u_left_minus[size];
 	__local double u_right_minus[size];

 	// worker'ы разбирают массивы 
 	for (int i = start; i < finish; i++) 
 	{ 
 		u_plus[i] = U_plus[pid * local_N + i - start];
 		u_minus[i] = U_minus[pid * local_N + i - start];
 	}   
    //////
 	// вычисления
 	//////////////////
 	double u_n_plus, u_np1_plus, u_n_minus, u_np1_minus;  

	for(int t_i = 0; t_i < save; t_i++)
	{   
		// складываем левые и правые точки в локальную память 
	 	u_left_plus[pid] = u_plus[start]; 
	 	u_left_minus[pid] = u_minus[start]; 
	
		u_right_plus[pid] = u_plus[local_N - 1 + start];
	 	u_right_minus[pid] = u_minus[local_N - 1 + start];

		// виртуальные точки и граничные условия
	 	if (pid == 0) 
	 	{ 
	 		u_plus[0] = x0(u_right_minus[size - 1], R0);
	 		u_minus[0] = xmax(u_right_plus[size - 1], Rmax);
	 	} 
	 	if (pid == (size - 1))
	 	{ 
	 		u_plus[local_N] = xmax(u_left_minus[0], Rmax);
	 		u_minus[local_N] = x0(u_left_plus[0], R0);
	 	}

	 	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	 	// половинный слой 
		for(int i = 0; i < local_N + 1; i++)
		{ 
	 		if(pid != 0 && i == 0) 
	 		{ 
	 			u_n_plus = u_right_plus[pid - 1]; 
				u_np1_plus = u_plus[i];
	 			u_n_minus = u_right_minus[pid - 1];
				u_np1_minus = u_minus[i];
	 		} 
	 		if(pid != size - 1 && i == local_N)
	 		{ 
				u_n_plus = u_plus[i - 1 + start];
	 			u_np1_plus = u_left_plus[pid + 1]; 
				u_n_minus = u_minus[i - 1 + start]; 
	 			u_np1_minus = u_left_minus[pid + 1]; 
	 		} 
	 		if((pid == 0 && i == 0) || (pid == size - 1 && i == local_N) || (i > 0 && i < local_N))
	 		{ 
	 			u_n_plus = u_plus[i - 1 + start]; 
	 			u_np1_plus = u_plus[i + start]; 
	 			u_n_minus = u_minus[i - 1 + start]; 
	 			u_np1_minus = u_minus[i + start]; 
	 		}
   
	 		u_half_plus[i] = cl_dt / cl_dx * (1 - cl_dt) * (u_np1_plus - u_n_plus);
	 		u_half_minus[i] = cl_dt / cl_dx * (1 - cl_dt) * (u_np1_minus - u_n_minus);
			
			// разности для полного слоя
			u_tmp_plus[i] = u_np1_plus - u_n_plus;
			u_tmp_minus[i] = u_np1_minus - u_n_minus;
		}

		// полный слой 
		for(int i = start; i < finish; i++) 
		{ 
			if(pid != 0 && i == 0) 
			{ 
				u_n_plus = u_right_plus[pid - 1]; 
				u_np1_plus = u_plus[i + 1];
				u_n_minus = u_right_minus[pid - 1];
				u_np1_minus = u_minus[i + 1];
			} 
			if(pid != size - 1 && i == local_N - 1 + start)
			{ 
				u_n_plus = u_plus[i - 1];
				u_np1_plus = u_left_plus[pid + 1]; 
				u_n_minus = u_minus[i - 1]; 
				u_np1_minus = u_left_minus[pid + 1];
			} 
			if((i > 0 && i < local_N - 1 + start) || (pid == size - 1 && i == local_N - 1))
			{ 
				u_n_plus = u_plus[i - 1]; 
				u_np1_plus = u_plus[i + 1]; 
				u_n_minus = u_minus[i - 1]; 
				u_np1_minus = u_minus[i + 1]; 
			}   

			u_plus[i] -= cl_dt / cl_dx * u_tmp_plus[i - start]
							- ((u_tmp_plus[i - start] / (u_np1_plus - u_plus[i] * 0.999999999)
							* u_half_plus[i + 1 - start]) - u_half_plus[i - start]); 
			u_minus[i] -= cl_dt / cl_dx * u_tmp_minus[i - start]
							- ((u_tmp_minus[i - start] / (u_np1_minus - u_minus[i] * 0.999999999)
							* u_half_minus[i + 1 - start]) - u_half_minus[i - start]); 			
		} 	 
	}

	for (int i = start; i < finish; i++) 	
	{ 	
		U_plus[pid * local_N + i - start] = u_plus[i];
		U_minus[pid * local_N + i - start] = u_minus[i];
	} 	
} 	