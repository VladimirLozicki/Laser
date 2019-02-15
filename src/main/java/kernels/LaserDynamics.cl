#pragma OPENCL EXTENSION cl_khr_fp64 : enable   // double

// граничные условия


static double x_0(double x,  double R_0)
{
	return x * R_0;
}

static double x_max(double x, double R_max)
{
	return x * R_max;
}


__kernel void laserDynamics(__global double * U_plus,__global double * U_minus, __global double * D)
{
	  int pid = get_global_id(0);
      double R_max = {{R_right}};
      double R_0 = {{R_left}};
      double cl_dt={{dt}};
      double cl_dx={{dx}};
      const int local_N={{local_n}};
      int save={{save_freq}};
      const int size={{maxGlobalWorkSize}};



         // для правильного разбора массивов
         int start = (pid == 0) ? 1 : 0;
         int finish = local_N + start;
         // вычисления будут проводиться в приватной (быстрой) памяти
          	__private double u_plus[500 + 1]; // на всех +1 для приватных точек; для унификации
          	__private double u_minus[500 + 1];
          	__private double u_tmp_plus[500 + 1];
          	__private double u_tmp_minus[500 + 1];
          	__private double u_half_plus[500 + 1];
          	__private double u_half_minus[500 + 1];

          	// общая память
          	// начальный слой
          	__local double u_left_plus[2]; // не size - 1, т. к. нужны ещё точки для граничных условий
          	__local double u_right_plus[2];
          	__local double u_left_minus[2];
          	__local double u_right_minus[2];

          	__local double ur0;
          	__local double ur1;

          	// worker'ы разбирают массивы
          	for (int i = start; i < finish; i++)
          	{
          		u_plus[i] = U_plus[pid * local_N + i - start];
          		u_minus[i] = U_minus[pid * local_N + i - start];
            }

          	// вычисления

          	double u_n_plus, u_np1_plus, u_n_minus, u_np1_minus;

         	for(int t_i = 0; t_i < save; t_i++)
         	{
         		// складываем левые и правые точки в локальную память
         	 	u_left_plus[pid] = u_plus[start];
         	 	u_left_minus[pid] = u_minus[start];


         		u_right_plus[pid] = u_plus[local_N - 1 + start];

         		if(pid==0){
         		ur0=u_right_plus[pid];
         		}
         		if(pid==1){
                 ur1=u_right_plus[pid];
                }



         		printf("before pid=%d, u_right_plus[%d] = %lf\n",pid,0,u_right_plus[0]);
         		printf("before pid=%d, u_right_plus[%d] = %lf\n",pid,1,u_right_plus[1]);
         		printf("\n");



         	 	u_right_minus[pid] = u_minus[local_N - 1 + start];

         		// виртуальные точки и граничные условия
         	 	if (pid == 0)
         	 	{
         	 		u_plus[0] = x_0(u_right_minus[size - 1], R_0);
         	 		u_minus[0] = x_max(u_right_plus[size - 1], R_max);
         	 	}
         	 	if (pid == (size - 1))
         	 	{
         	 		u_plus[local_N] = x_max(u_left_minus[0], R_max);
         	 		u_minus[local_N] = x_0(u_left_plus[0], R_0);
         	 	}
         	 	//barrier(CLK_LOCAL_MEM_FENCE);
         	 //	barrier(CLK_LOCAL_MEM_FENCE);
         	 	printf("pid=%d, barrier 1\n",pid);

         		printf("pid=%d, u_right_plus[%d] = %lf\n",pid,0,u_right_plus[0]);
         		printf("pid=%d, u_right_plus[%d] = %lf\n",pid,1,u_right_plus[1]);
         		printf("\n");
         		printf("pid=%d, ur0=%f, ur1=%f\n",pid,ur0,ur1);


         	 	// половинный слой
         		for(int i = 0; i < local_N + 1; i++)
         		{
         	 		if(pid != 0 && i == 0)
         	 		{
         	 			u_n_plus = u_right_plus[pid-1];
printf("pid=%d, %d u_n_plus = %lf, u_right_plus[%d] = %lf\n",pid,i, u_n_plus,pid-1, u_right_plus[pid-1]);
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
/*if(pid==1){
printf("%d u_np1_plus = %lf , u_n_plus = %lf\n",i, u_np1_plus, u_n_plus);
         		}*/
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
if(pid==1){
//printf("%d До u_plus 0= %lf   %d\n",i, u_plus[0], start);
//printf("%d u_np1_plus = %lf , u_n_plus = %lf\n",i, u_np1_plus, u_n_plus);
}
         	/*if(pid==1&& i==0){
printf("%d u_tmp_plus 0= %lf\n",pid, u_tmp_plus[0]);
printf("%d u_tmp_plus 1= %lf\n",pid, u_tmp_plus[1]);
printf("%d u_right_plus = %lf\n",pid, u_right_plus[1]);
printf("%d u_np1_plus = %lf\n",pid, u_np1_plus);
printf("%d u_half_plus 0= %lf\n",pid, u_half_plus[0]);
printf("%d u_half_plus 1= %lf\n",pid, u_half_plus[1]);
printf("%d Plus 0= %lf\n",pid, u_plus[0]);
printf("%d Plus 1= %lf\n",pid, u_plus[1]);
printf("%d Plus 499= %lf\n",pid, u_plus[499]);
printf("%d Plus 500= %lf\n",pid, u_plus[500]);
printf("%d ====== %lf\n",pid, cl_dt / cl_dx * u_tmp_plus[i - start]);
printf("%d u_plus[i] ====== %lf\n",pid, u_plus[i]);
}*/


         			/*u_plus[i] -= cl_dt / cl_dx * u_tmp_plus[i - start]
         							- ((u_tmp_plus[i - start] / (u_np1_plus - u_plus[i] * 0.999999999)
         							* u_half_plus[i + 1 - start]) - u_half_plus[i - start]);

         			u_minus[i] -= cl_dt / cl_dx * u_tmp_minus[i - start]
         							- ((u_tmp_minus[i - start] / (u_np1_minus - u_minus[i] * 0.999999999)
         							* u_half_minus[i + 1 - start]) - u_half_minus[i - start]);
         							*/
         			u_plus[i] -=  cl_dt / cl_dx * u_tmp_plus[i - start]-((u_tmp_plus[i - start] / (u_np1_plus - u_plus[i] * 0.999999999)* u_half_plus[i + 1 - start]) - u_half_plus[i - start]);

         			u_minus[i] -= cl_dt / cl_dx * u_tmp_minus[i - start]-((u_tmp_minus[i - start] / (u_np1_minus - u_minus[i] * 0.999999999)* u_half_minus[i + 1 - start]) - u_half_minus[i - start]);
         		}
         	}
         /*	if(pid==1){
printf("%d u_tmp_plus 0= %lf\n",pid, u_tmp_plus[0]);
printf("%d u_np1_plus = %lf\n",pid, u_np1_plus);
printf("%d u_half_plus 0= %lf\n",pid, u_half_plus[0]);
printf("%d u_half_plus 1= %lf\n",pid, u_half_plus[1]);
printf("%d Plus 0= %lf\n",pid, u_plus[0]);
printf("%d Plus 1= %lf\n",pid, u_plus[1]);
printf("%d Plus 499= %lf\n",pid, u_plus[499]);
printf("%d Plus 500= %lf\n",pid, u_plus[500]);
}*/
         	for (int i = start; i < finish; i++)
         	{
         		U_plus[pid * local_N + i - start] = u_plus[i];
         		U_minus[pid * local_N + i - start] = u_minus[i];
         	}


}