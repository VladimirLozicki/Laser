__kernel void laserDynamics(__global float* a,__global float* b,__global float* c)
{
    c[5]=a[5]+b[5];
      printf("Hello from process");
}