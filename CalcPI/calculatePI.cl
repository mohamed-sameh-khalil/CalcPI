__kernel void CalculatePiShared(__global float * c, ulong iNumIntervals) {
    __local float LocalPiValues[1024]; // work - group size = 256

    // work - item global index
    int glob_index = get_global_id(0);
    // work - item local index
    int local_index = get_local_id(0);
    // work - group index
    int group_index = get_group_id(0);
    // how many work - items are in WG?
    int WGsize = get_local_size(0);

    float x = 0.0;
    float y = 0.0;
    float pi = 0.0;

    while (glob_index < iNumIntervals) {
        x = (float)(1.0f / (float) iNumIntervals) * ((float) glob_index - 0.5f);
        y = (float) sqrt(1.0f - x * x);
        pi += 4.0f * (float)(y / (float) iNumIntervals);
        glob_index += get_global_size(0);
    }

    //store the product
    LocalPiValues[local_index] = pi;
    // wait for all threads in WG:
    barrier(CLK_LOCAL_MEM_FENCE);

    // Summation reduction:
    int i = WGsize / 2;
    while (i != 0) {
        if (local_index < i) {
            LocalPiValues[local_index] += LocalPiValues[local_index + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        i = i / 2;
    }

    // store partial dot product into global memory:
    if (local_index == 0) {
        c[group_index] = LocalPiValues[0];
    }
}