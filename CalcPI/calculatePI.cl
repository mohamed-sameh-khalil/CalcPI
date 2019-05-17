__kernel void CalculatePiShared(__global float * c, ulong iNumIntervals) {
    __local float LocalPiValues[256]; // work - group size = 256

    // work - item global index
    int iGID = get_global_id(0);
    // work - item local index
    int iLID = get_local_id(0);
    // work - group index
    int iWGID = get_group_id(0);
    // how many work - items are in WG?
    int iWGS = get_local_size(0);

    float x = 0.0;
    float y = 0.0;
    float pi = 0.0;

    while (iGID < iNumIntervals) {
        x = (float)(1.0f / (float) iNumIntervals) * ((float) iGID - 0.5f);
        y = (float) sqrt(1.0f - x * x);
        pi += 4.0f * (float)(y / (float) iNumIntervals);
        iGID += get_global_size(0);
    }

    //store the product
    LocalPiValues[iLID] = pi;
    // wait for all threads in WG:
    barrier(CLK_LOCAL_MEM_FENCE);

    // Summation reduction:
    int i = iWGS / 2;
    while (i != 0) {
        if (iLID < i) {
            LocalPiValues[iLID] += LocalPiValues[iLID + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        i = i / 2;
    }

    // store partial dot product into global memory:
    if (iLID == 0) {
        c[iWGID] = LocalPiValues[0];
    }
}