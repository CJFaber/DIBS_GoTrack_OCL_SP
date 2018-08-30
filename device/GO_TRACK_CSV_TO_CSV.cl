// Searches through the CSV inputBuffer to find the indexes of the beginning and
// end of the desired output for that line.

#define MAX_LINE_SIZE 64
//__attribute__((num_simd_work_items(16)))
//__attribute__((reqd_work_group_size(64,1,1)))
__kernel void input_coordinates(
    __global const char* restrict inputBuffer,
    __global const uint* restrict indexBuffer,
    __global char* restrict outputBuffer)
{
    uint lineIndex = indexBuffer[get_global_id(0)];
    uint writeIndex = get_global_id(0) * MAX_LINE_SIZE;
    char tokenDelimiter = ',';
    uint k = 0;
    uint numFound = 0;
    while (k < MAX_LINE_SIZE)
    {
        const char currChar = inputBuffer[lineIndex + k];
        if (currChar == tokenDelimiter)
        {
            numFound++;
        }
        if (numFound < 3)
        {
            outputBuffer[writeIndex + k] = currChar;
        } 
        else if (k < MAX_LINE_SIZE - 1)
        {
            outputBuffer[writeIndex + k] = ' ';
        }
        else {
            outputBuffer[writeIndex + k] = '\n';
        }
        k++;
    }
}

__kernel void input_coordinates{
    __global const char* restrict inputBuffer,
    __global 
