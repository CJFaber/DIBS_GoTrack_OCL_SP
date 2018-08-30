/**
 *
 * Author: Clayton Faber
 *
 * Transforms the GoTrack csv file format into a generalized GPS trajectory
 * csv file format of "ID,LATTITUDE,LONGITUDE". 
 * 
 * This Program is built for the seperate parts kernel (SP). Main kernel
 * actions are split into seperate kernels using pipes to pass the data
 * to the next kernel
 *
 * Find New lines -> Find Comma Delimiters -> Perform transform
 *
 * Coded using mem_bandwidth example as a template
 *
 */

#include <stdio.h>
#include <fstream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>

#include "CL/opencl.h"
// /tools/x86_64/intelharpv2/opencl/altera/16.0/hld/host/include/CL
//#include "ACLHostUtils.h"
#include "AOCLUtils/aocl_utils.h"


#define FIRST_ARG 1
#define NUM_EXPECTED_ARGS 2
#define LINE_OFFSET 1
#define MAX_LINE_SIZE 64
#define MAX_STR_SIZE 100
#define NUM_QUEUES 3

using namespace aocl_utils;

////////////////////////////////////////////////////////////////////
//	Function Declartions
////////////////////////////////////////////////////////////////////

// Loads input file at the provided filepath into the provided buffer
// Returns the size of the input file
size_t load_csv_to_buffer(char* filepath, char** buffer)
{
    //printf(filepath);
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        printf("Cannot open file \n");
	return 0;
    }

    // find the length of the file
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    *buffer = (char *)calloc(fileSize + 1, sizeof(char));

    size_t ret = fread(*buffer, fileSize, 1, file);
    //printf("Read %zu bytes\n", fileSize * ret);
    fclose(file);
    //printf("File successfully read \n");

    return fileSize;
}



// Writes file of provided file size with contents of provided buffer
void write_csv_file(char* filepath, char *buffer, size_t fileSize)
{
    std::ofstream outFile; 
	outFile.open(filepath);
	for(int i = 0; i < fileSize; ++i){
		//Looking for special char
		if(buffer[i] == '0x1a'){
			break;
		}
		else{
			outFile << buffer[i];
		}
	}
	outFile.close();
}


//Dumps opencl errors to the terminal
static void dump_error(const char *str, cl_int status) {
	
	printf("%s\n", str);
	printf("Error code: %d\n", status);
}



// free the resources allocated during initialization
static void freeResources() {
	
	if(newline_kernel){
    	clReleaseKernel(newline_kernel);
	}
	if(delimiter_kernel){
    	clReleaseKernel(delimiter_kernel);
  	}
	if(transform_kernel){
    	clReleaseKernel(transform_kernel);
  	}
	if(program){
    	clReleaseProgram(program);
	}
	if(queue[0]){
		clReleaseCommandQueue(queue[0]);
  	}
	if(queue[0]){
  		clReleaseCommandQueue(queue[1]);
	}
  	if(queue[0]){
		clReleaseCommandQueue(queue[2]);
	}
  	if(svm_inputBuffer){
   		clSVMFreeAltera(context,svm_inputBuffer);
  	}
  	if(svm_outputBuffer){
   		clSVMFreeAltera(context,svm_outputBuffer);
  	}
  	if(context){
   		clReleaseContext(context);
  	}
}


void cleanup() {
	//It DOES NOTHING!
	//Have to declare it due to altera Opencl
}

/////////////////////////////////////////////
// Globals
/////////////////////////////////////////////

//Kernel Names
static const char *newline_kernel_name = "FindNewLines";
static const char *delimiter_kernel_name = "FindDelimiters";
static const char *transform_kernel_name = "Transform";

//static const char* INPUT_FILEPATH = "data/go_track_trackspoints_2x.csv";
//static const char* OUTPUT_FILEPATH =  "data/go_track_trackspoints_transformed.csv";
char INPUT_FILEPATH [MAX_STR_SIZE];
char OUTPUT_FILEPATH [MAX_STR_SIZE];

// OpenCL variables
static cl_int err;                			// error code returned from OpenCL calls
static cl_platform_id platform;				// Holds the Platform ID 
static cl_device_id device;   			    // Holds the Device ID
static cl_context context;      		    // context
static cl_command_queue queue[NUM_QUEUES];  // command queue (Multiple for this imp)
static cl_program program;     			    // compute program
static cl_kernel newline_kernel;			// Kernel for finding new lines
static cl_kernel delimiter_kernel;			// Kernel for finding delimiters
static cl_kernel transform_kernel;			// Kernel for performing actual transform
static cl_int status;						// Status - used for flag warnings
static cl_int kernel_status;				// Used for kernel status warings
static cl_event kernel_event;				// Used for kernel events

//New globals
char* svm_inputBuffer;						// Input Buffer - Holds the csv file (SVM memory)
char* svm_outputBuffer;						// Output Buffer - Holds the output from the transform (SVM Memory)


int main(int argc, char *argv[])
{
    cl_uint num_platforms;
    cl_uint num_devices;
    char *inputBuffer;

    //Timing values
	double wtime_overall; //Overall execution not including reading in the buffer and writing the output
    double wtime_kernel;  //Kernel execution time - Just data
    double wtime_setup;   //Building the programing to be subtracted from overall execuiton
	
	//Input allocation values
	double wtime_SVM_inp_alloc[2];
	double wtime_SVM_outp_alloc[2];
	double wtime_SVM_map_inp_start[2];
	double wtime_SVM_map_inp_end[2];
	double wtime_SVM_map_outp_start[2];
	double wtime_SVM_map_outp_end[2];
   
	if(argc != NUM_EXPECTED_ARGS){
		printf("Too many or too few args!\n");
		return 1;
	}

    snprintf(INPUT_FILEPATH, 100, "data/series/trackspoints_size_%s.csv",argv[1]);
    snprintf(OUTPUT_FILEPATH, 100, "data/series/trackspoints_transformed_size_%s.csv", argv[1]);
    size_t inputSize = load_csv_to_buffer(INPUT_FILEPATH, &inputBuffer);
    if(inputSize == 0){
	printf("failed file read\n");
        return 1;
    }


	//Begin setup for OpenCL

	//Get platform IDs
    wtime_overall = omp_get_wtime();
	status = clGetPlatformIDs(1, &platform, &num_platforms);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetPlatformIDs.", status);
		freeResources();
		return 1;
	}

	if(num_platforms != 1) {
		printf("Found %d platforms!\n", num_platforms);
		freeResources();
		return 1;
	}

	// get the device ID
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);

	if(status != CL_SUCCESS) {
		dump_error("Failed clGetDeviceIDs.", status);
		freeResources();
		return 1;
	}

	if(num_devices != 1) {
		printf("Found %d devices!\n", num_devices);
		freeResources();
		return 1;
	}

	// create a context
	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateContext.", status);
		freeResources();
		return 1;
	}


/////////////////////////////////////////////////////////////////////////////////////////////////

	wtime_SVM_inp_alloc[0] = omp_get_wtime();
	// Set inputBuffer to SVM
	svm_inputBuffer = (char*)clSVMAllocAltera(context, CL_MEM_READ_WRITE,
							(sizeof(char)*((inputSize) + 1)), 0);
	// Transfer items to the SVM Memory
	for (size_t i = 0; i <= inputSize; i++){
		svm_inputBuffer[i] = (char)inputBuffer[i];
	}
	wtime_SVM_inp_alloc[1] = omp_get_wtime() - wtime_SVM_inp_alloc[0];
	

	//Set output buffer to SVM
	//Create an output buffer of the same size as the input to garuntee enough
	//		room
	wtime_SVM_outp_alloc[0] = omp_get_wtime();
	svm_outputBuffer = (char*)clSVMAllocAltera(context, CL_MEM_READ_WRITE, 
							(sizeof(char)*((inputSize) + 1)), 0);
	wtime_SVM_outp_alloc[1] = omp_get_wtime() - wtime_SVM_outp_alloc[0];

	//SVM Memory Check!

    if(!svm_outputBuffer || !svm_inputBuffer) {
		dump_error("Failed to allocate buffers.", status);
		freeResources();
		return 1;
    }

	//Create command queues
	for(int n = 0; n < NUM_QUEUES; ++n){
		queue[n] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);	
		if(status != CL_SUCCESS) {
			dump_error("Failed clCreateCommandQueue.", status);
			freeResources();
			return 1;
		}
	}


/////////////////////////////////////////////////////////////////////////////////////////////////

	//Load Program from binary
	wtime_setup = omp_get_wtime();
	size_t binsize = 0;
	unsigned char * binary_file = loadBinaryFile("bin/GO_TRACK_CSV_TO_CSV.aocx", &binsize);

	if(!binary_file) {
		dump_error("Failed loadBinaryFile.", status);
		freeResources();
		return 1;
	}
	program = clCreateProgramWithBinary(context, 1, &device, &binsize, (const unsigned char**)&binary_file, 
											&kernel_status, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateProgramWithBinary.", status);
		freeResources();
		return 1;
	}
	delete [] binary_file;
	
	//Build The program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	if(status != CL_SUCCESS) {
		dump_error("Failed clBuildProgram.", status);
		freeResources();
		return 1;
	}

    // create the kernels
    newline_kernel = clCreateKernel(program, "FindNewLines", &status);
    if(status != CL_SUCCESS) {
    	dump_error("Failed clCreateKernel.", status);
    	freeResources();
    	return 1;
    }
	delimiter_kernel = clCreateKernel(program, "FindDelimiters", &status);
	if(status != CL_SUCCESS) {
    	dump_error("Failed clCreateKernel.", status);
    	freeResources();
    	return 1;
    }
	transform_kernel = clCreateKernel(program, "Transform", &status);
    if(status != CL_SUCCESS) {
    	dump_error("Failed clCreateKernel.", status);
    	freeResources();
    	return 1;
    }
    wtime_setup = omp_get_wtime() - wtime_setup;
    
    // set the arguments
	
	//Arg1: Input buffer, Target: newline Kernel
	cl_ulong cl_inpSize = inputSize;
    	status = clSetKernelArgSVMPointerAltera(newline_kernel, 0, (void*)svm_inputBuffer);
    	if(status != CL_SUCCESS) {
    	dump_error("Failed set arg 0.", status);
    	return 1;
    }

	//Arg2: Number of input chars, Target: NewLine Kernel
	status = clSetKernelArg(newline_kernel, 1, sizeof(cl_ulong), &(cl_inpSize));
    if(status != CL_SUCCESS) {
    	dump_error("Failed Set arg 1.", status);
    	freeResources();
    	return 1;
    }

	//Arg1: Output buffer, Target: Transform kernel
    status = clSetKernelArgSVMPointerAltera(transform_kernel, 0, (void*)svm_outputBuffer);
    if(status != CL_SUCCESS) {
    	dump_error("Failed Set arg 2.", status);
    	freeResources();
    	return 1;
    }
	//Arg2: Number of input chars, Target Transform kernel
    status = clSetKernelArg(transform_kernel, 1, sizeof(cl_ulong), &(cl_inpSize));
    if(status != CL_SUCCESS) {
    	dump_error("Failed Set arg 3.", status);
		freeResources();
    	return 1;
    }


	wtime_SVM_map_inp_start[0] = omp_get_wtime();
    //Mapping svm mem
    status = clEnqueueSVMMap(queue[0], CL_TRUE, CL_MAP_READ, (void *)svm_inputBuffer, inputSize, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
    	dump_error("Failed clEnqueueSVMMap", status);
    	reeResources();
     	return 1;
    }
	wtime_SVM_map_inp_end[0] = omp_get_wtime() - wtime_SVM_map_inp_start[0];
  

	wtime_SVM_map_outp_start[0] = omp_get_wtime();
	status = clEnqueueSVMMap(queue[2], CL_TRUE, CL_MAP_WRITE, (void *)svm_outputBuffer, inputSize, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
    	dump_error("Failed clEnqueueSVMMap", status);
    	freeResources();
    	return 1;
    }
	wtime_SVM_map_outp_end[0] = omp_get_wtime() - wtime_SVM_map_outp_start[0];
    


	//Launch kernel
    wtime_kernel = omp_get_wtime();
    
	status = clEnqueueTask(queue[0], newline_kernel, 0, NULL, NULL); 
	if (status != CL_SUCCESS) {
		dump_error("Failed to launch kernel.", status);
    	freeResources();
    	return 1;
    }
    status = clEnqueueTask(queue[1], delimiter_kernel, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
      	dump_error("Failed to launch kernel.", status);
    	freeResources();
    	return 1;
    }
    status = clEnqueueTask(queue[2], transform_kernel, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
     	dump_error("Failed to launch kernel.", status);
      	freeResources();
      	return 1;
    }

	//Wait for kernels to finish
	clFinish(queue[0]);
	clFinish(queue[1]);
    clFinish(queue[2]);
	
	wtime_kernel = omp_get_wtime() - wtime_kernel;
	
	//Unmap memory!
	wtime_SVM_map_inp_start[1] = omp_get_wtime();
	status = clEnqueueSVMUnmap(queue[0], (void *)svm_inputBuffer, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
    	dump_error("Failed clEnqueueSVMUnmap", status);
    	freeResources();
    	return 1;
    }
	wtime_SVM_map_inp_end[1] = omp_get_wtime() - wtime_SVM_map_inp_start[1];
	

	wtime_SVM_map_outp_start[1] = omp_get_wtime();
	status = clEnqueueSVMUnmap(queue[2], (void *)svm_outputBuffer, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
    	dump_error("Failed clEnqueueSVMUnmap", status);
    	freeResources();
    	return 1;
    }
	wtime_SVM_map_outp_end[1] = omp_get_wtime() - wtime_SVM_map_outp_start[1];


	wtime_overall = omp_get_wtime() - wtime_overall;
   
	write_csv_file(OUTPUT_FILEPATH, svm_outputBuffer, inputSize);
	

	printf("%s, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", argv[1], wtime_kernel, wtime_overall, wtime_setup,
			wtime_SVM_inp_alloc[1], wtime_SVM_outp_alloc[1], wtime_SVM_map_inp_end[0], wtime_SVM_map_inp_end[1],
			wtime_SVM_map_outp_end[0], wtime_SVM_map_outp_end[1]);
    

	// Clean up!
    free(inputBuffer);
    freeResources();
    return EXIT_SUCCESS;
}

